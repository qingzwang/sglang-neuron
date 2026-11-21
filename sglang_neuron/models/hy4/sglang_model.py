# SPDX-License-Identifier: Apache-2.0
"""HYV4ForCausalLM: the host SGLang builds for a Hy4 checkpoint's text model (models/host.py).

Served aggregated on one node: attention TP over every rank (``--tp-size`` = the node's logical cores, 64 on a
trn2.48xlarge), the routed experts EP over the same ranks, and the KV context-parallel over them in the graphs' own
store (kv.py). The prefill and decode graphs (model.py) share one copy of the weights."""

from __future__ import annotations

import logging
import os

from torch import nn

from sglang_neuron import sglang_tree
from sglang_neuron.models.host import ModelHost
from sglang_neuron.models.hy4.config import hy4_config
from sglang_neuron.models.hy4.model import Hy4Graph
from sglang_neuron.models.hy4.runtime import Hy4StagedRuntime

logger = logging.getLogger(__name__)

# trn2 has no OCP e4m3fn; the experts' FP8 weights are held as values it reads alike (weights.py)
FP8_HLO_OPTION = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
_HLO = "--internal-hlo2tensorizer-options="


# the compiler splits a graph past this many units of MACs into modules; the prefill's programs are past its default,
# and the split hits a neuronx-cc internal error at the dense attention's mask (NCC_ILSA902), so they compile whole
NO_MODULAR_SPLIT = "--modular-flow-mac-threshold=1000000"


def with_fp8_option(args: list, options=(FP8_HLO_OPTION,)) -> list:
    out, found = [], False
    for arg in args:
        if arg.startswith(_HLO):
            found = True
            kept = [o for o in arg[len(_HLO):].split() if not o.startswith("--modular-flow-mac-threshold=")]
            arg = _HLO + " ".join(kept + [o for o in options if o not in kept])
        out.append(arg)
    if not found:
        out.append(_HLO + " ".join(options))
    return out


class HYV4ForCausalLM(ModelHost):
    name = "Hy4"
    # the graphs' attention is their own: a prefill reads its prior from the KV store through the request's table
    segmented_prefill = False
    prefill_ladder = False
    decode_context_ladder = False
    # a step as three compiled programs (runtime.py); SGLANG_NEURON_HY4_STAGED=0 compiles the whole model as one
    runtime_class = Hy4StagedRuntime if os.environ.get("SGLANG_NEURON_HY4_STAGED", "1") != "0" else None

    def check_launch(self, config) -> None:
        cfg = hy4_config(config)
        pc = sglang_tree.parallel_config()
        tp, ep = int(pc.tp_size), int(pc.ep_size or 1)
        if sglang_tree.attention_dp_size(pc) != 1:
            raise ValueError("Hy4 is served on attention TP over every rank without DP attention; launch without "
                             "--enable-dp-attention / --dp-size")
        if cfg.num_attention_heads % tp:
            raise ValueError(f"--tp-size {tp}: Hy4's {cfg.num_attention_heads} attention heads must shard evenly")
        if ep != tp:
            raise ValueError(f"--ep-size {ep}: Hy4's routed experts are spread over the attention-TP ranks; launch with "
                             f"--ep-size {tp}")
        from sglang.srt.runtime_context import get_schedule

        page = int(get_schedule().page_size or 0)
        if not page or page % tp:
            raise ValueError(f"--page-size {page}: Hy4's KV is context-parallel over the {tp} ranks; launch with a "
                             f"--page-size that is a multiple of {tp}")

    def build_graph(self, phase: str, config) -> nn.Module:
        return Hy4Graph(hy4_config(config), phase)

    def neuron_compiler_args(self, phase: str, default_args: list) -> list:
        # Both phases compile like the platform's decode (-O2, no modular-flow threshold). The prefill's group program
        # is past the modular-flow threshold, and the split hits a neuronx-cc internal error at the dense attention's
        # mask (NCC_ILSA902); unsplit it compiles, and runs no slower (measured on the 6-layer model).
        return with_fp8_option(super().neuron_compiler_args("decode", default_args),
                               (FP8_HLO_OPTION, NO_MODULAR_SPLIT) if phase == "prefill" else (FP8_HLO_OPTION,))

    def configure_kv_cache(self, configurator, *, pre_model_load_memory):
        from sglang_neuron.model_executor.runtime_init import neuron_device
        from sglang_neuron.models.hy4 import kv

        result, self.kv_store = kv.configure_kv_cache(configurator, self.model_graph.config, tp=self.tp_size,
                                                      device=neuron_device())
        return result

    def bind_kv_cache(self, pool) -> None:
        store = getattr(self, "kv_store", None)
        if store is None:
            raise RuntimeError("Hy4's KV store is built with its pools (configure_kv_cache), before the graphs bind")
        for graph in self.model_graphs.values():
            graph.bind_store(store)
        from sglang_neuron.backend import executor_cache

        executor_cache.invalidate()


EntryClass = [HYV4ForCausalLM]
