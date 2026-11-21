# SPDX-License-Identifier: Apache-2.0
"""Hy4's staged runtime: a step runs as three compiled programs instead of one.

A whole-model program of Hy4's 78 layers is far beyond what the compiler handles in reasonable time and memory on a
node of 64 ranks (each rank compiles its own NEFF: 64 compilations of a 78-layer graph exhaust the host's 2 TB). Hy4's
layers repeat: after layers 0 and 1 (both full indexer layers, layer 0 dense) come 19 groups of four, three shared
layers and a full one. So a step runs

1. the **prologue**: the embedding, the step's addressing and layers 0 and 1;
2. the **group** program once per group, each call over its own four layers' weights and caches (the same program, so
   consecutive calls need no program-switch fence and compile once);
3. the **epilogue**: the head and the device sampler.

Each program is a functional graph over the weights it is handed (torch.func.functional_call on a template group),
compiled once per bucket, and the platform's program-switch fence meets the ranks at the two switches between them."""

from __future__ import annotations

import logging

import torch
from torch import nn

from sglang_neuron.model_executor.graph_runner import NeuronModelRuntime

logger = logging.getLogger(__name__)

# the layers the prologue runs; after them the layers repeat in groups of GROUP (three shared, then a full one)
PROLOGUE_LAYERS = 2
GROUP = 4


def group_layout(config) -> list:
    """The layer ids of each repeating group, refusing a checkpoint whose layers do not repeat so."""
    n = config.num_hidden_layers
    body = list(range(PROLOGUE_LAYERS, n))
    if len(body) % GROUP:
        raise ValueError(f"Hy4's staged runtime runs layers {PROLOGUE_LAYERS}.. in groups of {GROUP}; {len(body)} do not "
                         "split so")
    groups = [body[i:i + GROUP] for i in range(0, len(body), GROUP)]
    pattern = [(config.indexer_full[i], config.mlp_dense[i]) for i in groups[0]]
    for g in groups:
        if [(config.indexer_full[i], config.mlp_dense[i]) for i in g] != pattern:
            raise ValueError(f"Hy4's staged runtime needs every layer group alike; group {g} differs from {groups[0]}")
    if not config.indexer_full[0]:
        raise ValueError("Hy4's staged runtime needs layer 0 to be a full indexer layer")
    return groups


class _Prologue(nn.Module):
    def __init__(self, graph, token_io=None):
        super().__init__()
        self.graph = graph
        self.__dict__["token_io"] = token_io

    def forward(self, input_ids, positions, slot_mapping, block_table, block_size: int, rank, token_control=None,
                blocks: int = 0):
        g = self.graph
        if blocks:
            # a prefill context bucket: the request's first pages
            block_table = block_table[:, :blocks]
        tio = self.token_io
        if tio is not None and tio.assemble:
            # the decode's fixed-width token I/O (model_executor.token_io): this step's ids from the last step's device
            # tokens
            input_ids = tio._assembled(input_ids, token_control)[:positions.shape[0]]
        md = {"block_size": block_size, "slot_mapping": slot_mapping, "block_table_tensor": block_table}
        streams, step = g.prologue(input_ids, positions, {g.layer_names[0]: md}, rank)
        state = None
        for layer in g.layers[:PROLOGUE_LAYERS]:
            attn = layer.self_attn
            streams, state = g.run_layer(layer, attn.cache, attn.index_cache, streams, step, state, rank,
                                         replicated=layer is g.layers[0])
        return streams, step, state


class _Group(nn.Module):
    """A repeating group's layers. The template group is built anew on the meta device: its parameters must be no
    group's own, or dynamo would trace the first group's weights as constants of the program instead of as the inputs
    each call replaces."""

    def __init__(self, graph, layer_ids, template: bool = False):
        super().__init__()
        if template:
            from sglang_neuron.models.hy4.model import DecoderLayer

            with torch.device("meta"):
                layers = [DecoderLayer(graph.config, i, graph.phase, graph.group) for i in layer_ids]
        else:
            layers = [graph.layers[i] for i in layer_ids]
        self.layers = nn.ModuleList(layers)
        self.__dict__["graph"] = graph  # not a submodule: its parameters are not the group's

    def forward(self, caches, index_cache, streams, step, state, rank):
        g = self.graph
        for j, layer in enumerate(self.layers):
            streams, state = g.run_layer(layer, caches[j], index_cache if layer.self_attn.full else None,
                                         streams, step, state, rank)
        return streams, state


class _Head(nn.Module):
    """The epilogue as the device sampler's graph (models.host.SampledExecution reads these attributes)."""

    def __init__(self, graph):
        super().__init__()
        self.__dict__["graph"] = graph
        self.lm_head_tp_group = graph.lm_head_tp_group
        self.config = graph.config

    def forward(self, streams, positions, sampling_positions):
        return self.graph.epilogue(streams, sampling_positions)


class _HeadIO(nn.Module):
    """The epilogue (sampled, or the logits) and the decode's fixed-width token output."""

    def __init__(self, head, token_io=None):
        super().__init__()
        self.head = head
        self.__dict__["token_io"] = token_io

    def forward(self, streams, positions, token_ids_out=None, **sampling):
        result = self.head(streams=streams, positions=positions, **sampling)
        tio = self.token_io
        if tio is None:
            return result
        return tio._sampled(result, positions.shape[0], token_ids_out)


class StagedStep:
    """The step's three programs, called as the runtime calls its compiled graph (``**kwargs``)."""

    def __init__(self, execution_model, compile_fn):
        from sglang_neuron.model_executor.token_io import FixedDecodeTokenIO
        from sglang_neuron.models.host import SampledExecution

        token_io = None
        if isinstance(execution_model, FixedDecodeTokenIO):
            token_io, execution_model = execution_model, execution_model.model
        self.token_io = token_io
        self._warmed = set()  # the prefill step shapes whose every context bucket ran
        sampled = isinstance(execution_model, SampledExecution)
        graph = execution_model.model_graph if sampled else execution_model
        self.graph = graph
        self.groups = group_layout(graph.config)
        template = _Group(graph, self.groups[0], template=True)
        names = [name for name, _ in template.named_parameters()]
        # each group's parameters under the template's names (layers.<j>.<name>)
        self.group_params = []
        for ids in self.groups:
            group = _Group(graph, ids)
            params = dict(group.named_parameters())
            if list(params) != names:
                raise RuntimeError(f"Hy4 group {ids} has other parameters than the template group")
            self.group_params.append(params)
        self.group_caches = [[graph.layers[i].self_attn for i in ids] for ids in self.groups]
        import collections
        where = collections.Counter((str(p.device), str(p.dtype)) for params in self.group_params for p in params.values())
        where_all = collections.Counter((str(p.device), str(p.dtype)) for p in graph.parameters())
        logger.info("Hy4 staged %s: group params by (device, dtype) %s; graph params %s", graph.phase, dict(where),
                    dict(where_all))

        def group_fn(params, caches, index_cache, streams, step, state, rank):
            return torch.func.functional_call(template, params, (caches, index_cache, streams, step, state, rank))

        self.prologue = compile_fn(_Prologue(graph, token_io))
        self.group = compile_fn(group_fn, tensor_guards_only=True)
        head = _Head(graph)
        self.head = compile_fn(_HeadIO(SampledExecution(head, execution_model.mode) if sampled else head, token_io))

    # the prefill's context buckets (tokens): set by the runtime, with the host length of the step it launches next
    context_buckets = ()
    next_length = None

    def __call__(self, **kwargs):
        if self.context_buckets and self.graph.phase == "prefill":
            return self._bucketed(kwargs)
        return self._timed(**kwargs) if _TIMING else self._call(**kwargs)

    def _bucketed(self, kwargs):
        """A prefill over the smallest context bucket that holds its request (its cached prior and its new tokens):
        the programs over a table cut to that many pages, so a request pays for its own length, not the server's
        context. The first prefill (the warm-up's) runs every bucket and the whole table first, which compiles each;
        reruns of a step write the same keys, so the chosen bucket's run, last, is the step's."""
        md = kwargs["attn_metadata"][self.graph.layer_names[0]]
        table = md["block_table_tensor"]
        page = int(md["block_size"])
        length, self.next_length = self.next_length, None
        buckets = [b for b in self.context_buckets if b // page < table.shape[-1]]
        pick = next((b for b in buckets if length is not None and b >= length), None)
        chosen = [pick]
        rows = tuple(kwargs["positions"].shape)
        if rows not in self._warmed:
            # a step shape's first prefill (the warm-up's) compiles every bucket's programs
            chosen = [b for b in buckets + [None] if b != pick] + [pick]
            self._warmed.add(rows)
        if _TIMING and int(self.graph.group.coordinator.rank_in_group) == 0:
            logger.info("Hy4 prefill length %s -> context bucket %s (table %d pages)", length, chosen,
                        table.shape[-1])
        out = None
        for bucket in chosen:
            blocks = 0 if bucket is None else bucket // page
            out = self._timed(blocks=blocks, **kwargs) if _TIMING else self._call(blocks=blocks, **kwargs)
        return out

    def _timed(self, blocks: int = 0, **kwargs):
        """HY4_DEBUG_TIMING=1: each program's device time (the step synchronized after each), logged on rank 0."""
        import time

        sync = lambda t: t.to("cpu") if isinstance(t, torch.Tensor) else next(iter(t.values())).to("cpu")  # noqa: E731
        t0 = time.perf_counter()
        md = kwargs["attn_metadata"][self.graph.layer_names[0]]
        rank = kwargs["rank"]
        streams, step, state = self.prologue(kwargs["input_ids"], kwargs["positions"], md["slot_mapping"],
                                             md["block_table_tensor"], int(md["block_size"]), rank,
                                             kwargs.get("token_control"), blocks)
        sync(streams)
        t1 = time.perf_counter()
        for params, attns in zip(self.group_params, self.group_caches):
            index_cache = next(a.index_cache for a in attns if a.full)
            streams, state = self.group(params, [a.cache for a in attns], index_cache, streams, step, state, rank)
        sync(streams)
        t2 = time.perf_counter()
        head_kwargs = {k: v for k, v in kwargs.items() if k.startswith("sampling_")}
        if kwargs.get("token_ids_out") is not None:
            head_kwargs["token_ids_out"] = kwargs["token_ids_out"]
        out = self.head(streams=streams, positions=kwargs["positions"], **head_kwargs)
        sync(out)
        t3 = time.perf_counter()
        if int(self.graph.group.coordinator.rank_in_group) == 0:
            logger.info("Hy4 timing %s rows=%d: prologue %.1f ms, %d groups %.1f ms (%.2f each), head %.1f ms",
                        self.graph.phase, kwargs["positions"].shape[0], (t1 - t0) * 1e3, len(self.groups),
                        (t2 - t1) * 1e3, (t2 - t1) * 1e3 / len(self.groups), (t3 - t2) * 1e3)
        return out

    def _call(self, blocks: int = 0, **kwargs):
        md = kwargs["attn_metadata"][self.graph.layer_names[0]]
        rank = kwargs["rank"]
        streams, step, state = self.prologue(kwargs["input_ids"], kwargs["positions"], md["slot_mapping"],
                                             md["block_table_tensor"], int(md["block_size"]), rank,
                                             kwargs.get("token_control"), blocks)
        debug = _DEBUG and int(self.graph.group.coordinator.rank_in_group) == 0
        if debug:
            logger.info("Hy4 debug prologue streams %s", _norms(streams))
        for gi, (params, attns) in enumerate(zip(self.group_params, self.group_caches)):
            index_cache = next(a.index_cache for a in attns if a.full)
            streams, state = self.group(params, [a.cache for a in attns], index_cache, streams, step, state, rank)
            if debug:
                logger.info("Hy4 debug group %d streams %s state %s", gi, _norms(streams), _norms(state[0]))
        head_kwargs = {k: v for k, v in kwargs.items() if k.startswith("sampling_")}
        if kwargs.get("token_ids_out") is not None:
            head_kwargs["token_ids_out"] = kwargs["token_ids_out"]
        return self.head(streams=streams, positions=kwargs["positions"], **head_kwargs)


_DEBUG = __import__("os").environ.get("HY4_DEBUG_NORMS") == "1"
_TIMING = __import__("os").environ.get("HY4_DEBUG_TIMING") == "1"


def _norms(t):
    t = t.detach().to("cpu").float()
    return f"sum|x|={t.abs().sum().item():.6g} row0={t.reshape(t.shape[0], -1)[0].abs().sum().item():.6g}"


# the prefill's context buckets (tokens) below the server's context; HY4_PREFILL_BUCKETS="" compiles the context alone.
# A bucket of a power-of-two pages keeps a rank's keys a multiple of 512 (the NKI kernels' key tile): the server's whole
# table is one page longer
PREFILL_CONTEXT_BUCKETS = tuple(int(b) for b in __import__("os").environ.get(
    "HY4_PREFILL_BUCKETS", "32768,131072,524288").split(",") if b.strip())


class Hy4StagedRuntime(NeuronModelRuntime):
    def build_model_inputs(self, md, token_ids, bucket, *args, **kwargs):
        out = super().build_model_inputs(md, token_ids, bucket, *args, **kwargs)
        staged = getattr(self, "staged", None)
        if md.is_prefill and staged is not None:
            # the request's whole length on the host: the step picks its context bucket by it (StagedStep._bucketed)
            staged.next_length = int(md.cached_seq_len) + int(md.num_tokens)
        return out

    def build_compiled(self):
        import torch._dynamo

        from sglang_neuron.backend import envs as backend_envs
        from sglang_neuron.backend import stack
        from sglang_neuron.model_executor import program_fence
        from sglang_neuron.model_executor.graph_runner import _NO_EAGER_FALLBACK

        torch._dynamo.config.cache_size_limit = 2**62
        torch._dynamo.config.accumulated_cache_size_limit = 2**62
        options = self._compile_options()
        self.compiler_args = options.pop("compiler_args")
        cpu_mode = backend_envs.SGLANG_NEURON_CPU_MODE
        if not cpu_mode:
            options = {**options, **_NO_EAGER_FALLBACK}
        backend = stack.compile_backend_name(cpu_mode=cpu_mode)

        def compile_fn(fn, tensor_guards_only: bool = False):
            opts = dict(options)
            if tensor_guards_only:
                # the group program is called 19 times a step over other weights: only its tensors' guards matter.
                # Not the prologue's: its context bucket is an int argument (blocks) over the same full table, and
                # without that guard every bucket would rerun the first bucket's program
                opts["guard_filter_fn"] = torch.compiler.keep_tensor_guards_unsafe
            return torch.compile(fn, backend=backend, fullgraph=True, dynamic=False, options=opts)

        staged = StagedStep(self.execution_model, compile_fn)
        if self.graphs == "prefill":
            staged.context_buckets = PREFILL_CONTEXT_BUCKETS
        self.staged = staged
        logger.info("Hy4 staged runtime (%s): prologue of %d layers, %d groups of %d, epilogue", self.graphs,
                    PROLOGUE_LAYERS, len(staged.groups), GROUP)
        # three programs a step: the fence meets the ranks at the switches between them
        self.compiled = program_fence.wrap_compiled(staged, programs=3 * len(self._warm_shapes()))
        self.direct_submit = None
        self.direct_launch = None
