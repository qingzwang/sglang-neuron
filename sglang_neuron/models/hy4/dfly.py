# SPDX-License-Identifier: Apache-2.0
"""Hy4's DFly2 drafter (the checkpoint's ``model.dfly_layers.*``) and its step on the decode graph.

The drafter is block-parallel: one forward runs ``Q`` = ``dfly_block_size`` query rows of a request (the bonus token,
then ``Q - 1`` mask tokens) at positions ``p .. p + Q - 1``, ``p`` the bonus token's. Its first row's embedding is
fused with the target's final hidden state of the row before (``mtp_fusion``, slot 0). Each of its 5 layers is a
dense pre-norm layer (no iHC) whose MLA attention keeps its KV in the cache of a target layer (61, 65, 69, 73, 77) and
reads, beside the block's own keys up to its row, the keys that target layer's indexer selected for the row before
(the index share). A sequential hidden-correction chain then proposes the drafts: draft ``k`` is the target LM head's
greedy token of block row ``k + 1``'s final hidden corrected by the embedding of draft ``k - 1`` (the bonus for
``k = 0``).

On Trn2 a speculative step is one call of the decode programs over ``Q`` rows a request (the verify: the bonus and
the last step's drafts, each row sampled by the device sampler) followed by ``DFlyStep``: the acceptance (a draft is
accepted while it equals the target's token of the row before: for a greedy draft that is exactly speculative
sampling), the accepted row's hidden state and selections, and the drafter over the next block, whose positions start
after the accepted tokens. The selections are the target's own: the decode group program returns each full layer's
compacted selection (rows of this rank's cache, ``state``), and the producer layer of each draft layer ends a group.

KV context parallel (kv.py): the block's keys are written to the slots of the next verify (which overwrites them),
by the ranks that own them, and each rank attends its own selected keys and its own block keys; the ranks merge as
the target's attention does."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
from torch import nn

from sglang_neuron.models.hy4 import weights
from sglang_neuron.models.hy4.config import DTYPE
from sglang_neuron.models.transformer.model import RMSNorm
from sglang_neuron.utils.weight_loader import set_weight_loader

logger = logging.getLogger(__name__)

PREFIX = "model.dfly_layers"
ALGORITHM = "DFLASH"  # the SGLang algorithm whose scheduling (a fixed verify block a step) DFly runs under


@dataclass(frozen=True)
class DFlyConfig:
    layers: int
    block: int  # Q: the bonus row and Q - 1 drafts
    mask_token_id: int
    kv_layers: tuple  # the target layer whose KV cache each draft layer shares
    index_layers: tuple  # the target layer whose selection each draft layer reads
    intermediate_size: int
    correction_size: int


def dfly_config(hf_config):
    """The checkpoint's DFly drafter, or None. ``HY4_DFLY_LAYERS`` (comma list) remaps the target layers a draft
    layer shares (a truncated model's bring-up)."""
    n = getattr(hf_config, "dfly_num_hidden_layers", None)
    if not n:
        return None
    fixed = {"enable_ihc": False, "mlp_type": "dense", "attn_causal": True, "mtp_fusion": "slot0",
             "enable_hidden_correction": True}
    for key, want in fixed.items():
        got = getattr(hf_config, f"dfly_{key}", want)
        if got != want:
            raise ValueError(f"Hy4 DFly: dfly_{key}={got!r} is not served (only {want!r})")
    if getattr(hf_config, "dfly_context_local_window", None):
        raise ValueError("Hy4 DFly: a context-local window is not served")
    kv = tuple(int(i) for i in hf_config.dfly_target_kv_layers)
    index = getattr(hf_config, "dfly_target_index_layer", kv)
    index = tuple(int(i) for i in (index if isinstance(index, (list, tuple)) else [index] * int(n)))
    override = os.environ.get("HY4_DFLY_LAYERS", "")
    if override:
        kv = index = tuple(int(i) for i in override.split(","))
    if index != kv:
        raise ValueError(f"Hy4 DFly: index layers {index} differ from the KV layers {kv}; not served")
    return DFlyConfig(layers=int(n), block=int(hf_config.dfly_block_size), mask_token_id=int(hf_config.dfly_mask_token_id),
                      kv_layers=kv, index_layers=index,
                      intermediate_size=int(getattr(hf_config, "dfly_intermediate_size", None)
                                            or hf_config.intermediate_size),
                      correction_size=int(getattr(hf_config, "dfly_hidden_correction_intermediate_size", None)
                                          or hf_config.hidden_size))


def enabled() -> bool:
    """Whether the launch serves speculative decoding with the DFly drafter."""
    try:
        from sglang_neuron import serving_policy

        return serving_policy.current().speculative_algorithm == ALGORITHM
    except Exception:  # noqa: BLE001 - outside a launch (unit tests)
        return os.environ.get("HY4_DFLY", "0") == "1"


class _Layer(nn.Module):
    def __init__(self, config, dfly: DFlyConfig, target_layer: int, group):
        super().__init__()
        from sglang_neuron.models.hy4.model import MLP, Attention

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, target_layer, group, full=False)
        self.mlp = MLP(config, dfly.intermediate_size, group.size)


class Drafter(nn.Module):
    def __init__(self, config, dfly: DFlyConfig, group):
        super().__init__()
        from sglang_neuron.models.hy4.model import MLP

        h, tp = config.hidden_size, group.size
        self.dfly = dfly
        self.layers = nn.ModuleList([_Layer(config, dfly, i, group) for i in dfly.kv_layers])
        self.final_norm = RMSNorm(h, config.rms_norm_eps)
        # mtp_fusion: embeds + gate * eh_proj([enorm(embeds), hnorm(target_hidden)]); eh_proj's output columns shard
        self.enorm = RMSNorm(h, config.rms_norm_eps)
        self.hnorm = RMSNorm(h, config.rms_norm_eps)
        self.eh_proj = nn.Parameter(torch.empty(2 * h, h // tp, dtype=DTYPE))
        self.fusion_gate = nn.Parameter(torch.empty(h, dtype=DTYPE))
        set_weight_loader(self.eh_proj, weights.column_shard(h // tp))
        # hidden_correction: hidden + down(silu(gate) * up) of [hidden_norm(hidden), embed_norm(previous embedding)]
        self.hidden_norm = RMSNorm(h, config.rms_norm_eps)
        self.embed_norm = RMSNorm(h, config.rms_norm_eps)
        self.correction = MLP(config, dfly.correction_size, tp, inputs=2 * h)
        for p in self.fp8_parameters():
            set_weight_loader(p, weights.fp8(_loader_of(p)))

    def fp8_parameters(self):
        """The parameters the checkpoint holds in block FP8 (each mapped to its weight and scale)."""
        for layer in self.layers:
            a = layer.self_attn
            yield from (a.q_a, a.q_b, a.kv_a, a.w_uk, a.w_uv, a.gate, a.o, layer.mlp.gate_up, layer.mlp.down)
            if hasattr(a, "qkv_sh"):
                yield a.qkv_sh
        yield from (self.correction.gate_up, self.correction.down)

    def weight_mappings(self) -> dict:
        def fp8(*names):
            return [k for n in names for k in (f"{n}.weight", f"{n}.weight_scale")]

        m = {"dfly.final_norm.weight": f"{PREFIX}.final_norm.weight",
             "dfly.enorm.weight": f"{PREFIX}.mtp_fusion.enorm.weight",
             "dfly.hnorm.weight": f"{PREFIX}.mtp_fusion.hnorm.weight",
             "dfly.eh_proj": f"{PREFIX}.mtp_fusion.eh_proj.weight",
             "dfly.fusion_gate": f"{PREFIX}.mtp_fusion.gate",
             "dfly.hidden_norm.weight": f"{PREFIX}.hidden_correction.hidden_norm.weight",
             "dfly.embed_norm.weight": f"{PREFIX}.hidden_correction.embed_norm.weight",
             "dfly.correction.gate_up": fp8(f"{PREFIX}.hidden_correction.gate_proj",
                                            f"{PREFIX}.hidden_correction.up_proj"),
             "dfly.correction.down": fp8(f"{PREFIX}.hidden_correction.down_proj")}
        for i in range(len(self.layers)):
            src, dst = f"{PREFIX}.{i}", f"dfly.layers.{i}"
            a = f"{src}.self_attn"
            m.update({
                f"{dst}.input_layernorm.weight": f"{src}.input_layernorm.weight",
                f"{dst}.post_attention_layernorm.weight": f"{src}.post_attention_layernorm.weight",
                f"{dst}.self_attn.q_a": fp8(f"{a}.q_a_proj"),
                f"{dst}.self_attn.q_a_layernorm.weight": f"{a}.q_a_layernorm.weight",
                f"{dst}.self_attn.q_b": fp8(f"{a}.q_b_proj"),
                f"{dst}.self_attn.kv_a": fp8(f"{a}.kv_a_proj_with_mqa"),
                f"{dst}.self_attn.kv_a_layernorm.weight": f"{a}.kv_a_layernorm.weight",
                f"{dst}.self_attn.w_uk": fp8(f"{a}.kv_b_proj"),
                f"{dst}.self_attn.w_uv": fp8(f"{a}.kv_b_proj"),
                f"{dst}.self_attn.gate": fp8(f"{a}.linear_gate"),
                f"{dst}.self_attn.o": fp8(f"{a}.o_proj"),
                f"{dst}.self_attn.sinks": f"{a}.learnable_sink_param",
                f"{dst}.mlp.gate_up": fp8(f"{src}.mlp.gate_proj", f"{src}.mlp.up_proj"),
                f"{dst}.mlp.down": fp8(f"{src}.mlp.down_proj"),
            })
            if hasattr(self.layers[i].self_attn, "qkv_sh"):
                m[f"{dst}.self_attn.qkv_sh"] = fp8(f"{a}.q_a_proj", f"{a}.kv_a_proj_with_mqa", f"{a}.linear_gate")
        return m


def _loader_of(p):
    from sglang_neuron.utils.weight_loader import get_weight_loader

    return get_weight_loader(p)


def _core_ok(g, x) -> bool:
    from sglang_neuron.models.hy4 import model as m

    return m._NKI_ATTN_CORE and x.device.type != "cpu" and x.shape[0] <= 8 and not m._ABLATE


def accept(tokens, cand, q: int):
    """(accepted drafts [B], the new bonus token [B]) of a verify: ``tokens`` [B Q] the target's token of each row,
    ``cand`` [B Q] the verified ids (the bonus, then the drafts; -1 for no draft). Draft i is accepted while every
    draft up to it equals the target's token of the row before."""
    t = tokens.reshape(-1, q).to(torch.int64)
    c = cand.reshape(-1, q).to(torch.int64)
    match = (c[:, 1:] == t[:, :-1]).to(torch.int64)
    n = torch.cumprod(match, dim=1).sum(1)
    return n, t.gather(1, n[:, None])[:, 0]


class DFlyStep(nn.Module):
    """The program after a verify: the acceptance and the next block's drafts (the module docstring)."""

    def __init__(self, graph):
        super().__init__()
        self.__dict__["graph"] = graph

    def forward(self, streams, states, tokens, cand, base, slots, caches, page: int, rank):
        """``streams`` [B Q, n, H] the verify's final iHC streams; ``states`` each draft layer's producer selection
        (rows, keep) [B Q, cap]; ``tokens`` [B Q] its sampled tokens; ``cand`` [B Q] its ids (-1: no draft);
        ``base`` [B] each request's first verified position; ``slots`` [B, 2 Q - 1] the KV slots of positions
        base + 1 .. base + 2 Q - 1 (0: none); ``caches`` the draft layers' (target layers') latent caches ->
        {accepted [B], bonus [B], drafts [B, Q - 1]} int32."""
        from sglang_neuron import sampling
        from sglang_neuron.models.hy4.model import _write, _write_slots, rope_cos_sin

        g = self.graph
        d = g.dfly
        cfg = g.config
        q = d.dfly.block
        b = base.shape[0]
        n, bonus = accept(tokens, cand, q)
        src = torch.arange(b, device=base.device, dtype=torch.int64) * q + n  # the accepted rows
        hidden = g.norm(g.hc_head.head(streams.index_select(0, src)))  # the target's final hidden of each [B, H]
        # the block: the new bonus at base + n + 1, then the mask tokens
        offs = torch.arange(q, device=base.device, dtype=torch.int64)
        positions = (base.to(torch.int64) + n + 1)[:, None] + offs[None, :]  # [B, Q]
        block_slots = slots.to(torch.int64).gather(1, n[:, None] + offs[None, :])  # [B, Q]
        ids = torch.where(offs[None, :] == 0, bonus[:, None], torch.full_like(positions, d.dfly.mask_token_id))
        x = g.embed_tokens(ids.reshape(-1), scatter_tokens=False, rank=rank).view(b, q, -1)
        x0 = x[:, 0]
        fused = torch.cat([d.enorm(x0), d.hnorm(hidden)], dim=-1) @ d.eh_proj
        fused = g.group.all_gather(fused, dim=-1)
        x = torch.cat([(x0 + d.fusion_gate * fused)[:, None], x[:, 1:]], dim=1).reshape(b * q, -1)
        cos, sin = rope_cos_sin(positions.reshape(-1), g.inv_freq)
        write_rows, owned = _write_slots(block_slots.reshape(-1), page, g.kv_block, rank)
        # each query row's keys: its producer's selection at the accepted row, then the block's keys up to its own
        causal = offs[None, :] <= offs[:, None]  # [row, key]
        block_keep = (owned & (block_slots.reshape(-1) > 0)).view(b, 1, q) & causal[None]
        block_rows = write_rows.view(b, 1, q).expand(b, q, q)
        for layer, (rows, keep), cache in zip(d.layers, states, caches):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            if hasattr(attn, "qkv_sh") and x.device.type != "cpu" and x.shape[0] <= 8:
                # the attention input kernel over the sharded q_a / kv_a (kernels/attn_in.py), the cache rows written
                query, compact, _ = attn.project_fused(x, layer.input_layernorm.weight, cos, sin, cache, write_rows)
            else:
                query, compact, _ = attn.project(h, cos, sin)
                _write(cache, write_rows, compact)
            cap = rows.shape[-1]
            hist = rows.index_select(0, src).to(torch.int64)[:, None, :].expand(b, q, cap)
            hist_keep = keep.index_select(0, src)[:, None, :].expand(b, q, cap)
            key_rows = torch.cat([hist, block_rows], dim=-1).reshape(b * q, cap + q)
            key_keep = torch.cat([hist_keep, block_keep], dim=-1).reshape(b * q, cap + q)
            if _core_ok(g, x):
                # the attention, merge and output projection with their collectives in one kernel
                # (kernels/attn_core.py); mla_sparse needs 2 x keys a multiple of 512: padded (masked) to 256s
                kp = -key_rows.shape[1] % 256
                kr = torch.cat([key_rows, key_rows.new_zeros((key_rows.shape[0], kp))], dim=1)
                kk = torch.cat([key_keep, key_keep.new_zeros((key_keep.shape[0], kp))], dim=1)
                x = x + g._attend_core_kernel(attn, cache, query, kr, kk, h)
            else:
                attended = g._attend_sparse(attn, cache, query, key_rows, key_keep)
                x = x + attn.output(attended, h)
            x = x + g.group.all_reduce(layer.mlp(layer.post_attention_layernorm(x)))
        hs = d.final_norm(x).view(b, q, -1)
        prev = bonus
        drafts = []
        for k in range(q - 1):
            e = g.embed_tokens(prev, scatter_tokens=False, rank=rank)
            hk = hs[:, k + 1]
            hk = hk + g.group.all_reduce(d.correction(torch.cat([d.hidden_norm(hk), d.embed_norm(e)], dim=-1)))
            logits = hk.float() @ g.lm_head.weight.float().t()
            prev = sampling.greedy_sharded_logits(logits, g.lm_head_tp_group, rank, cfg.vocab_size,
                                                  top_logprobs=0)["token_ids"].to(torch.int64)
            drafts.append(prev)
        return {"accepted": n.to(torch.int32), "bonus": bonus.to(torch.int32),
                "drafts": torch.stack(drafts, dim=1).to(torch.int32)}
