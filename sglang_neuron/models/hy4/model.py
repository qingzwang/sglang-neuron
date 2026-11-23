# SPDX-License-Identifier: Apache-2.0
"""Hy4 as the runtime's role graphs: one class serves the prefill graph (one request per forward) and the decode graph
(one token per row), over every rank of the attention TP.

Layout, at attention TP ``tp`` (64 on a trn2.48xlarge at LNC2):

- **Hidden states** are replicated: every rank holds the step's ``hc_mult`` iHC residual streams of every token.
- **Attention heads** shard over the ranks (``heads_per_rank`` each) for the projections: rank r computes its heads'
  queries, folds the key up-projection into them (absorbed MLA), and after attention applies its heads' value
  up-projection, sigmoid gate and ``o_proj`` input columns; the ranks' partial outputs are all-reduced.
- **KV** is context-parallel over the same ranks (kv.py): rank r holds the latent rows (and, on a full indexer layer,
  the index keys) of the tokens ``[r * b, (r + 1) * b)`` of every page of ``tp * b`` tokens. Each rank gathers every
  head's absorbed query (one all-gather), attends over its own keys, and the ranks merge their partial softmaxes with
  the head's learned sink counted once (an all-reduce of the maxima, one of the sums, and a reduce-scatter that leaves
  each rank its own heads).
- **Sparse attention.** A full layer's indexer scores this rank's keys for every query; the global top ``index_topk``
  of each query is found as a score threshold, by a few rounds of histogram bisection whose counts are all-reduced
  (``topk_mask``), so no index list ever crosses ranks. Each rank then attends over its own selected keys: the
  attention is the masked dense attention over the rank's share, which is exactly the sparse attention over the
  selected keys. A shared layer reuses the previous full layer's selection.
- **Experts**: rank r holds a contiguous block of ``n_routed_experts / tp``; every rank routes every token and runs
  its experts for the tokens routed to them, beside its TP shard of the shared expert; one all-reduce sums the MoE.

The step's tensors are bf16 but where noted: the iHC mixers, router and indexer scores, softmax statistics and LM head
are fp32, as the checkpoint's reference keeps them."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from sglang.srt.distributed.parallel_state import get_attn_tp_group
from torch import nn

from sglang_neuron.layers import ColumnParallelLinear
from sglang_neuron.layers.embedding import VocabDimShardedEmbedding
from sglang_neuron.mem_cache.kv_cache import KVSpec
from sglang_neuron.models.hy4 import weights
from sglang_neuron.models.hy4.config import DTYPE, Hy4Config
from sglang_neuron.models.transformer.model import RMSNorm
from sglang_neuron.parallel import collectives as neuron_collectives
from sglang_neuron.parallel import groups
from sglang_neuron.utils.checkpoints import SafetensorsCheckpoint
from sglang_neuron.utils.weight_loader import set_weight_loader

# a masked score: far below any real one, and exp(masked - finite) is exactly 0 in fp32
MASKED = -1.0e30
# the selection's bisection: rounds of bins between the selected keys' bounds; the threshold is then within
# (max - min) / bins ** rounds of the exact top-k bound, so at most the keys that close above it join the selection
SELECT_BINS = 16
SELECT_ROUNDS = 4
# up to this many rows the attention merge gathers every rank's (max, sum) in one all-gather (Attention.merge)
MERGE_GATHER_ROWS = 64
# the decode's (one all-reduce a round; its counting is over a few rows): 16 ** 4 = 256 ** 2, the counting's cost
# growing with the bins (DFly verify, full model: 16 x 4 54.5 ms of groups, 64 x 3 54.8, 256 x 2 57.3; the same
# accepts). HY4_DECODE_SELECT "bins,rounds" overrides
DECODE_SELECT_BINS, DECODE_SELECT_ROUNDS = (int(v) for v in os.environ.get("HY4_DECODE_SELECT", "16,4").split(","))
# the key tile of the prefill's attention and indexer: one tile's scores are [chunk, heads, KEY_TILE] fp32
KEY_TILE = 1024
# Sparse attention over a compacted selection: each rank attends each query over at most SPARSE_CAP of its own
# selected keys (the highest-scoring), gathered by row, instead of over all its keys under a mask. The selection
# spreads over the ranks (a page's tokens are dealt in blocks of page / tp), about index_topk / tp keys a rank, so a
# cap well above that keeps every selected key in practice; a rank holding more drops its lowest-scoring ones.
# 0 runs the masked dense attention over every key of the rank (exact, its cost grows with the graph's context).
SPARSE_CAP = int(os.environ.get("SGLANG_NEURON_HY4_SPARSE_CAP", "256"))
# the query tile of the prefill's sparse attention: one tile gathers [QUERY_TILE, SPARSE_CAP, compact] rows
QUERY_TILE = 256
# A prefill over at most this many keys a rank attends densely under the selection mask instead of compacting it: the
# masked attention over a rank's keys costs less than the compaction's top-k and the per-query gather up to here
PREFILL_DENSE_KEYS = int(os.environ.get("HY4_PREFILL_DENSE_KEYS", "4096"))
# profiling only (wrong results): HY4_ABLATE=moe,attn,hc,index skips those parts of every layer
_ABLATE = set(filter(None, os.environ.get("HY4_ABLATE", "").split(",")))


class _Group:
    """The attention-TP group's collectives, no-ops on one rank (the CPU tests)."""

    def __init__(self):
        self.coordinator = groups.get_neuron_attn_tp_group()
        self.size = int(self.coordinator.world_size)

    def all_reduce(self, x, op="sum"):
        if self.size == 1 or "ar" in _ABLATE:
            return x
        return neuron_collectives.all_reduce(self.coordinator, x, op)

    def all_gather(self, x, dim):
        if self.size == 1:
            return x
        if "ag" in _ABLATE:
            shape = list(x.shape)
            shape[dim] *= self.size
            return x.unsqueeze(dim).expand(*x.shape[:dim], self.size, *x.shape[dim:]).reshape(shape)
        return neuron_collectives.all_gather(self.coordinator, x, dim=dim)

    def reduce_scatter(self, x, dim):
        if self.size == 1:
            return x
        if "rs" in _ABLATE:
            return x.narrow(dim, 0, x.shape[dim] // self.size)
        return neuron_collectives.reduce_scatter(self.coordinator, x, dim=dim)


def rope_cos_sin(positions, inv_freq):
    """cos and sin [T, rope] of each position, computed in fp32 and rounded to bf16 as the reference applies them."""
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(DTYPE), emb.sin().to(DTYPE)


def rotate(x, cos, sin):
    """Non-interleaved RoPE over x's last dim; x [T, ..., rope], cos and sin [T, rope]."""
    shape = [cos.shape[0]] + [1] * (x.dim() - 2) + [cos.shape[1]]
    c, s = cos.view(shape), sin.view(shape)
    half = x.shape[-1] // 2
    return x * c + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * s


def topk_mask(scores, valid, k: int, group: _Group, bins: int = None, rounds: int = None):
    """The keys of each row among the ``k`` highest-scoring of the row's valid keys on every rank: ``scores`` and
    ``valid`` are [R, T] (fp32 and bool) over this rank's keys. Every valid key where a row has at most ``k``;
    otherwise the keys at or above a threshold found by bisecting the row's global score range in SELECT_ROUNDS rounds
    of SELECT_BINS bins, each round one all-reduce of the bins' counts."""
    big = 3.0e38
    hi = torch.where(valid, scores, torch.full_like(scores, -big)).amax(-1)
    lo = torch.where(valid, scores, torch.full_like(scores, big)).amin(-1)
    bounds = group.all_reduce(torch.stack([hi, -lo], dim=-1), "max")
    hi, lo = bounds[:, 0], -bounds[:, 1]
    # rows with no valid key anywhere (padding): an empty range
    lo = torch.where(lo > hi, hi, lo)
    s = torch.where(valid, scores, torch.full_like(scores, -big))
    bins = bins or SELECT_BINS
    steps = torch.arange(bins, dtype=torch.float32, device=scores.device)
    total = None
    for _ in range(rounds or SELECT_ROUNDS):
        width = (hi - lo) / bins
        edges = lo[:, None] + width[:, None] * steps[None, :]
        counts = None
        for start in range(0, s.shape[1], KEY_TILE):
            tile = s[:, start:start + KEY_TILE]
            c = (tile[:, :, None] >= edges[:, None, :]).to(torch.float32).sum(1)
            counts = c if counts is None else counts + c
        counts = group.all_reduce(counts)
        if total is None:
            # the first edge is the global minimum: every valid key counts
            total = counts[:, 0]
        # the highest edge with at least k keys at or above it (the counts fall with the edge)
        j = ((counts >= k).to(torch.float32).sum(-1) - 1).clamp(min=0)
        lo = lo + width * j
        hi = lo + width
    # logic only: a torch.where over the boolean masks hits a neuronx-cc internal error (NCC_ILSA902) once the dense
    # prefill attention consumes the mask
    return valid & ((s >= lo[:, None]) | (total <= k)[:, None])


class Attention(nn.Module):
    """Gated MLA with a learned sink per head over the context-parallel KV; on a full layer, also the DSA indexer."""

    def __init__(self, config: Hy4Config, layer_idx: int, group: _Group, full=None):
        """``full``: whether the layer runs its own indexer, by default the checkpoint's indexer_types (a DFly draft
        layer, dfly.py, runs none)."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.name = f"layers.{layer_idx}.self_attn"
        self.group = group
        tp = group.size
        if config.num_attention_heads % tp:
            raise ValueError(f"{config.num_attention_heads} attention heads do not shard over an attention TP of {tp}")
        self.heads = heads = config.num_attention_heads // tp
        hidden, latent = config.hidden_size, config.kv_lora_rank
        self.full = config.indexer_full[layer_idx] if full is None else full
        self.q_a = nn.Parameter(torch.empty(hidden, config.q_lora_rank, dtype=DTYPE))
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b = nn.Parameter(torch.empty(config.q_lora_rank, heads * config.q_head_dim, dtype=DTYPE))
        self.kv_a = nn.Parameter(torch.empty(hidden, config.compact_dim, dtype=DTYPE))
        self.kv_a_layernorm = RMSNorm(latent, config.rms_norm_eps)
        self.w_uk = nn.Parameter(torch.empty(heads, config.qk_nope_head_dim, latent, dtype=DTYPE))
        self.w_uv = nn.Parameter(torch.empty(heads, config.v_head_dim, latent, dtype=DTYPE))
        self.gate = nn.Parameter(torch.empty(hidden, heads * config.v_head_dim, dtype=DTYPE))
        self.o = nn.Parameter(torch.empty(heads * config.v_head_dim, hidden, dtype=DTYPE))
        # every head's sink: the merge counts each head's once
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads, dtype=torch.float32))
        if _QKV_SH:  # (a DFly draft layer's too: dfly.py loads it from its FP8 weights)
            # this rank's q_a / kv_a columns and gate, partition-major (kernels/attn_in.qkv_shard_layout)
            cs = config.q_lora_rank // tp + config.compact_dim // tp + heads * config.v_head_dim
            self.qkv_sh = nn.Parameter(torch.empty(weights.BLOCK, hidden // weights.BLOCK, cs, dtype=DTYPE))
            set_weight_loader(self.qkv_sh, weights.qkv_shard_loader(config, heads, tp))
        set_weight_loader(self.q_a, weights.transposed())
        set_weight_loader(self.q_b, weights.q_b_loader(config, heads))
        set_weight_loader(self.kv_a, weights.kv_a_loader(config))
        set_weight_loader(self.w_uk, weights.kv_b_loader(config, heads, values=False))
        set_weight_loader(self.w_uv, weights.kv_b_loader(config, heads, values=True))
        set_weight_loader(self.gate, weights.gate_loader(config, heads))
        set_weight_loader(self.o, weights.o_loader(config, heads))
        set_weight_loader(self.sinks, weights.as_float())
        if self.full:
            ih, idim = config.index_n_heads, config.index_head_dim
            self.idx_q = nn.Parameter(torch.empty(config.q_lora_rank, ih * idim, dtype=DTYPE))
            self.idx_k = nn.Parameter(torch.empty(hidden, idim, dtype=DTYPE))
            self.idx_k_norm_weight = nn.Parameter(torch.empty(idim, dtype=torch.float32))
            self.idx_k_norm_bias = nn.Parameter(torch.empty(idim, dtype=torch.float32))
            self.idx_w = nn.Parameter(torch.empty(hidden, ih, dtype=DTYPE))
            for p in (self.idx_q, self.idx_k, self.idx_w):
                set_weight_loader(p, weights.transposed())
            if _IDXQ_SH:
                self.idx_q_sh = nn.Parameter(torch.empty(config.q_lora_rank, ih * idim // tp, dtype=DTYPE))
                set_weight_loader(self.idx_q_sh, weights.column_shard(ih * idim // tp))
            set_weight_loader(self.idx_k_norm_weight, weights.as_float())
            set_weight_loader(self.idx_k_norm_bias, weights.as_float())
        # bound by bind_store: this layer's latent rows [rows, compact] and, on a full layer, index keys [rows, idim]
        self.cache = None
        self.index_cache = None

    # -- projections -------------------------------------------------------------------------------------------------
    def project(self, x, cos, sin):
        """(every head's absorbed query [T, heads_all, compact], the tokens' compact rows [T, compact], the query
        latent [T, q_lora])."""
        cfg = self.config
        t = x.shape[0]
        if _NKI_PROJ and x.device.type != "cpu" and t <= 32:  # only a decode has so few rows
            # the decode's replicated down-projections in one NKI kernel at the weights' DMA rate (kernels/gemv_wide.py)
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.hy4.kernels.gemv_wide import gemv_wide
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            qa, kva = wrap_nki(gemv_wide)[get_logical_nc_config()](x.t().contiguous(), self.q_a, self.kv_a)
            q_lat = self.q_a_layernorm(qa.to(DTYPE))
            lat_pe = kva.to(DTYPE)
        else:
            q_lat = self.q_a_layernorm(x @ self.q_a)
            lat_pe = x @ self.kv_a
        q = (q_lat @ self.q_b).view(t, self.heads, cfg.q_head_dim)
        q_nope, q_pe = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
        q_pe = rotate(q_pe, cos, sin)
        q_abs = torch.einsum("thd,hdr->thr", q_nope, self.w_uk)
        query = torch.cat([q_abs, q_pe], dim=-1)
        latent, k_pe = lat_pe.split([cfg.kv_lora_rank, cfg.qk_rope_head_dim], dim=-1)
        compact = torch.cat([self.kv_a_layernorm(latent), rotate(k_pe, cos, sin)], dim=-1)
        # every head's query on every rank: the attention runs over this rank's share of the keys for all heads
        query = self.group.all_gather(query, dim=1)
        return query, compact, q_lat

    def project_fused(self, x_raw, ln_w, cos, sin, cache, write_rows):
        """project(input_layernorm(x_raw)) for a decode of <= 8 rows in one NKI kernel (kernels/attn_in.py), the
        query's all-gather and the cache write (rows ``write_rows``) included."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.attn_in import attn_in_decode
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        cfg = self.config
        q, compact, q_lat, _ = wrap_nki(attn_in_decode, alias_outputs_in_place=True)[get_logical_nc_config()](
            x_raw.to(DTYPE).contiguous(), ln_w.reshape(1, -1), self.q_a, self.q_a_layernorm.weight.reshape(1, -1),
            self.q_b, self.w_uk, self.kv_a, self.kv_a_layernorm.weight.reshape(1, -1), cos.contiguous(),
            sin.contiguous(), cache, write_rows.to(torch.int32).reshape(1, -1),
            ranks=(tuple(range(self.group.size)),), eps=float(cfg.rms_norm_eps), lat_eps=float(self.q_a_layernorm.eps),
            qkv_sh=getattr(self, "qkv_sh", None))
        return q, compact, q_lat

    def index_project(self, x, q_lat, cos, sin, decode: bool = False):
        """The indexer's queries [T, ih, idim] and head weights [T, ih] (fp32), and the tokens' index keys [T, idim]."""
        cfg = self.config
        t = x.shape[0]
        ih, idim, rope = cfg.index_n_heads, cfg.index_head_dim, cfg.qk_rope_head_dim
        if decode and _IDXQ_SH:
            q = self.group.all_gather(q_lat @ self.idx_q_sh, -1).view(t, ih, idim)
        else:
            q = (q_lat @ self.idx_q).view(t, ih, idim)
        # the checkpoint's indexer rotates the last rope channels of each head
        q = torch.cat([q[..., :idim - rope], rotate(q[..., idim - rope:], cos, sin)], dim=-1)
        k = F.layer_norm((x @ self.idx_k).float(), (idim,), self.idx_k_norm_weight, self.idx_k_norm_bias,
                         cfg.rms_norm_eps).to(DTYPE)
        k = torch.cat([k[..., :idim - rope], rotate(k[..., idim - rope:], cos, sin)], dim=-1)
        # the reference keeps weights_proj in fp32
        w = (x.float() @ self.idx_w.float()) * (ih ** -0.5) * (idim ** -0.5)
        return q, w, k

    def output(self, attended, x):
        """This rank's heads' merged latents [T, heads, latent] through their value up-projection, sigmoid gate and
        o_proj columns; the ranks' partial outputs are all-reduced."""
        t = x.shape[0]
        values = torch.einsum("thr,hdr->thd", attended.to(DTYPE), self.w_uv).reshape(t, -1)
        gated = values * torch.sigmoid(x @ self.gate)
        return self.group.all_reduce(gated @ self.o)

    # -- attention over this rank's keys -----------------------------------------------------------------------------
    def merge(self, out, m, l):
        """Merge every rank's partial softmax (out [T, H, latent] fp32, unnormalized; m, l [T, H]) with each head's
        sink counted once, and leave this rank its own heads' attended latents [T, heads, latent]."""
        sinks = self.sinks[None, :]
        if m.shape[0] <= MERGE_GATHER_ROWS:
            # few rows (the decode): one all-gather of every rank's (m, l) instead of two all-reduces, latency bound
            ml = self.group.all_gather(torch.stack([m, l], dim=0)[None], dim=0)  # [ranks, 2, T, H]
            big_m = torch.maximum(ml[:, 0].amax(0), sinks)
            w = torch.exp(m - big_m)
            total = (torch.exp(ml[:, 0] - big_m[None]) * ml[:, 1]).sum(0) + torch.exp(sinks - big_m)
        else:
            big_m = self.group.all_reduce(torch.maximum(m, sinks), "max")
            w = torch.exp(m - big_m)
            total = self.group.all_reduce(w * l) + torch.exp(sinks - big_m)
        scaled = out * (w / total)[..., None]
        t, h, r = scaled.shape
        mine = self.group.reduce_scatter(scaled.reshape(t, h * r), dim=-1)
        return mine.view(t, self.heads, r)

    def merge_head_major(self, out, m, l):
        """merge over head-major partials (out [H, T, latent] fp32; m, l [H, T]), as kernels/mla_dense.py writes them:
        the reduce-scatter over the heads takes each rank's heads as one contiguous block."""
        sinks = self.sinks[:, None]
        big_m = self.group.all_reduce(torch.maximum(m, sinks), "max")
        w = torch.exp(m - big_m)
        total = self.group.all_reduce(w * l) + torch.exp(sinks - big_m)
        scaled = out * (w / total)[..., None]
        h, t, r = scaled.shape
        mine = self.group.reduce_scatter(scaled.reshape(h, t * r), dim=0)
        return mine.view(self.heads, t, r).transpose(0, 1)


def _local_rows(block_table, block: int):
    """Each table row's local cache rows [rows, blocks x block] (a -1 block reads block 0)."""
    bt = block_table.to(torch.int64).clamp(min=0)
    offs = torch.arange(block, dtype=torch.int64, device=bt.device)
    return (bt[..., None] * block + offs).reshape(*bt.shape[:-1], -1)


def _local_positions(blocks: int, page: int, block: int, rank, device):
    """The positions of this rank's tokens [blocks x block] of a table: block j's are j * page + rank * block + i."""
    j = torch.arange(blocks, dtype=torch.int64, device=device)[:, None] * page
    i = torch.arange(block, dtype=torch.int64, device=device)[None, :]
    return (j + rank.to(torch.int64).reshape(()) * block + i).reshape(-1)


def _write_slots(slots, page: int, block: int, rank):
    """The local rows of the step's token slots on this rank, and whether this rank owns each (the others write the
    reserved page 0's first row, which no real key reads)."""
    slots = slots.to(torch.int64)
    off = slots % page
    owned = (off // block) == rank.to(torch.int64).reshape(())
    rows = (slots // page) * block + off % block
    return torch.where(owned & (slots > 0), rows, torch.zeros_like(rows)), owned


def _write(cache, rows, values):
    cache[rows] = values.to(cache.dtype)


# the prefill's NKI attention and indexer kernels (kernels/); HY4_NKI=0 runs their XLA forms
_NKI = os.environ.get("HY4_NKI", "1") != "0"
_NKI_ATTN = _NKI and os.environ.get("HY4_NKI_ATTN", "1") != "0"
_NKI_INDEX = _NKI and os.environ.get("HY4_NKI_INDEX", "1") != "0"
_NKI_MOE = _NKI and os.environ.get("HY4_NKI_MOE", "1") != "0"
_MOE_DECODE_V2 = os.environ.get("HY4_MOE_DECODE_V2", "1") != "0"
_NKI_COMPACT = _NKI and os.environ.get("HY4_NKI_COMPACT", "1") != "0"
_NKI_HC_DECODE = _NKI and os.environ.get("HY4_NKI_HC_DECODE", "1") != "0"
_HC_CHAIN = os.environ.get("HY4_HC_CHAIN", "0") == "1"  # the decode's layer-boundary HC post + pre in one kernel
_NKI_ATTN_IN = _NKI and os.environ.get("HY4_NKI_ATTN_IN", "0") == "1"  # the decode attention input kernel
# its q_a / kv_a sharded over the attention TP (kernels/attn_in.py qkv_sh; the prefill keeps the replicated ones)
_QKV_SH = _NKI_ATTN_IN and os.environ.get("HY4_QKV_SH", "0") == "1"
# the decode indexer's scores in one NKI kernel a request (kernels/index_scores.py index_scores_decode): a DFly
# verify's rows share their keys, gathered once
_NKI_INDEX_DECODE = _NKI and os.environ.get("HY4_NKI_INDEX_DECODE", "0") == "1"
# the decode indexer's query projection sharded over the attention TP (an all-gather of [T, ih idim / tp] slices)
# instead of the replicated idx_q (16 MB bf16 a full layer); the prefill keeps the replicated one
_IDXQ_SH = os.environ.get("HY4_IDXQ_SH", "0") == "1"
_NKI_ATTN_CORE = _NKI and os.environ.get("HY4_NKI_ATTN_CORE", "0") == "1"  # the decode attention core kernel
_NKI_LAYER = _NKI and os.environ.get("HY4_NKI_LAYER", "0") == "1"  # a shared decode layer in one kernel
_NKI_MOE_SUB = _NKI and os.environ.get("HY4_NKI_MOE_SUB", "0") == "1"  # the decode MoE sublayer kernel
# the routed experts sharded TP-n x EP-(ranks / n) instead of EP (kernels/moe_tp.py: the decode sublayer;
# kernels/moe_prefill_tp.py: the prefill), 0: EP. Needs the decode sublayer kernel (HY4_NKI_MOE_SUB / HY4_NKI_LAYER).
_MOE_TP = int(os.environ.get("HY4_MOE_TP", "0")) if _NKI else 0
_NKI_PROJ = _NKI and os.environ.get("HY4_NKI_PROJ", "0") == "1"  # decode q_a / kv_a through kernels/gemv_wide.py
_NKI_MOE_BLOCK = _NKI and os.environ.get("HY4_NKI_MOE_BLOCK", "0") == "1"  # off: slower in the served graph so far


def _nki_attend_ok(query, rows) -> bool:
    """Whether the dense prefill attention runs as kernels/mla_dense.py: on a NeuronCore, keys in tiles of 512, and row
    blocks of 128 (tokens x heads) that split over the two programs' unrolled iterations."""
    from sglang_neuron.models.hy4.kernels.mla_dense import UNROLL

    t, h = query.shape[0], query.shape[1]
    return (_NKI_ATTN and query.device.type != "cpu" and rows.shape[0] % 512 == 0 and 128 % h == 0
            and (t * h // 128) % (2 * UNROLL) == 0)


_NKI_HC = _NKI and os.environ.get("HY4_NKI_HC", "1") != "0"
_NKI_SPARSE = _NKI and os.environ.get("HY4_NKI_SPARSE", "1") != "0"
# a multi-row decode's (DFly verify) attention through the sparse prefill kernel
_NKI_SPARSE_DECODE = _NKI_SPARSE and os.environ.get("HY4_NKI_SPARSE_DECODE", "0") == "1"  # off: no faster in the served graph (70.0 vs 70.2 ms a 7-row verify)


def _nki_sparse_ok(query, rows) -> bool:
    """Whether the sparse prefill attention runs as kernels/mla_sparse.py: row blocks of 128 (tokens x heads) whose
    tokens' slots fill whole 512-key tiles, at most 2048."""
    t, h = query.shape[0], query.shape[1]
    slots = (128 // h) * rows.shape[1] if 128 % h == 0 else 0
    return (_NKI_SPARSE and query.device.type != "cpu" and slots and slots % 512 == 0 and slots <= 2048
            and (t * h) % 256 == 0)


def _nki_hc_ok(streams) -> bool:
    """Whether a prefill's hyper-connection runs as kernels/hc.py: on a NeuronCore, in 128-token tiles over the two
    programs."""
    return _NKI_HC and streams.device.type != "cpu" and streams.dim() == 3 and streams.shape[0] % 256 == 0


def _nki_index_ok(iq, key_rows) -> bool:
    """Whether the prefill indexer scores run as kernels/index_scores.py."""
    t, ih, idim = iq.shape
    return (_NKI_INDEX and iq.device.type != "cpu" and idim == 128 and 128 % ih == 0 and t % 256 == 0
            and key_rows.shape[-1] % 512 == 0)


class MLP(nn.Module):
    """SiLU-gated, the intermediate dimension sharded over the ranks; the caller sums the partial."""

    def __init__(self, config: Hy4Config, intermediate: int, tp: int, inputs: int = None):
        """``inputs``: the input width, by default the hidden size."""
        super().__init__()
        if intermediate % tp:
            raise ValueError(f"an intermediate size of {intermediate} does not shard over {tp} ranks")
        rows = intermediate // tp
        self.gate_up = nn.Parameter(torch.empty(inputs or config.hidden_size, 2 * rows, dtype=DTYPE))
        self.down = nn.Parameter(torch.empty(rows, config.hidden_size, dtype=DTYPE))
        set_weight_loader(self.gate_up, weights.mlp_gate_up_loader(rows))
        set_weight_loader(self.down, weights.mlp_down_loader(rows))

    def forward(self, x):
        gate, up = (x @ self.gate_up).chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down


def _dequant(weight, scale):
    """A transposed block-FP8 weight [K, N] (trn2-halved) times its [K / 128, N / 128] block scales, as bf16: one NKI
    pass on a NeuronCore (kernels/dequant.py; in the XLA graph the compiler folded this into the matmul's tiling at
    ~15x the cost), torch on the host."""
    if weight.device.type == "cpu":
        k, n = weight.shape
        return (weight.to(DTYPE).view(k // weights.BLOCK, weights.BLOCK, n // weights.BLOCK, weights.BLOCK).float()
                * scale[:, None, :, None]).to(DTYPE).view(k, n)
    from sglang_neuron.backend.stack import wrap_nki
    from sglang_neuron.models.hy4.kernels.dequant import dequant_fp8_block
    from sglang_neuron.utils.hardware_config import get_logical_nc_config

    return wrap_nki(dequant_fp8_block)[get_logical_nc_config()](weight, scale)


class MoE(nn.Module):
    """The router, this rank's block of routed experts and its TP shard of the shared expert."""

    def __init__(self, config: Hy4Config, tp: int, phase: str):
        super().__init__()
        if config.n_routed_experts % tp:
            raise ValueError(f"{config.n_routed_experts} routed experts do not split over {tp} ranks")
        self.config = config
        self.num_local = config.n_routed_experts // tp
        self.phase = phase
        hidden, inner = config.hidden_size, config.moe_intermediate_size
        e, b = self.num_local, weights.BLOCK
        self.router = nn.Parameter(torch.empty(hidden, config.n_routed_experts, dtype=DTYPE))
        self.router_bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
        if not _MOE_TP:
            # transposed, the contraction first (weights.experts_*_loader): gate columns then up
            self.gate_up = nn.Parameter(torch.empty(e, hidden, 2 * inner, dtype=torch.float8_e4m3fn))
            self.gate_up_scale = nn.Parameter(torch.empty(e, hidden // b, 2 * inner // b, dtype=torch.float32))
            self.down = nn.Parameter(torch.empty(e, inner, hidden, dtype=torch.float8_e4m3fn))
            self.down_scale = nn.Parameter(torch.empty(e, inner // b, hidden // b, dtype=torch.float32))
            set_weight_loader(self.gate_up, weights.experts_gate_up_loader())
            set_weight_loader(self.gate_up_scale, weights.experts_gate_up_scale_loader())
            set_weight_loader(self.down, weights.experts_down_loader())
            set_weight_loader(self.down_scale, weights.experts_down_scale_loader())
        self.shared = MLP(config, inner * config.n_shared_experts, tp) if config.n_shared_experts else None
        self.tp = tp
        # the decode MoE sublayer kernel's (kernels/moe_sublayer.py) copies of the router and the shared gate / up,
        # partition-major (whole DMA packets)
        self.router_pm = nn.Parameter(torch.empty(weights.BLOCK, hidden * config.n_routed_experts // weights.BLOCK,
                                                  dtype=DTYPE))
        set_weight_loader(self.router_pm, weights.router_pm_loader())
        if self.shared is not None:
            rows = inner * config.n_shared_experts // tp
            self.sh_gate_up_pm = nn.Parameter(torch.empty(2, weights.BLOCK, hidden * rows // weights.BLOCK, dtype=DTYPE))
            set_weight_loader(self.sh_gate_up_pm, weights.shared_pm_loader(rows))
        if _MOE_TP:
            # each EP group's experts, this rank's slice of each (weights.experts_tp_loaders)
            self.tp_local = config.n_routed_experts // (tp // _MOE_TP)
            sw, kh = inner // _MOE_TP, hidden // b
            self.gate_up_tp = nn.Parameter(torch.empty(self.tp_local, b, kh, 2 * sw, dtype=torch.float8_e4m3fn))
            self.gate_up_scale_tp = nn.Parameter(torch.empty(self.tp_local, kh, 2, dtype=torch.float32))
            self.down_tp = nn.Parameter(torch.empty(self.tp_local, sw, hidden, dtype=torch.float8_e4m3fn))
            self.down_scale_tp = nn.Parameter(torch.empty(self.tp_local, kh, dtype=torch.float32))
            for prm, ld in zip((self.gate_up_tp, self.gate_up_scale_tp, self.down_tp, self.down_scale_tp),
                               weights.experts_tp_loaders(_MOE_TP, inner)):
                set_weight_loader(prm, ld)
        set_weight_loader(self.router, weights.transposed())
        set_weight_loader(self.router_bias, weights.as_float())

    def route(self, x):
        """Each token's top-k experts [T, k] and their weights [T, k] (fp32): sigmoid scores, the bias for the choice
        only, the chosen scores renormalized and scaled."""
        cfg = self.config
        scores = torch.sigmoid(x.float() @ self.router.float())
        _, experts = torch.topk(scores + self.router_bias[None, :], cfg.num_experts_per_tok, dim=-1)
        chosen = scores.gather(1, experts)
        chosen = chosen / (chosen.sum(-1, keepdim=True) + 1e-20) * cfg.routed_scaling_factor
        return experts, chosen

    def forward(self, x, rank, real=None):
        """``real`` [T] (1 for a real row, 0 for padding): padding rows route to no expert in the prefill (identical
        rows all choose the same experts, which would fill a local expert's blocks with them)."""
        cfg = self.config
        if (self.phase == "decode" and _NKI_MOE_BLOCK and x.device.type != "cpu" and x.shape[0] <= 32
                and cfg.num_experts_per_tok == 8 and self.shared is not None and not _ABLATE):
            return self._block_kernel(x, rank)
        if "moe_route" in _ABLATE:
            experts = torch.zeros(x.shape[0], cfg.num_experts_per_tok, dtype=torch.int64, device=x.device)
            chosen = torch.ones(x.shape[0], cfg.num_experts_per_tok, dtype=torch.float32, device=x.device)
        else:
            experts, chosen = self.route(x)
        if "moe_experts" in _ABLATE:
            return (x * chosen.sum(-1, keepdim=True).to(DTYPE)) + self.shared(x)
        if _MOE_TP:
            return self._forward_tp(x, rank, experts, chosen, real)
        first = rank.to(torch.int64).reshape(()) * self.num_local
        local = first + torch.arange(self.num_local, device=x.device)
        # each token's weight on each local expert [T, E], zero off its choices
        aff = (chosen[:, :, None] * (experts[:, :, None] == local[None, None, :]).to(torch.float32)).sum(1)
        if real is not None and self.phase != "decode":
            aff = aff * real[:, None]
        if self.phase == "decode" and x.device.type != "cpu":
            out = self._decode_kernel(x, aff)
        elif _NKI_MOE and x.device.type != "cpu" and x.shape[0] % 128 == 0:
            out = self._prefill_kernel(x, aff)
        else:
            out = self._dense(x, aff)
        if self.shared is not None:
            out = out + self.shared(x)
        return out

    def _forward_tp(self, x, rank, experts, chosen, real):
        """The TP-sharded experts outside the decode sublayer kernel: the prefill (kernels/moe_prefill_tp.py, this
        rank's slices of its EP group's experts over their routed tokens), plus the shared expert. The partial sums
        meet in the layer's all-reduce like the EP ones."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.moe_prefill import moe_blocks
        from sglang_neuron.models.hy4.kernels.moe_prefill_tp import moe_prefill_tp
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        t = x.shape[0]
        if x.device.type == "cpu":
            raise NotImplementedError("HY4_MOE_TP: the TP experts run on the device kernels only")
        first = (rank.to(torch.int64).reshape(()) // _MOE_TP) * self.tp_local
        local = first + torch.arange(self.tp_local, device=x.device)
        aff = (chosen[:, :, None] * (experts[:, :, None] == local[None, None, :]).to(torch.float32)).sum(1)
        if real is not None and self.phase != "decode":
            aff = aff * real[:, None]
        xp = x.to(DTYPE)
        pad = -t % 128
        if pad:
            # a decode batch beyond the sublayer kernel's 8 rows (e.g. 4 requests' DFly verify): padded rows route
            # to no expert
            xp = torch.cat([xp, xp.new_zeros((pad, xp.shape[1]))])
            aff = torch.cat([aff, aff.new_zeros((pad, aff.shape[1]))])
        tp_ = t + pad
        tok, tok_out, a, bexp, nblk = moe_blocks(aff)
        out = wrap_nki(moe_prefill_tp)[get_logical_nc_config()](
            xp, tok, tok_out, a, bexp, ((nblk + 1) // 2).to(torch.int32), self.gate_up_tp,
            self.gate_up_scale_tp, self.down_tp, self.down_scale_tp, limit=float(self.config.swiglu_limit or 3.0e38))
        out = out.view(2, tp_, -1).sum(0)[:t].to(DTYPE)
        if self.shared is not None:
            out = out + self.shared(x)
        return out

    def _dense(self, x, aff):
        """Every local expert over every token, each weight dequantized once (the prefill)."""
        cfg = self.config
        inner, limit = cfg.moe_intermediate_size, cfg.swiglu_limit
        out = torch.zeros(x.shape[0], cfg.hidden_size, dtype=torch.float32, device=x.device)
        for e in range(self.num_local):
            gu = (x @ _dequant(self.gate_up[e], self.gate_up_scale[e])).float()
            gate, up = gu[:, :inner], gu[:, inner:]
            if limit:
                gate = gate.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            h = (F.silu(gate) * up).to(DTYPE)
            out = out + (h @ _dequant(self.down[e], self.down_scale[e])).float() * aff[:, e:e + 1]
        return out.to(DTYPE)

    def sublayer_experts(self, rank):
        """(base, gate_up, gate_up_scale, down, down_scale) the decode sublayer kernel takes: the EP layout, or with
        HY4_MOE_TP the rank's EP group's TP slices (base: the group's first expert)."""
        if _MOE_TP:
            base = ((rank.to(torch.int32) // _MOE_TP).reshape(1, 1) * self.tp_local).contiguous()
            return base, self.gate_up_tp, self.gate_up_scale_tp, self.down_tp, self.down_scale_tp
        base = (rank.to(torch.int32).reshape(1, 1) * self.num_local).contiguous()
        return base, self.gate_up, self.gate_up_scale, self.down, self.down_scale

    def sublayer(self, x, ln_w, rank):
        """The decode MoE sublayer in one NKI kernel (kernels/moe_sublayer.py): RMSNorm, routing, shared and routed
        experts and the all-reduce; the all-reduced output [T, H] bf16."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.moe_sublayer import moe_sublayer_decode
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        cfg = self.config
        return wrap_nki(moe_sublayer_decode)[get_logical_nc_config()](
            x.to(DTYPE).contiguous(), ln_w.reshape(1, -1), self.router_pm, self.router_bias.reshape(1, -1),
            *self.sublayer_experts(rank), self.sh_gate_up_pm, self.shared.down,
            ranks=(tuple(range(self.tp)),), top_k=cfg.num_experts_per_tok, scaling=float(cfg.routed_scaling_factor),
            limit=float(cfg.swiglu_limit or 3.0e38), eps=float(cfg.rms_norm_eps))

    def _block_kernel(self, x, rank):
        """The whole decode block (router, choice, routed experts, shared expert) in one NKI kernel
        (kernels/moe_block.py): the giga kernel's MoE block."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.moe_block import moe_block_decode, shared_gate_up_layout, tile_rows
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        cfg = self.config
        base = (rank.to(torch.int32).reshape(1, 1) * self.num_local).contiguous()
        out = wrap_nki(moe_block_decode)[get_logical_nc_config()](
            x.to(DTYPE).t().contiguous(), self.router, self.router_bias.reshape(1, -1), base,
            tile_rows(x.shape[0], x.device), self.gate_up, self.gate_up_scale, self.down, self.down_scale,
            shared_gate_up_layout(self.shared.gate_up), self.shared.down, top_k=cfg.num_experts_per_tok,
            scaling=float(cfg.routed_scaling_factor), limit=float(cfg.swiglu_limit or 3.0e38))
        return out.sum(0).to(DTYPE)

    def _prefill_kernel(self, x, aff):
        """Each local expert over its routed tokens only, in one NKI kernel (kernels/moe_prefill.py)."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.moe_prefill import moe_blocks, moe_prefill_fp8
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        t = x.shape[0]
        tok, tok_out, a, bexp, nblk = moe_blocks(aff)
        out = wrap_nki(moe_prefill_fp8)[get_logical_nc_config()](
            x.to(DTYPE), tok, tok_out, a, bexp, nblk, self.gate_up, self.gate_up_scale, self.down, self.down_scale,
            limit=float(self.config.swiglu_limit or 3.0e38))
        return out.view(2, t, -1).sum(0).to(DTYPE)

    def _decode_kernel(self, x, aff):
        """The hit local experts only, in one NKI kernel (kernels/moe_decode.py)."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.moe_decode import moe_decode_fp8
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        hit = (aff > 0).any(0).to(torch.float32)  # [E]
        # the hit experts first (a stable order: hit, then the expert id)
        key = hit * self.num_local - torch.arange(self.num_local, device=x.device, dtype=torch.float32) / 2
        order = torch.topk(key, self.num_local).indices.to(torch.int32).reshape(1, -1)
        n_hit = hit.sum().to(torch.int32).reshape(1, 1)
        limit = float(self.config.swiglu_limit or 3.0e38)
        args = (x.t().contiguous(), aff.t().contiguous(), order, n_hit, self.gate_up, self.gate_up_scale, self.down,
                self.down_scale)
        if _MOE_DECODE_V2:
            # token-major: the block scales in the activations, each weight block one matmul (kernels/moe_decode2.py)
            from sglang_neuron.models.hy4.kernels.moe_decode2 import TILE, moe_decode_fp8_ct, moe_decode_fp8_tm

            # up to TILE tokens on the tensor engine's column tiles, four matmuls at once
            kernel = moe_decode_fp8_ct if x.shape[0] <= TILE else moe_decode_fp8_tm
            return wrap_nki(kernel)[get_logical_nc_config()](*args, limit=limit).sum(0).to(DTYPE)
        parts = wrap_nki(moe_decode_fp8)[get_logical_nc_config()](*args, limit=limit)
        return parts.sum(0).t().to(DTYPE)


class HyperConnection(nn.Module):
    """iHC: the ``hc_mult`` residual streams mixed into one sublayer input, and the sublayer's output mixed back."""

    def __init__(self, config: Hy4Config, rows: int):
        super().__init__()
        self.config = config
        width = config.hc_mult * config.hidden_size
        self.fn = nn.Parameter(torch.empty(rows, width, dtype=torch.float32))
        self.base = nn.Parameter(torch.empty(rows, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(2 if rows == 2 * config.hc_mult else 1, dtype=torch.float32))

    def _mixes(self, streams):
        flat = streams.flatten(1).float()
        norm = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        return (flat @ self.fn.t()) * norm

    def pre(self, streams, replicated: bool = False):
        """(the sublayer input [T, H], the post weights [T, hc_mult] (fp32)). ``replicated``: every stream is the same
        (the embedding's, before the first layer), so the mixes are one stream's against the streams' summed columns:
        over the broadcast streams the compiler lowers the mixes' matmul to a vector-engine reduction (~40 ms a 1K-token
        prefill)."""
        cfg = self.config
        n = cfg.hc_mult
        if not replicated and _nki_hc_ok(streams):
            return self._pre_kernel(streams)
        if replicated:
            x0 = streams[:, 0].float()
            norm = torch.rsqrt(x0.square().mean(-1, keepdim=True) + cfg.rms_norm_eps)
            mixes = (x0 @ self.fn.view(self.fn.shape[0], n, -1).sum(1).t()) * norm
        else:
            mixes = self._mixes(streams)
        pre = torch.sigmoid(mixes[:, :n] * self.scale[0] + self.base[:n]) + cfg.hc_eps
        post = cfg.hc_magnitude * torch.sigmoid(mixes[:, n:] * self.scale[1] + self.base[n:]) + cfg.hc_eps
        if replicated:
            return (streams[:, 0].float() * pre.sum(-1, keepdim=True)).to(DTYPE), post
        return (pre[..., None] * streams.float()).sum(1).to(DTYPE), post

    def _pre_kernel(self, streams):
        """pre in one NKI kernel (kernels/hc.py): the streams read once, every fp32 intermediate in SBUF."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.hc import hc_pre
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        cfg = self.config
        n, rows = cfg.hc_mult, self.fn.shape[0]
        fnT = self.fn.t()
        hi = fnT.to(DTYPE)
        lo = (fnT - hi.float()).to(DTYPE)
        scale = torch.cat([self.scale[0].expand(n), self.scale[-1].expand(rows - n)]) if rows == 2 * n else \
            self.scale[0].expand(rows)
        x, post = wrap_nki(hc_pre)[get_logical_nc_config()](
            streams.reshape(streams.shape[0], -1), torch.cat([hi, lo], 1).contiguous(),
            torch.stack([scale, self.base]).contiguous(), rms_eps=cfg.rms_norm_eps, hc_eps=cfg.hc_eps,
            magnitude=cfg.hc_magnitude)
        return x, post

    def affine(self):
        """[2, R] fp32: each mixing row's scale and base, as the decode HC kernels take them."""
        n, rows = self.config.hc_mult, self.fn.shape[0]
        scale = torch.cat([self.scale[0].expand(n), self.scale[-1].expand(rows - n)]) if rows == 2 * n else \
            self.scale[0].expand(rows)
        return torch.stack([scale, self.base]).contiguous()

    def post_pre_decode(self, streams, out, post):
        """post(streams, out, post), then this mixer's pre, in one decode kernel: (streams', x, post')."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.hc_decode import decode_constants, hc_decode, hc_decode_batched
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        cfg = self.config
        n, rows = cfg.hc_mult, self.fn.shape[0]
        t = streams.shape[0]
        # several tokens a program (a DFly verify's rows): side by side in one pass (75 -> 57 us at 7 rows)
        kernel = hc_decode_batched if t > 2 else hc_decode
        scale = torch.cat([self.scale[0].expand(n), self.scale[-1].expand(rows - n)]) if rows == 2 * n else \
            self.scale[0].expand(rows)
        ones, place = decode_constants(n, streams.device)
        new, x, post_next = wrap_nki(kernel)[get_logical_nc_config()](
            streams.reshape(t, -1), out.to(DTYPE), post, self.fn, torch.stack([scale, self.base]).contiguous(), ones,
            place, rms_eps=cfg.rms_norm_eps, hc_eps=cfg.hc_eps, magnitude=cfg.hc_magnitude)
        return new.reshape(streams.shape), x, post_next

    @staticmethod
    def post(streams, out, post):
        if _nki_hc_ok(streams):
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.hy4.kernels.hc import hc_post
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            t = streams.shape[0]
            return wrap_nki(hc_post)[get_logical_nc_config()](streams.reshape(t, -1), out.to(DTYPE),
                                                              post).reshape(streams.shape)
        return (post[..., None] * out.float()[:, None, :] + streams.float()).to(DTYPE)

    def head(self, streams):
        cfg = self.config
        if _nki_hc_ok(streams):
            return self._pre_kernel(streams)[0]
        pre = torch.sigmoid(self._mixes(streams) * self.scale[0] + self.base) + cfg.hc_eps
        return (pre[..., None] * streams.float()).sum(1).to(DTYPE)


class DecoderLayer(nn.Module):
    def __init__(self, config: Hy4Config, layer_idx: int, phase: str, group: _Group):
        super().__init__()
        self.attn_hc = HyperConnection(config, 2 * config.hc_mult)
        self.mlp_hc = HyperConnection(config, 2 * config.hc_mult)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config, layer_idx, group)
        self.dense = config.mlp_dense[layer_idx]
        self.mlp = (MLP(config, config.intermediate_size, group.size) if self.dense
                    else MoE(config, group.size, phase))


class Hy4Graph(nn.Module):
    """One role graph (``phase`` "prefill" or "decode")."""

    def __init__(self, config: Hy4Config, phase: str, hf_config=None, dfly=None):
        """``dfly``: the checkpoint's DFly drafter (dfly.DFlyConfig) where the launch serves speculative decoding."""
        super().__init__()
        if phase not in ("prefill", "decode"):
            raise ValueError(f"a role graph is a prefill or a decode, not {phase!r}")
        self.config = config
        self.phase = phase
        self.group = group = _Group()
        self.tp_group = get_attn_tp_group()
        self.lm_head_tp_group = coordinator = group.coordinator
        self.embed_tokens = VocabDimShardedEmbedding(config.vocab_size, config.hidden_size, dtype=DTYPE,
                                                     tp_group=coordinator.device_group)
        self.layers = nn.ModuleList([DecoderLayer(config, i, phase, group) for i in range(config.num_hidden_layers)])
        self.hc_head = HyperConnection(config, config.hc_mult)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, dtype=DTYPE, gather_output=True,
                                            tp_group=coordinator.device_group)
        set_weight_loader(self.embed_tokens.weight,
                          weights.vocab_rows_loader(self.embed_tokens.vocab_size_per_rank, config.hidden_size))
        set_weight_loader(self.lm_head.weight,
                          weights.vocab_rows_loader(self.lm_head.out_features_per_rank, config.hidden_size))
        rope = config.qk_rope_head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rope, 2, dtype=torch.float32) / rope))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # the runtime keys each layer's attention metadata by these names; Hy4's layer_types name its sparse attention,
        # which the runtime's derivation does not know
        self.layer_names = [f"layers.{i}.self_attn" for i in range(config.num_hidden_layers)]
        self.sliding_windows = [None] * config.num_hidden_layers
        # the KV store's page geometry (kv.py), set by bind_store
        self.kv_block = None
        self.sparse_cap = SPARSE_CAP
        if dfly is not None:
            from sglang_neuron.models.hy4.dfly import Drafter

            self.dfly = Drafter(config, dfly, group)
        else:
            self.dfly = None

    # -- the KV store ------------------------------------------------------------------------------------------------
    def bind_store(self, store) -> None:
        self.kv_block = int(store.block)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            attn.cache = store.latent[i]
            attn.index_cache = store.index[self.config.index_slot(i)] if attn.full else None

    def get_kv_spec(self) -> KVSpec:
        # no layer reads SGLang's pool: the KV lives in the graph's own context-parallel store (kv.py)
        return KVSpec(layers=[])

    def bind_kv_cache(self, kv_caches: dict) -> None:
        pass

    # -- the step ----------------------------------------------------------------------------------------------------
    # A step is the prologue (the embedding and the step's addressing), every layer (run_layer) and the epilogue (the
    # head). forward runs them as one program; the staged runtime (runtime.py) runs them as separately compiled
    # programs, the repeating layer groups as one program called once per group.
    @torch.no_grad()
    def forward(self, input_ids, positions, attn_metadata, sampling_positions, rank):
        streams, step = self.prologue(input_ids, positions, attn_metadata, rank)
        state = None
        for layer in self.layers:
            attn = layer.self_attn
            streams, state = self.run_layer(layer, attn.cache, attn.index_cache, streams, step, state, rank,
                                            replicated=layer is self.layers[0])
        return self.epilogue(streams, sampling_positions)

    def prologue(self, input_ids, positions, attn_metadata, rank):
        """(the embedded iHC streams [T, hc_mult, H], the step's addressing: (cos, sin, the tokens' write rows, each
        table row's local key rows [R, keys], which of them each query may attend [R, keys]))."""
        cfg = self.config
        md = attn_metadata[self.layer_names[0]]
        page, block = int(md["block_size"]), self.kv_block
        if block is None:
            raise RuntimeError("the Hy4 graph has no KV store bound (Hy4 host: bind_kv_cache)")
        cos, sin = rope_cos_sin(positions, self.inv_freq)
        hidden = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        streams = hidden[:, None, :].expand(-1, cfg.hc_mult, -1).contiguous()
        write_rows, _ = _write_slots(md["slot_mapping"], page, block, rank)
        table = md["block_table_tensor"]
        key_rows = _local_rows(table, block)  # [rows, keys]
        key_pos = _local_positions(table.shape[-1], page, block, rank, table.device)  # [keys]
        if self.phase == "decode":
            # row b attends the keys at or before its position, over its own table
            valid = (key_pos[None, :] <= positions.to(torch.int64)[:, None]) & (
                table.to(torch.int64).repeat_interleave(block, dim=1) >= 0)
        else:
            valid = key_pos[None, :] <= positions.to(torch.int64)[:, None]
            valid = valid & (table[0].to(torch.int64).repeat_interleave(block) >= 0)[None, :]
        # as uint8: the step crosses program boundaries (runtime.py), and a boolean program input of this shape hits a
        # neuronx-cc internal error (NCC_ILSA902) in the dense prefill attention
        # the step's real rows (a padding row's slot is 0: page 0 is reserved), as the MoE's routing mask
        real = (md["slot_mapping"].to(torch.int64) > 0).to(torch.float32)
        return streams, (cos, sin, write_rows, key_rows, valid.to(torch.uint8), real)

    def run_layer(self, layer, cache, index_cache, streams, step, state, rank, replicated: bool = False,
                  pending=None, defer: bool = False):
        """One decoder layer over ``streams``; ``state`` is the selection of the last full layer (a shared layer's),
        returned updated by a full layer: (compacted rows, keep) with SPARSE_CAP, else (mask,).

        ``pending`` (out, post): the last layer's MLP output whose hyper-connection post update is still to apply,
        fused here with this layer's pre (hc_decode); ``defer``: return this layer's MLP output that way, as
        (streams, state, (out, post)), instead of applying it (the decode, between the layers of a group)."""
        cos, sin, write_rows, key_rows, valid, real = step
        valid = valid != 0
        attn = layer.self_attn
        if (_NKI_LAYER and self.phase == "decode" and not attn.full and not layer.dense and state is not None
                and len(state) == 2 and streams.shape[0] <= 8 and streams.device.type != "cpu" and not _ABLATE):
            # the whole layer in one kernel (kernels/layer_decode.py)
            streams, out, post = self._layer_kernel(layer, cache, streams, step, state, rank, pending)
            if defer:
                return streams, state, (out, post)
            return HyperConnection.post(streams, out, post), state
        if pending is not None:
            streams, x, post = layer.attn_hc.post_pre_decode(streams, pending[0], pending[1])
        elif "hc" in _ABLATE:
            x, post = streams[:, 0], None
        else:
            x, post = layer.attn_hc.pre(streams, replicated)
        if _NKI_ATTN_IN and self.phase == "decode" and x.shape[0] <= 8 and x.device.type != "cpu" and not _ABLATE:
            # the norm, the projections, RoPE and the query's all-gather in one kernel (kernels/attn_in.py); the
            # graph's own normed input feeds the indexer and the output gate only
            query, compact, q_lat = attn.project_fused(x, layer.input_layernorm.weight, cos, sin, cache, write_rows)
            x = layer.input_layernorm(x)
        else:
            x = layer.input_layernorm(x)
            query, compact, q_lat = attn.project(x, cos, sin)
            _write(cache, write_rows, compact)
        if attn.full and "index" in _ABLATE:
            if state is None:
                rows = key_rows[:, :self.sparse_cap] if self.phase == "decode" else key_rows[0][:self.sparse_cap][None].expand(x.shape[0], -1)
                state = (rows, torch.ones_like(rows, dtype=torch.bool))
        elif attn.full:
            iq, iw, ik = attn.index_project(x, q_lat, cos, sin, decode=self.phase == "decode")
            _write(index_cache, write_rows, ik)
            # (not in the prologue's layer 0: its program got slower with it, 6.3 -> ~10 ms at 512K)
            selected, scores = self._select(index_cache, iq, iw, key_rows, valid, kernel_ok=attn.layer_idx > 0)
            # the dense prefill's mask crosses program boundaries (runtime.py) as uint8: a boolean program output of
            # this shape hits a neuronx-cc internal error (NCC_ILSA902)
            # the dense prefill carries the selection as an additive score bias (arithmetic only: selects over the
            # mask, in and across programs, hit a neuronx-cc internal error, NCC_ILSA902)
            state = self._compact(selected, scores, key_rows) if self._sparse(key_rows) else (
                (selected.to(torch.float32) - 1.0) * -MASKED,)
        if (_NKI_ATTN_CORE and self.phase == "decode" and self._sparse(key_rows) and x.shape[0] <= 8
                and x.device.type != "cpu" and not _ABLATE and attn.heads == 1):
            # the attention, its merge and the output projection with their three collectives in one kernel
            out = self._attend_core_kernel(attn, cache, query, *state, x)
        else:
            if "attn" in _ABLATE:
                attended = query[:, :attn.heads, :self.config.kv_lora_rank].float()
            elif self._sparse(key_rows):
                attended = self._attend_sparse(attn, cache, query, *state)
            else:
                attended = self._attend(attn, cache, query, key_rows, state[0])
            out = attn.output(attended, x)
        if "hc" in _ABLATE:
            streams = streams + out[:, None, :]
            x = streams[:, 0]
        elif self.phase == "decode" and _NKI_HC_DECODE and streams.device.type != "cpu":
            # the attention's post and the MLP's pre in one NKI kernel (kernels/hc_decode.py)
            streams, x, post = layer.mlp_hc.post_pre_decode(streams, out, post)
        else:
            streams = HyperConnection.post(streams, out, post)
            x, post = layer.mlp_hc.pre(streams)
        if (_NKI_MOE_SUB and self.phase == "decode" and not layer.dense and x.shape[0] <= 8
                and x.device.type != "cpu" and not _ABLATE and layer.mlp.shared is not None):
            # norm, MoE and its all-reduce in one kernel (kernels/moe_sublayer.py)
            out = layer.mlp.sublayer(x, layer.post_attention_layernorm.weight, rank)
        else:
            x = layer.post_attention_layernorm(x)
            if "moe" in _ABLATE and not layer.dense:
                out = layer.mlp.shared(x)
            else:
                out = layer.mlp(x) if layer.dense else layer.mlp(x, rank, real)
            out = self.group.all_reduce(out)
        if defer:
            return streams, state, (out, post)
        streams = streams + out[:, None, :] if "hc" in _ABLATE else HyperConnection.post(streams, out, post)
        return streams, state

    def _layer_kernel(self, layer, cache, streams, step, state, rank, pending):
        """A shared decode layer in one NKI kernel: (streams before the MLP's post update, the MLP output, its post)."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.hc_decode import decode_constants
        from sglang_neuron.models.hy4.kernels.layer_decode import layer_decode
        from sglang_neuron.models.hy4.kernels.mla_dense import onehot_rows
        from sglang_neuron.models.hy4.kernels.mla_sparse import block_slots
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        cfg = self.config
        cos, sin, write_rows, key_rows, valid, real = step
        rows, keep = state
        attn, moe = layer.self_attn, layer.mlp
        t = streams.shape[0]
        n = cfg.hc_mult
        if pending is None:
            pending = (streams.new_zeros((t, cfg.hidden_size)), torch.zeros(t, n, dtype=torch.float32,
                                                                              device=streams.device))
        pad = -t % 4
        if pad:
            rows = torch.cat([rows, rows.new_zeros((pad, rows.shape[1]))])
            keep = torch.cat([keep, keep.new_zeros((pad, keep.shape[1]))])
        slots, bias = block_slots(rows, keep, cfg.num_attention_heads, cache.shape[0])
        ones, place = decode_constants(n, streams.device)
        new, out, post, _ = wrap_nki(layer_decode, alias_outputs_in_place=True)[get_logical_nc_config()](
            streams.reshape(t, -1), pending[0].to(DTYPE), pending[1], layer.attn_hc.fn, layer.attn_hc.affine(), layer.mlp_hc.fn,
            layer.mlp_hc.affine(), ones, place,
            layer.input_layernorm.weight.reshape(1, -1), attn.q_a, attn.q_a_layernorm.weight.reshape(1, -1), attn.q_b,
            attn.w_uk, attn.kv_a, attn.kv_a_layernorm.weight.reshape(1, -1), attn.gate, cos.contiguous(),
            sin.contiguous(), cache, write_rows.to(torch.int32).reshape(1, -1), slots, bias,
            onehot_rows(cfg.num_attention_heads, streams.device), attn.sinks.reshape(1, -1), attn.w_uv[0], attn.o,
            layer.post_attention_layernorm.weight.reshape(1, -1), moe.router_pm, moe.router_bias.reshape(1, -1),
            *moe.sublayer_experts(rank), moe.sh_gate_up_pm, moe.shared.down,
            ranks=(tuple(range(self.group.size)),), eps=float(cfg.rms_norm_eps), hc_eps=float(cfg.hc_eps),
            magnitude=float(cfg.hc_magnitude), scale=float(cfg.softmax_scale), top_k=cfg.num_experts_per_tok,
            scaling=float(cfg.routed_scaling_factor), limit=float(cfg.swiglu_limit or 3.0e38), t_pad=t + pad,
            qkv_sh=getattr(attn, "qkv_sh", None), tag=f"_l{attn.layer_idx}")
        return new.reshape(streams.shape), out, post

    def defers_hc(self, streams) -> bool:
        """Whether a group's layers hand their MLP's post update to the next layer's pre (one hc_decode kernel)."""
        return _HC_CHAIN and self.phase == "decode" and _NKI_HC_DECODE and streams.device.type != "cpu" and not _ABLATE

    def epilogue(self, streams, sampling_positions):
        """The sampled rows' logits [rows, vocab / tp] (fp32: enable_lm_head_fp32)."""
        streams = streams.index_select(0, sampling_positions)
        hidden = self.norm(self.hc_head.head(streams))
        return hidden.float() @ self.lm_head.weight.float().t()

    def _sparse(self, key_rows) -> bool:
        """Whether the attention runs over the compacted selection (else masked over every key of the rank)."""
        if not self.sparse_cap:
            return False
        return self.phase == "decode" or key_rows.shape[-1] > PREFILL_DENSE_KEYS

    def _select(self, cache, iq, iw, key_rows, valid, kernel_ok: bool = True):
        """The selection mask [T, keys] of a full layer over this rank's keys, and their scores."""
        cfg = self.config
        q = self.dfly.dfly.block if self.dfly is not None else 0
        if (kernel_ok and self.phase == "decode" and _NKI_INDEX_DECODE and q and iq.shape[0] % q == 0 and iq.device.type != "cpu"
                and key_rows.shape[-1] % 512 == 0 and (q * iq.shape[1]) <= 256):
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.hy4.kernels.index_scores import index_scores_decode, placement_mask_decode
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            t, ih, idim = iq.shape
            tp = 256 // ih  # the rows padded to two 128-row query blocks
            mask = placement_mask_decode(ih, tp, iq.device)
            parts = []
            for b in range(t // q):  # each request's rows (dfly_worker repeats its table over them)
                qb = torch.cat([iq[b * q:(b + 1) * q], iq.new_zeros(tp - q, ih, idim)]).reshape(tp * ih, idim)
                wb = torch.cat([iw[b * q:(b + 1) * q], iw.new_zeros(tp - q, ih)]).to(torch.float32).reshape(-1, 128)
                keys = cache[key_rows[b * q]]  # [K, idim]
                parts.append(wrap_nki(index_scores_decode)[get_logical_nc_config()](
                    qb.t().contiguous(), keys, wb, mask)[:q])
            scores = torch.cat(parts) if len(parts) > 1 else parts[0]
        elif self.phase == "decode":
            keys = cache[key_rows].float()  # [B, keys, idim]
            s = torch.einsum("bhd,bkd->bhk", iq.float(), keys).relu()
            # the head weighting as a multiply and a sum over the heads: as an einsum it lowers to a batched matmul of
            # one column per row, which the compiler feeds through layout DMAs (most of a full layer's prefill time)
            scores = (s * iw[:, :, None]).sum(1)
        elif _nki_index_ok(iq, key_rows):
            # the scores in one NKI kernel (kernels/index_scores.py): no per-head product leaves the core
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.hy4.kernels.index_scores import index_scores, placement_mask
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            keys = cache[key_rows[0]]  # [K, idim]
            t, ih, idim = iq.shape
            scores = wrap_nki(index_scores)[get_logical_nc_config()](
                iq.reshape(t * ih, idim).t().contiguous(), keys.t().contiguous(), iw.reshape(-1, 128),
                placement_mask(ih, iq.device))
        else:
            rows = key_rows[0]
            parts = []
            for start in range(0, rows.shape[0], KEY_TILE):
                keys = cache[rows[start:start + KEY_TILE]].float()
                s = torch.einsum("thd,kd->thk", iq.float(), keys).relu()
                parts.append((s * iw[:, :, None]).sum(1))
            scores = torch.cat(parts, dim=1)
        if self.phase == "decode":
            # a decode step's collectives are latency bound: the same resolution in two rounds of wider bins
            return topk_mask(scores, valid, cfg.index_topk, self.group, DECODE_SELECT_BINS, DECODE_SELECT_ROUNDS), scores
        return topk_mask(scores, valid, cfg.index_topk, self.group), scores

    def _compact(self, selected, scores, key_rows):
        """(the local rows [R, cap] of each row's highest-scoring selected keys on this rank, whether each is one
        [R, cap]); a row with fewer selected keys pads with unselected ones, masked."""
        cap = min(self.sparse_cap, scores.shape[-1])
        if self.phase == "decode" and _NKI_COMPACT and scores.device.type != "cpu":
            # the selection itself, compacted on the GpSimd engine (kernels/decode_compact.py): no top-k over the keys
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.hy4.kernels.decode_compact import decode_compact
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            rows, keep = wrap_nki(decode_compact)[get_logical_nc_config()](
                selected.to(torch.float32), key_rows.to(torch.int32), cap=cap, pad_row=0)
            return rows, keep > 0
        ranked = torch.where(selected, scores, torch.full_like(scores, MASKED))
        values, idx = torch.topk(ranked, cap, dim=-1)
        keep = values > MASKED / 2
        if self.phase == "decode":
            rows = key_rows.gather(1, idx)
        else:
            rows = key_rows[0][idx]
        return rows, keep

    def _attend_core_kernel(self, attn, cache, query, rows, keep, x):
        """The decode attention core in one NKI kernel (kernels/attn_core.py): the rows padded to whole kernel row
        blocks; the all-reduced attention output [T, H]."""
        from sglang_neuron.backend.stack import wrap_nki
        from sglang_neuron.models.hy4.kernels.attn_core import attn_core_decode
        from sglang_neuron.models.hy4.kernels.mla_dense import onehot_rows
        from sglang_neuron.models.hy4.kernels.mla_sparse import block_slots
        from sglang_neuron.utils.hardware_config import get_logical_nc_config

        t, h, d = query.shape
        pad = -t % 4
        if pad:
            query = torch.cat([query, query.new_zeros((pad, h, d))])
            rows = torch.cat([rows, rows.new_zeros((pad, rows.shape[1]))])
            keep = torch.cat([keep, keep.new_zeros((pad, keep.shape[1]))])
        slots, bias = block_slots(rows, keep, h, cache.shape[0])
        gate_x = torch.sigmoid(x @ attn.gate)
        return wrap_nki(attn_core_decode)[get_logical_nc_config()](
            query.to(DTYPE).reshape(-1, 128, d), cache, slots, bias, onehot_rows(h, query.device),
            attn.sinks.reshape(1, -1), attn.w_uv[0], gate_x, attn.o, ranks=(tuple(range(self.group.size)),),
            scale=float(self.config.softmax_scale), t_real=t)

    def _attend_sparse(self, attn, cache, query, rows, keep):
        """Each head of each query over the rank's compacted selected keys (rows [R, cap]), merged across the ranks."""
        cfg = self.config
        scale, latent = cfg.softmax_scale, cfg.kv_lora_rank
        if self.phase == "decode" and _NKI_SPARSE_DECODE and query.device.type != "cpu" and rows.shape[0] >= 4:
            # a multi-row decode (a DFly verify): the prefill's kernel over the rows padded to whole row blocks
            t = query.shape[0]
            pad = -t % 4
            if pad:
                query = torch.cat([query, query.new_zeros((pad,) + tuple(query.shape[1:]))])
                rows = torch.cat([rows, rows.new_zeros((pad, rows.shape[1]))])
                keep = torch.cat([keep, keep.new_zeros((pad, keep.shape[1]))])
            if _nki_sparse_ok(query, rows):
                from sglang_neuron.backend.stack import wrap_nki
                from sglang_neuron.models.hy4.kernels.mla_dense import onehot_rows
                from sglang_neuron.models.hy4.kernels.mla_sparse import block_slots, mla_sparse
                from sglang_neuron.utils.hardware_config import get_logical_nc_config

                h, d = query.shape[1], query.shape[2]
                slots, bias = block_slots(rows, keep, h, cache.shape[0])
                out, ml = wrap_nki(mla_sparse)[get_logical_nc_config()](
                    query.to(DTYPE).reshape(-1, 128, d), cache, slots, bias, onehot_rows(h, query.device), scale=scale)
                return attn.merge_head_major(out[:, :t], ml[:, :t, 0], ml[:, :t, 1])
            query, rows, keep = query[:t], rows[:t], keep[:t]
        if self.phase != "decode" and _nki_sparse_ok(query, rows):
            # each row block over its tokens' gathered rows in one NKI kernel (kernels/mla_sparse.py)
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.hy4.kernels.mla_dense import onehot_rows
            from sglang_neuron.models.hy4.kernels.mla_sparse import block_slots, mla_sparse
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            h, d = query.shape[1], query.shape[2]
            slots, bias = block_slots(rows, keep, h, cache.shape[0])
            out, ml = wrap_nki(mla_sparse)[get_logical_nc_config()](
                query.to(DTYPE).reshape(-1, 128, d), cache, slots, bias, onehot_rows(h, query.device), scale=scale)
            return attn.merge_head_major(out, ml[..., 0], ml[..., 1])
        ms, ls, outs = [], [], []
        tile = rows.shape[0] if self.phase == "decode" else QUERY_TILE
        for start in range(0, rows.shape[0], tile):
            keys = cache[rows[start:start + tile]]  # [r, cap, compact]
            q = query[start:start + tile]
            s = torch.einsum("rhd,rkd->rhk", q, keys).float() * scale
            s = torch.where(keep[start:start + tile, None, :], s, torch.full_like(s, MASKED))
            m = s.amax(-1)
            p = torch.exp(s - m[..., None])
            ls.append(p.sum(-1))
            ms.append(m)
            outs.append(torch.einsum("rhk,rkd->rhd", p.to(DTYPE), keys[..., :latent]).float())
        cat = (lambda xs: xs[0]) if len(outs) == 1 else (lambda xs: torch.cat(xs, dim=0))
        return attn.merge(cat(outs), cat(ms), cat(ls))

    def _attend(self, attn, cache, query, key_rows, selected):
        """This rank's heads' merged attended latents: each head over this rank's selected keys, merged across the
        ranks with the sink."""
        cfg = self.config
        scale = cfg.softmax_scale
        latent = cfg.kv_lora_rank
        if self.phase == "decode":
            keys = cache[key_rows]  # [B, keys, compact]
            s = torch.einsum("bhd,bkd->bhk", query, keys).float() * scale
            s = torch.where(selected[:, None, :], s, torch.full_like(s, MASKED))
            m = s.amax(-1)
            p = torch.exp(s - m[..., None])
            l = p.sum(-1)
            out = torch.einsum("bhk,bkr->bhr", p.to(DTYPE), keys[..., :latent]).float()
            return attn.merge(out, m, l)
        rows = key_rows[0]
        t, h = query.shape[0], query.shape[1]
        if _nki_attend_ok(query, rows):
            # flash-style in one NKI kernel (kernels/mla_dense.py): no score leaves the core
            from sglang_neuron.backend.stack import wrap_nki
            from sglang_neuron.models.hy4.kernels.mla_dense import UNROLL, mla_dense, onehot_rows
            from sglang_neuron.utils.hardware_config import get_logical_nc_config

            keys = cache[rows]  # [K, compact]
            k, d = keys.shape
            out, ml = wrap_nki(mla_dense)[get_logical_nc_config()](
                query.to(DTYPE).reshape(-1, UNROLL, 128, d), keys, keys.t().contiguous(),
                selected.to(DTYPE).reshape(-1, UNROLL, 128 // h, k), onehot_rows(h, query.device), scale=scale)
            return attn.merge_head_major(out, ml[..., 0], ml[..., 1])
        m = torch.full((t, h), MASKED, dtype=torch.float32, device=query.device)
        l = torch.zeros((t, h), dtype=torch.float32, device=query.device)
        out = torch.zeros((t, h, latent), dtype=torch.float32, device=query.device)
        for start in range(0, rows.shape[0], KEY_TILE):
            keys = cache[rows[start:start + KEY_TILE]]  # [K, compact]
            s = torch.einsum("thd,kd->thk", query, keys).float() * scale
            s = s + selected[:, None, start:start + KEY_TILE]
            m_new = torch.maximum(m, s.amax(-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new[..., None])
            l = l * alpha + p.sum(-1)
            out = out * alpha[..., None] + torch.einsum("thk,kr->thr", p.to(DTYPE), keys[:, :latent]).float()
            m = m_new
        return attn.merge(out, m, l)

    # -- weights -----------------------------------------------------------------------------------------------------
    def hf_weight_mappings(self, rank: int) -> dict:
        """Parameter name to its checkpoint key, or the keys it is built from, for attention-TP ``rank``, which holds
        the routed experts [rank x local, (rank + 1) x local)."""
        cfg = self.config
        mappings = {"embed_tokens.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight",
                    "lm_head.weight": "model.embed_tokens.weight" if cfg.tie_word_embeddings else "lm_head.weight",
                    "hc_head.fn": "model.hc_head.hc_head_fn", "hc_head.base": "model.hc_head.hc_head_base",
                    "hc_head.scale": "model.hc_head.hc_head_scale"}
        for i, layer in enumerate(self.layers):
            src, dst = f"model.layers.{i}", f"layers.{i}"
            a = f"{src}.self_attn"
            fp8 = not (i == 0)
            for hc, name in (("attn_hc", "hc_attn_layer"), ("mlp_hc", "hc_mlp_layer")):
                mappings.update({f"{dst}.{hc}.fn": f"{src}.{name}.hc_pre.hc_fn",
                                 f"{dst}.{hc}.base": f"{src}.{name}.hc_pre.hc_base",
                                 f"{dst}.{hc}.scale": f"{src}.{name}.hc_pre.hc_scale"})
            mappings.update({
                f"{dst}.input_layernorm.weight": f"{src}.input_layernorm.weight",
                f"{dst}.post_attention_layernorm.weight": f"{src}.post_attention_layernorm.weight",
                f"{dst}.self_attn.q_a": f"{a}.q_a_proj.weight",
                f"{dst}.self_attn.q_a_layernorm.weight": f"{a}.q_a_layernorm.weight",
                f"{dst}.self_attn.q_b": f"{a}.q_b_proj.weight",
                f"{dst}.self_attn.kv_a": f"{a}.kv_a_proj_with_mqa.weight",
                f"{dst}.self_attn.kv_a_layernorm.weight": f"{a}.kv_a_layernorm.weight",
                f"{dst}.self_attn.w_uk": f"{a}.kv_b_proj.weight",
                f"{dst}.self_attn.w_uv": f"{a}.kv_b_proj.weight",
                f"{dst}.self_attn.gate": ([f"{a}.linear_gate.weight", f"{a}.linear_gate.weight_scale"] if fp8
                                          else [f"{a}.linear_gate.weight"]),
                **({f"{dst}.self_attn.qkv_sh": [f"{a}.q_a_proj.weight", f"{a}.kv_a_proj_with_mqa.weight"] + (
                    [f"{a}.linear_gate.weight", f"{a}.linear_gate.weight_scale"] if fp8
                    else [f"{a}.linear_gate.weight"])} if hasattr(layer.self_attn, "qkv_sh") else {}),
                f"{dst}.self_attn.o": ([f"{a}.o_proj.weight", f"{a}.o_proj.weight_scale"] if fp8
                                       else [f"{a}.o_proj.weight"]),
                f"{dst}.self_attn.sinks": f"{a}.learnable_sink_param",
            })
            if layer.self_attn.full:
                mappings.update({
                    f"{dst}.self_attn.idx_q": f"{a}.indexer.wq_b.weight",
                    **({f"{dst}.self_attn.idx_q_sh": f"{a}.indexer.wq_b.weight"} if _IDXQ_SH else {}),
                    f"{dst}.self_attn.idx_k": f"{a}.indexer.wk.weight",
                    f"{dst}.self_attn.idx_k_norm_weight": f"{a}.indexer.k_norm.weight",
                    f"{dst}.self_attn.idx_k_norm_bias": f"{a}.indexer.k_norm.bias",
                    f"{dst}.self_attn.idx_w": f"{a}.indexer.weights_proj.weight",
                })
            m = f"{src}.mlp"
            if layer.dense:
                mappings.update(_mlp_mappings(m, f"{dst}.mlp"))
                continue
            moe = layer.mlp
            local = range(rank * moe.num_local, (rank + 1) * moe.num_local)
            mappings.update({
                f"{dst}.mlp.router": f"{m}.gate.weight",
                f"{dst}.mlp.router_pm": f"{m}.gate.weight",
                f"{dst}.mlp.router_bias": f"{m}.gate.e_score_correction_bias",
            })
            if not _MOE_TP:
                mappings.update({
                    f"{dst}.mlp.gate_up": [f"{m}.experts.{e}.{p}_proj.weight" for e in local for p in ("gate", "up")],
                    f"{dst}.mlp.gate_up_scale": [f"{m}.experts.{e}.{p}_proj.weight_scale" for e in local
                                                 for p in ("gate", "up")],
                    f"{dst}.mlp.down": [f"{m}.experts.{e}.down_proj.weight" for e in local],
                    f"{dst}.mlp.down_scale": [f"{m}.experts.{e}.down_proj.weight_scale" for e in local],
                })
            if _MOE_TP:
                grp = range(rank // _MOE_TP * moe.tp_local, (rank // _MOE_TP + 1) * moe.tp_local)
                mappings.update({
                    f"{dst}.mlp.gate_up_tp": [f"{m}.experts.{e}.{p}_proj.weight" for e in grp for p in ("gate", "up")],
                    f"{dst}.mlp.gate_up_scale_tp": [f"{m}.experts.{e}.{p}_proj.weight_scale" for e in grp
                                                    for p in ("gate", "up")],
                    f"{dst}.mlp.down_tp": [f"{m}.experts.{e}.down_proj.weight" for e in grp],
                    f"{dst}.mlp.down_scale_tp": [f"{m}.experts.{e}.down_proj.weight_scale" for e in grp],
                })
            if moe.shared is not None:
                mappings.update(_mlp_mappings(f"{m}.shared_experts", f"{dst}.mlp.shared"))
                mappings[f"{dst}.mlp.sh_gate_up_pm"] = [f"{m}.shared_experts.gate_proj.weight",
                                                       f"{m}.shared_experts.up_proj.weight"]
        if self.dfly is not None:
            mappings.update(self.dfly.weight_mappings())
        return mappings

    def load_weights(self, checkpoint_path: str, device: torch.device) -> None:
        rank = self.group.coordinator.rank_in_group
        checkpoint = SafetensorsCheckpoint(checkpoint_path)
        state = checkpoint.load_sharded_pipelined(rank, self.group.size, self, self.hf_weight_mappings(rank),
                                                  device).state_dict
        self.load_state_dict(state, strict=False, assign=True)


def _mlp_mappings(src: str, dst: str) -> dict:
    return {f"{dst}.gate_up": [f"{src}.gate_proj.weight", f"{src}.up_proj.weight"],
            f"{dst}.down": f"{src}.down_proj.weight"}
