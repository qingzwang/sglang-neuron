# SPDX-License-Identifier: Apache-2.0
"""Hy4's checkpoint loaders: each builds this rank's shard of a parameter from the checkpoint's tensors, in the layout
the graphs read.

The checkpoint is compressed-tensors block FP8 (128 x 128 blocks, ``weight * weight_scale``) for the routed experts and
for the attention's ``o_proj`` and ``linear_gate`` (layer 0's attention is bf16). Every other weight is bf16, and the
iHC mixers, the sinks and the router's bias are fp32.

FP8 on trn2: the checkpoint's e4m3 is the OCP ``e4m3fn`` (finite, to 448); trn2's e4m3 is IEEE-like (to 240). The two
encode every value below 256 alike, so an expert weight is held halved (every value at most 224, exact but in the
fn subnormals, which lose their last bit) with its scale doubled, and the graphs compile with the compiler's
fn-as-e4m3 reinterpretation (``sglang_model.FP8_HLO_OPTION``). The attention's FP8 weights are dequantized to bf16 here,
once: they are small next to the experts, and the projections then run as plain bf16 matmuls.

Attention heads shard over the attention TP (heads_per_rank a rank); the routed experts split into contiguous blocks,
one per rank; the dense MLP's and the shared expert's intermediate dimension shard over the ranks."""

from __future__ import annotations

import torch

from sglang_neuron.models.hy4.config import Hy4Config
from sglang_neuron.utils.weight_loader import SafetensorsWeightLoader

BLOCK = 128


def trn2_fp8(weight: torch.Tensor) -> torch.Tensor:
    """An e4m3fn weight halved, as the e4m3 values trn2 reads alike (the module docstring)."""
    return (weight.to(torch.float32) * 0.5).to(torch.float8_e4m3fn)


def dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """A block-FP8 weight [out, in] times its [out / 128, in / 128] scale, as bf16."""
    out, inner = weight.shape
    blocks = weight.to(torch.float32).view(out // BLOCK, BLOCK, inner // BLOCK, BLOCK)
    return (blocks * scale.to(torch.float32)[:, None, :, None]).view(out, inner).to(torch.bfloat16)


def _maybe_dequant(slices, rows=None, cols=None) -> torch.Tensor:
    """A weight's slice (rows, cols as Python slices of the [out, in] weight; whole blocks) as bf16: dequantized when
    the mapping names its scale after it."""
    rows = rows or slice(None)
    cols = cols or slice(None)
    weight = slices[0][rows, cols]
    if len(slices) == 1:
        return weight.to(torch.bfloat16)
    scale_rows = slice(None if rows.start is None else rows.start // BLOCK,
                       None if rows.stop is None else rows.stop // BLOCK)
    scale_cols = slice(None if cols.start is None else cols.start // BLOCK,
                       None if cols.stop is None else cols.stop // BLOCK)
    return dequant(weight, slices[1][scale_rows, scale_cols])


def rope_halves(dim: int) -> torch.Tensor:
    """The order that takes RoPE channels from interleaved pairs (0, 1), (2, 3), ... to the halves [0, 2, ...] and
    [1, 3, ...] the graphs rotate. The checkpoint's MLA rotates interleaved pairs (SGLang's rope_interleave, on unless
    a config says otherwise); its indexer rotates halves already."""
    return torch.cat([torch.arange(0, dim, 2), torch.arange(1, dim, 2)])


def kv_a_loader(config: Hy4Config) -> SafetensorsWeightLoader:
    """kv_a_proj_with_mqa [latent + rope, H], replicated, as [H, latent + rope] with its RoPE rows in halves."""
    order = torch.cat([torch.arange(config.kv_lora_rank), config.kv_lora_rank + rope_halves(config.qk_rope_head_dim)])
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:][order].t())


def transposed() -> SafetensorsWeightLoader:
    """A replicated [out, in] projection as [in, out]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].t())


def as_float() -> SafetensorsWeightLoader:
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:].to(torch.float32))


def q_b_loader(config: Hy4Config, heads: int) -> SafetensorsWeightLoader:
    """q_b_proj [heads x (nope + rope), q_lora] as this rank's [q_lora, heads / tp x (nope + rope)], each head's RoPE
    rows in halves."""
    d, nope = config.q_head_dim, config.qk_nope_head_dim
    width = heads * d
    order = torch.cat([torch.arange(nope), nope + rope_halves(config.qk_rope_head_dim)])

    def transform(slices, rank):
        rows = slices[0][rank * width:(rank + 1) * width]
        return rows.view(heads, d, -1)[:, order].reshape(width, -1).t()

    return SafetensorsWeightLoader(transform=transform)


def kv_b_loader(config: Hy4Config, heads: int, values: bool) -> SafetensorsWeightLoader:
    """This rank's heads of kv_b_proj [heads x (nope + v), latent]: the key part [heads / tp, nope, latent], which the
    absorbed attention folds into the query, or the value part [heads / tp, v, latent]."""
    nope, v = config.qk_nope_head_dim, config.v_head_dim

    def transform(slices, rank):
        rows = slices[0][rank * heads * (nope + v):(rank + 1) * heads * (nope + v)].view(heads, nope + v, -1)
        return rows[:, nope:] if values else rows[:, :nope]

    return SafetensorsWeightLoader(transform=transform)


def gate_loader(config: Hy4Config, heads: int) -> SafetensorsWeightLoader:
    """linear_gate [heads x v, H] (FP8 but on layer 0): this rank's heads as bf16 [H, heads / tp x v]."""
    width = heads * config.v_head_dim
    return SafetensorsWeightLoader(transform=lambda slices, rank: _maybe_dequant(
        slices, rows=slice(rank * width, (rank + 1) * width)).t())


def o_loader(config: Hy4Config, heads: int) -> SafetensorsWeightLoader:
    """o_proj [H, heads x v] (FP8 but on layer 0): this rank's input columns as bf16 [heads / tp x v, H]."""
    width = heads * config.v_head_dim
    return SafetensorsWeightLoader(transform=lambda slices, rank: _maybe_dequant(
        slices, cols=slice(rank * width, (rank + 1) * width)).t())


def mlp_gate_up_loader(rows: int) -> SafetensorsWeightLoader:
    """A bf16 MLP's gate and up projections as one [H, 2 x I / tp], this rank's gate columns then its up columns."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.cat(
        [s[rank * rows:(rank + 1) * rows] for s in slices], dim=0).t())


def mlp_down_loader(rows: int) -> SafetensorsWeightLoader:
    """A bf16 MLP's down projection [H, I]: this rank's input columns as [I / tp, H]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:, rank * rows:(rank + 1) * rows].t())


def _t8(w: torch.Tensor) -> torch.Tensor:
    """An FP8 matrix transposed (through its bytes: torch transposes no float8 on the host)."""
    return w.view(torch.uint8).t().contiguous().view(w.dtype)


def experts_gate_up_loader() -> SafetensorsWeightLoader:
    """The rank's experts' gate and up projections (FP8 [I, H] each; the mapping lists gate then up per expert) as
    the transposed [E, H, 2I] the kernels stream (gate columns, then up), halved for trn2."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.stack(
        [torch.cat([_t8(trn2_fp8(gate[:])), _t8(trn2_fp8(up[:]))], dim=1)
         for gate, up in zip(slices[0::2], slices[1::2])]))


def experts_gate_up_scale_loader() -> SafetensorsWeightLoader:
    """Their [I / 128, H / 128] scales in the same blocking, [E, H / 128, 2I / 128] fp32, doubled with the halved
    weights."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.stack(
        [torch.cat([gate[:].t(), up[:].t()], dim=1) for gate, up in zip(slices[0::2], slices[1::2])]
    ).to(torch.float32).contiguous() * 2.0)


def experts_down_loader() -> SafetensorsWeightLoader:
    """The rank's experts' down projections (FP8 [H, I] each) as the transposed [E, I, H], halved for trn2."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.stack([_t8(trn2_fp8(d[:])) for d in slices]))


def experts_down_scale_loader() -> SafetensorsWeightLoader:
    """Their scales as [E, I / 128, H / 128] fp32, doubled."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: torch.stack(
        [d[:].t() for d in slices]).to(torch.float32).contiguous() * 2.0)


def vocab_rows_loader(rows: int, hidden_size: int) -> SafetensorsWeightLoader:
    from sglang_neuron.utils.weight_loader import pad_to_shape

    return SafetensorsWeightLoader(transform=lambda slices, rank: pad_to_shape(
        slices[0][rank * rows:(rank + 1) * rows], (rows, hidden_size)))


class Fp8Slice:
    """A block-FP8 checkpoint weight [out, in] (its slice and its [ceil(out / 128), ceil(in / 128)] scale's) read as
    bf16, sliced like the weight's own slice: each read dequantizes the whole blocks it covers, then trims. The DFly
    draft (dfly.py) stores every projection so, while its loaders are the target's bf16 ones."""

    def __init__(self, weight, scale):
        self.weight, self.scale = weight, scale

    def get_shape(self):
        return self.weight.get_shape()

    def __getitem__(self, key):
        shape = self.weight.get_shape()
        key = key if isinstance(key, tuple) else (key,)
        key = key + (slice(None),) * (2 - len(key))
        bounds = []
        for k, n in zip(key, shape):
            if not isinstance(k, slice) or k.step not in (None, 1):
                raise ValueError(f"an FP8 weight slice takes unit-step ranges, not {k!r}")
            start, stop, _ = k.indices(n)
            lo, hi = start // BLOCK * BLOCK, min(-(-stop // BLOCK) * BLOCK, n)
            bounds.append((start, stop, lo, hi))
        (r0, r1, R0, R1), (c0, c1, C0, C1) = bounds
        w = self.weight[R0:R1, C0:C1].to(torch.float32)
        s = self.scale[R0 // BLOCK:-(-R1 // BLOCK), C0 // BLOCK:-(-C1 // BLOCK)].to(torch.float32)
        s = s.repeat_interleave(BLOCK, 0)[:R1 - R0].repeat_interleave(BLOCK, 1)[:, :C1 - C0]
        return (w * s)[r0 - R0:r1 - R0, c0 - C0:c1 - C0].to(torch.bfloat16)


def fp8(loader: SafetensorsWeightLoader) -> SafetensorsWeightLoader:
    """``loader`` over block-FP8 weights: the mapping lists each weight then its scale, and ``loader`` sees one bf16
    slice (Fp8Slice) per weight."""
    def transform(slices, rank):
        views = [Fp8Slice(w, s) for w, s in zip(slices[0::2], slices[1::2])]
        return loader.load(views, rank) if loader.transform is None else loader.transform(views, rank)

    return SafetensorsWeightLoader(transform=transform)


def column_shard(rows: int) -> SafetensorsWeightLoader:
    """A [out, in] projection whose output rows shard over the ranks: this rank's [in, out / tp]."""
    return SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][rank * rows:(rank + 1) * rows].t())


def router_pm_loader() -> SafetensorsWeightLoader:
    """The router (gate.weight [NE, H]) as kernels/moe_sublayer.py's partition-major [128, H / 128 * NE] bf16."""
    from sglang_neuron.models.hy4.kernels.moe_sublayer import router_partition_major

    return SafetensorsWeightLoader(transform=lambda slices, rank: router_partition_major(
        slices[0][:].t().contiguous().to(torch.bfloat16)))


def shared_pm_loader(rows: int) -> SafetensorsWeightLoader:
    """This rank's shared gate / up ([H, 2 rows] as mlp_gate_up_loader builds it) as kernels/moe_sublayer.py's
    partition-major [2, 128, H / 128 * rows]."""
    from sglang_neuron.models.hy4.kernels.moe_sublayer import shared_partition_major

    inner = mlp_gate_up_loader(rows)
    return SafetensorsWeightLoader(transform=lambda slices, rank: shared_partition_major(
        inner.transform(slices, rank).contiguous(), 2))
