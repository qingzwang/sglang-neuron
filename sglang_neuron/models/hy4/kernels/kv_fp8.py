# SPDX-License-Identifier: Apache-2.0
"""Hy4's FP8 KV rows (HY4_KV_FP8): a cache row of ``n8`` trn2-e4m3 values followed by ``nb`` bf16 values, stored as
uint8 bytes [R, n8 + 2 nb] — the MLA rows: 512 latent channels in FP8, the 64 RoPE channels bf16 (640 bytes for 1152);
the index keys: 128 channels in FP8, none bf16.

XLA on this stack does not reinterpret FP8 bytes (a ``view(torch.uint8)`` of an FP8 tensor lowers to a value
conversion and an FP8 round trip folds away), so every conversion is a kernel's: the engines convert in SBUF and the
DMAs move the packed bytes. The writers pre-scale the FP8 channels into e4m3 range (models/hy4/model.py: the per-layer
scale, folded back into the query and value weights).

    kv_pack_write: rows of bf16 values packed and scattered into the cache (in place)
    kv_unpack_gather: cache rows gathered and unpacked to bf16
    unpack_rows: the SBUF unpack, shared with mla_sparse's gather"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def unpack_rows(dst, src8, parts, m, n8, nb):
    """dst [parts, m, n8 + nb] bf16 <- src8 [parts, m, n8 + 2 nb] uint8 (SBUF)."""
    f8 = src8.view(nl.float8_e4m3)
    nisa.tensor_copy(dst=dst[0:parts, 0:m, 0:n8], src=f8[0:parts, 0:m, 0:n8])
    if nb:
        b16 = src8.view(nl.bfloat16)
        nisa.tensor_copy(dst=dst[0:parts, 0:m, n8:n8 + nb], src=b16[0:parts, 0:m, n8 // 2:n8 // 2 + nb],
                         engine=nisa.scalar_engine)


def pack_rows(dst8, src, parts, m, n8, nb):
    """dst8 [parts, m, n8 + 2 nb] uint8 <- src [parts, m, n8 + nb] bf16 (SBUF; FP8 rounding to nearest even)."""
    f8 = dst8.view(nl.float8_e4m3)
    nisa.tensor_copy(dst=f8[0:parts, 0:m, 0:n8], src=src[0:parts, 0:m, 0:n8])
    if nb:
        b16 = dst8.view(nl.bfloat16)
        nisa.tensor_copy(dst=b16[0:parts, 0:m, n8 // 2:n8 // 2 + nb], src=src[0:parts, 0:m, n8:n8 + nb],
                         engine=nisa.scalar_engine)


def kv_pack_write(cache, rows, x, n8=512):
    """cache [R, W] uint8 (W = n8 + 2 nb); rows [N] int32 (N a multiple of 128; R: skipped); x [N, n8 + nb] bf16 ->
    cache with x's rows packed into rows (wrap_nki with alias_outputs_in_place). Slot j = m 128 + p."""
    N, D = x.shape
    W = cache.shape[1]
    nb = D - n8
    assert W == n8 + 2 * nb and N % P == 0, f"kv_pack_write: rows of {D} values into {W} bytes, {N} rows"
    M = N // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    for m in range(pid, M, n_prg):
        idx = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx[0:P, 0:1], src=rows.ap(pattern=[[1, P], [1, 1]], offset=m * P))
        xs = nl.ndarray((P, 1, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs[0:P, 0, 0:D], src=x.ap(pattern=[[D, P], [1, D]], offset=m * P * D))
        pk = nl.ndarray((P, 1, W), dtype=nl.uint8, buffer=nl.sbuf)
        pack_rows(pk, xs, P, 1, n8, nb)
        nisa.dma_copy(dst=cache.ap(pattern=[[W, P], [1, W]], offset=0, vector_offset=idx[0:P, 0:1], indirect_dim=0),
                      src=pk[0:P, 0, 0:W], dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
    return cache


def kv_unpack_gather(cache, rows_t, n8=512):
    """cache [R, W] uint8; rows_t [N] int32: the rows of slots j = m 128 + p laid [128, N / 128] (gather_order; R reads
    zeros) -> [N, n8 + nb] bf16 in slot order. One gather of every row (N W 3 bytes a partition fit in SBUF)."""
    N = rows_t.shape[0]
    W = cache.shape[1]
    nb = (W - n8) // 2
    D = n8 + nb
    M = N // P
    assert N % P == 0 and M * (W + 2 * D) <= 160 * 1024, f"kv_unpack_gather: {N} rows of {W} bytes"
    out = nl.ndarray((N, D), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="kv_unpack_out")
    pid = nl.program_id(axis=0)
    if pid == 0:
        idx = nl.ndarray((P, M), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx[0:P, 0:M], src=rows_t.ap(pattern=[[1, P], [P, M]], offset=0))
        k8 = nl.ndarray((P, M, W), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(dst=k8[0:P, 0:M, 0:W], value=0)
        nisa.dma_copy(dst=k8[0:P, 0:M, 0:W], src=cache.ap(pattern=[[W, P * M], [1, W]], offset=0,
                                                         vector_offset=idx[0:P, 0:M], indirect_dim=0),
                      dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
        kb = nl.ndarray((P, M, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        unpack_rows(kb, k8, P, M, n8, nb)
        nisa.dma_copy(dst=out.ap(pattern=[[D, P], [P * D, M], [1, D]], offset=0), src=kb[0:P, 0:M, 0:D])
    return out


def gather_order(rows):
    """kv_unpack_gather's rows_t from rows [N] in slot order (torch, the caller's graph)."""
    n = rows.shape[0]
    return rows.reshape(n // P, P).t().contiguous().reshape(-1)
