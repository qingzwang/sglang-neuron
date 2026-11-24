# SPDX-License-Identifier: Apache-2.0
"""Each row's k highest scores on the GpSimd engine's top-k (``nisa.topk``), for the indexer's selection.

``nisa.topk`` solves one top-k per group of 16 partitions, reading its source as a snake (element s at partition
s % 16, column s // 16). A row's K scores load blocked instead (partition q holds scores [q C, (q + 1) C), C =
K / 16: one contiguous DMA a row), so snake position s is score (s % 16) C + s // 16; the k winners come back in
the snake's first k / 16 columns, their positions inverted by a shift and a mask. Eight rows an instruction.

    topk_rows(scores [R, K] bf16, k) -> (values [R, k] bf16, indices [R, k] int32), in no particular order

K is a multiple of 16 (pad with a finite minimum: the top-k reads NaN as the lowest value), at most 32768 (wider
calls hung the next launch, GLM-5.2's notes); k a multiple of 16."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
G = 16  # partitions a snake


def topk_rows(scores, k=256):
    R, K = scores.shape
    assert K % G == 0 and K <= 32768 and k % G == 0 and k <= K, f"topk_rows: {K} scores, top {k}"
    C = K // G
    KC = k // G
    vals = nl.ndarray((R, k), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="topk_rows_vals")
    idxs = nl.ndarray((R, k), dtype=nl.int32, buffer=nl.shared_hbm, name="topk_rows_idx")
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    RG = P // G  # rows an instruction
    for r0 in range(pid * RG, R, n_prg * RG):
        nr = min(RG, R - r0)
        src = nl.ndarray((P, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        if nr < RG:
            nisa.memset(dst=src[0:P, 0:C], value=-3.0e38)
        for j in range(nr):
            nisa.dma_copy(dst=src[j * G:(j + 1) * G, 0:C], src=scores.ap(pattern=[[C, G], [1, C]], offset=(r0 + j) * K))
        v = nl.ndarray((P, k), dtype=nl.bfloat16, buffer=nl.sbuf)
        ix = nl.ndarray((P, k), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=v[0:P, 0:k], idx_dst=ix[0:P, 0:k], src=src[0:P, 0:C], n=K)
        # snake position s -> score (s % 16) C + s // 16; an SBUF -> SBUF DMA (ordered after the GpSimd write, and
        # the uint32 -> int32 conversion), not a compute read
        si = nl.ndarray((P, KC), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=si[0:P, 0:KC], src=ix[0:P, 0:KC], dge_mode=nisa.dge_mode.none)
        hi = nl.ndarray((P, KC), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hi[0:P, 0:KC], data=si[0:P, 0:KC], op0=nl.right_shift, operand0=4)
        lo = nl.ndarray((P, KC), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lo[0:P, 0:KC], data=si[0:P, 0:KC], op0=nl.bitwise_and, operand0=G - 1)
        cand = nl.ndarray((P, KC), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cand[0:P, 0:KC], data=lo[0:P, 0:KC], op0=nl.multiply, operand0=C)
        nisa.tensor_tensor(dst=cand[0:P, 0:KC], data1=cand[0:P, 0:KC], data2=hi[0:P, 0:KC], op=nl.add)
        vo = nl.ndarray((P, KC), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=vo[0:P, 0:KC], src=v[0:P, 0:KC], dge_mode=nisa.dge_mode.none)
        for j in range(nr):
            # result position (partition q, column c) of row j: out[r, q KC + c] (any order serves)
            nisa.dma_copy(dst=vals.ap(pattern=[[KC, G], [1, KC]], offset=(r0 + j) * k), src=vo[j * G:(j + 1) * G, 0:KC])
            nisa.dma_copy(dst=idxs.ap(pattern=[[KC, G], [1, KC]], offset=(r0 + j) * k),
                          src=cand[j * G:(j + 1) * G, 0:KC])
    return vals, idxs
