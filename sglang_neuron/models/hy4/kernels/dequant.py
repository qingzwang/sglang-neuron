# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 weight dequantization in one NKI kernel: ``out [O, I] bf16 = w [O, I] fp8 * s[O / 128, I / 128]``.

The prefill's expert matmuls read each expert weight once per chunk; dequantized in the XLA graph, the compiler fused
the cast and the block-scale multiply into its matmul tiling and spent ~15x the kernel's time on it (measured in the
served group program; a standalone graph of the same ops did not show it). As a kernel the dequantization is one pass
over the weight: each 128-row block of the weight is loaded into SBUF, every 128-column block scaled by its scale as a
per-partition scalar and cast to bf16, and the block stored. The two programs of a logical core take alternate row
blocks."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def dequant_fp8_block(w, s):
    """``w`` [O, I] FP8 (trn2's e4m3), ``s`` [O / 128, I / 128] fp32 -> [O, I] bf16."""
    O, I = w.shape  # noqa: E741
    ob_n, ib_n = O // P, I // P
    out = nl.ndarray((O, I), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="dequant_out")
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    for ob in range(pid, ob_n, n_prg):
        wt = nl.ndarray((P, I), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt[0:P, 0:I], src=w.ap(pattern=[[I, P], [1, I]], offset=ob * P * I))
        # the row block's scales on every partition: [128, I / 128]
        st = nl.ndarray((P, ib_n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st[0:P, 0:ib_n], src=s.ap(pattern=[[0, P], [1, ib_n]], offset=ob * ib_n))
        ot = nl.ndarray((P, I), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ib in range(ib_n):
            nisa.tensor_scalar(dst=ot[0:P, ib * P:(ib + 1) * P], data=wt[0:P, ib * P:(ib + 1) * P],
                               op0=nl.multiply, operand0=st[0:P, ib:ib + 1])
        nisa.dma_copy(dst=out.ap(pattern=[[I, P], [1, I]], offset=ob * P * I), src=ot[0:P, 0:I])
    return out
