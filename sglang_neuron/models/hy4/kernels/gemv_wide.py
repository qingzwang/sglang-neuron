# SPDX-License-Identifier: Apache-2.0
"""A decode step's projections of one input in one NKI kernel, the tensor engine at the weights' DMA rate.

In the decode graph the replicated down-projections of the attention (q_a 6144 x 2048, kv_a 6144 x 576, bf16) run on
the tensor engine at ~240 and ~94 GB/s of weights (the served 7-row verify group, prof_nx/vg7r): with a few tokens the
compiler's matmuls move narrow weight tiles. Here the tokens [128, T] of a row block are the stationary operand of
every matmul of that block (T <= 32 columns, so four matmuls run at once in the array's 32-column tiles) and each
matmul moves 512 weight columns (~1.5 TB/s of bf16 a program, kbench/gemv_rate.py): the kernel streams the weights at
the DMA rate.

Each weight's columns split in halves over the two programs of a logical core, each half in chunks of at most 512
columns, one chunk a tile; the products accumulate over the row blocks in PSUM."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
TILE = 32
NT = P // TILE
W = 512
KB_DMA = 4
NBUF = 4
DMA_ENGINES = (nisa.engine.sync, nisa.engine.scalar)


def gemv_wide(xT, w0, w1=None, w2=None, w3=None):
    """xT [K, T] bf16 (T <= 32); w0 .. w3 (the given ones): [K, N_i] bf16 (N_i even) -> one [T, N_i] fp32 each."""
    ws = [w for w in (w0, w1, w2, w3) if w is not None]
    K, T = xT.shape
    KH = K // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    outs = [nl.ndarray((T, w.shape[1]), dtype=nl.float32, buffer=nl.shared_hbm, name=f"gemv_wide_out{i}")
            for i, w in enumerate(ws)]
    xs = nl.ndarray((P, KH, T), dtype=xT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs[0:P, 0:KH, 0:T], src=xT.ap(pattern=[[T, P], [P * T, KH], [1, T]], offset=0))
    dmas = 0
    for i, w in enumerate(ws):
        N = w.shape[1]
        n = N // n_prg
        c0 = pid * n
        mine = [(s0, min(W, n - s0)) for s0 in range(0, n, W)]  # this program's chunks of the weight
        for g0 in range(0, len(mine), NT):
            group = mine[g0:g0 + NT]
            span = group[-1][0] + group[-1][1] - group[0][0]
            ps = [nl.ndarray((P, W), dtype=nl.float32, buffer=nl.psum) for _ in group]
            # a fixed pool of weight buffers, in turn: the DMAs run NBUF row-block groups ahead of the matmuls
            pool = [nl.ndarray((P, KB_DMA, NT * W), dtype=w.dtype, buffer=nl.sbuf) for _ in range(NBUF)]
            for kb in range(KH):
                if kb % KB_DMA == 0:
                    b = pool[(kb // KB_DMA) % NBUF]
                    # rows of at most 2 KB a DMA: the DMA engines move whole 2 KB packets then (smaller rows split
                    # into ~650 B packets, at half the rate)
                    step = 2048 // w.dtype.itemsize if hasattr(w.dtype, "itemsize") else 1024
                    for d0 in range(0, span, step):
                        dw = min(step, span - d0)
                        nisa.dma_copy(dst=b[0:P, 0:KB_DMA, d0:d0 + dw],
                                      src=w.ap(pattern=[[N, P], [P * N, KB_DMA], [1, dw]],
                                               offset=kb * P * N + c0 + group[0][0] + d0),
                                      dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[dmas % 2])
                        dmas += 1
                for c, (s0, width) in enumerate(group):
                    o0 = s0 - group[0][0]
                    nisa.nc_matmul(dst=ps[c][c * TILE:c * TILE + T, 0:width], stationary=xs[0:P, kb, 0:T],
                                   moving=b[0:P, kb % KB_DMA, o0:o0 + width], tile_position=(0, c * TILE),
                                   tile_size=(P, TILE), accumulate=None)
            for c, (s0, width) in enumerate(group):
                o = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o[c * TILE:c * TILE + T, 0:width], src=ps[c][c * TILE:c * TILE + T, 0:width])
                nisa.dma_copy(dst=outs[i].ap(pattern=[[N, T], [1, width]], offset=c0 + s0),
                              src=o[c * TILE:c * TILE + T, 0:width])
    return tuple(outs)
