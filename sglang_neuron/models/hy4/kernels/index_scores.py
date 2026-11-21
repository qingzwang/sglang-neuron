# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's prefill scores in one NKI kernel: ``scores[t, k] = sum_h w[t, h] * relu(q[t, h] . key[k])``.

In the XLA graph the per-head products ([tokens, heads, keys] fp32, 128 MB a 1K-token, 1K-key tile) do not fit in
SBUF: the compiler spills them to HBM and reads them back, and the head-weighted sum lowered to a batched matmul fed by
layout DMAs. Here nothing but the scores leaves the core. For each block of 128 tokens and each tile of 512 keys:

1. a block of 128 (token, head) query rows is the stationary operand against the keys (contraction over the index
   head dim, 128), so PSUM holds the block's [128 (token, head), 512] products;
2. the scalar engine applies the relu on the way from PSUM to SBUF (bf16);
3. the head weighting and the sum over a token's heads is a second matmul whose stationary operand is the block's
   weights placed in the token's column (``mask``, a 0 / 1 [128, heads, 128] placement: row p of query block j is
   token ``(128 / heads) * j + p // heads`` of the 128-token block), accumulated over the query blocks in one PSUM tile
   [128 tokens, 512 keys].

The two programs of a logical core take alternate 128-token blocks."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
KEY_TILE = 512


def index_scores(qT, keysT, w, mask):
    """qT [idim = 128, T * heads] bf16 (column t * heads + h); keysT [128, K] bf16; w [T * heads / 128, 128] fp32 (the
    flat [T, heads] weights); mask [128, heads, 128] bf16 -> scores [T, K] fp32."""
    D, NQ = qT.shape
    _, K = keysT.shape
    _, J, _ = mask.shape  # J = heads: the query blocks of a 128-token block
    T = NQ // J
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    out = nl.ndarray((T, K), dtype=nl.float32, buffer=nl.shared_hbm, name="index_scores_out")

    keys = nl.ndarray((P, K), dtype=keysT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=keys[0:P, 0:K], src=keysT.ap(pattern=[[K, P], [1, K]], offset=0))
    msk = nl.ndarray((P, J, P), dtype=mask.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=msk[0:P, 0:J, 0:P], src=mask.ap(pattern=[[J * P, P], [P, J], [1, P]], offset=0))

    for b in range(pid, T // P, n_prg):
        q = nl.ndarray((P, J * P), dtype=qT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q[0:P, 0:J * P], src=qT.ap(pattern=[[NQ, P], [1, J * P]], offset=b * J * P))
        # wt[p, j]: the weight of query row p of block j (flat index b * J * 128 + j * 128 + p)
        wt = nl.ndarray((P, J), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt[0:P, 0:J], src=w.ap(pattern=[[1, P], [P, J]], offset=b * J * P))
        wb = nl.ndarray((P, J, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(J):
            nisa.tensor_scalar(dst=wb[0:P, j, 0:P], data=msk[0:P, j, 0:P], op0=nl.multiply,
                               operand0=wt[0:P, j:j + 1])
        for kt in range(K // KEY_TILE):
            acc = nl.ndarray((P, KEY_TILE), dtype=nl.float32, buffer=nl.psum)
            for j in range(J):
                s = nl.ndarray((P, KEY_TILE), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=s[0:P, 0:KEY_TILE], stationary=q[0:P, j * P:(j + 1) * P],
                               moving=keys[0:P, kt * KEY_TILE:(kt + 1) * KEY_TILE])
                r = nl.ndarray((P, KEY_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.activation(dst=r[0:P, 0:KEY_TILE], op=nl.relu, data=s[0:P, 0:KEY_TILE])
                nisa.nc_matmul(dst=acc[0:P, 0:KEY_TILE], stationary=wb[0:P, j, 0:P], moving=r[0:P, 0:KEY_TILE],
                               accumulate=(j > 0))
            o = nl.ndarray((P, KEY_TILE), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o[0:P, 0:KEY_TILE], src=acc[0:P, 0:KEY_TILE])
            nisa.dma_copy(dst=out.ap(pattern=[[K, P], [1, KEY_TILE]], offset=b * P * K + kt * KEY_TILE),
                          src=o[0:P, 0:KEY_TILE])
    return out


def placement_mask(heads: int, device=None):
    """The [128, heads, 128] 0 / 1 placement of ``index_scores``: row p of query block j feeds token column
    (128 / heads) * j + p // heads."""
    import torch

    p = torch.arange(P, device=device)[:, None, None]
    j = torch.arange(heads, device=device)[None, :, None]
    c = torch.arange(P, device=device)[None, None, :]
    return (c == (P // heads) * j + p // heads).to(torch.bfloat16)
