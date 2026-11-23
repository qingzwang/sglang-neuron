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


def index_scores_decode(qT, keys_rm, w, mask):
    """The decode's scores of a request's rows (a DFly verify's rows share their keys): qT [128, Tp * heads] bf16
    (column t * heads + h, Tp = 128 / heads * NB); keys_rm [K, 128] bf16 (as gathered: transposed here, on the
    tensor engine, instead of an XLA transpose: ~80 us at 8192 keys); w [Tp * heads / 128, 128] fp32; mask
    [128, NB, Tp] bf16 (placement_mask_decode) -> scores [Tp, K] fp32. Each 512-key tile: NB matmuls of 128 (token,
    head) query rows against the keys, the relu on the way out of PSUM, and NB matmuls of the placed weights summing
    each token's heads into [Tp, 512]; the two programs take alternate key tiles."""
    D, NQ = qT.shape
    K, _ = keys_rm.shape
    _, NB, Tp = mask.shape
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    out = nl.ndarray((Tp, K), dtype=nl.float32, buffer=nl.shared_hbm, name="index_scores_decode_out")
    q = nl.ndarray((P, NQ), dtype=qT.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=q[0:P, 0:NQ], src=qT.ap(pattern=[[NQ, P], [1, NQ]], offset=0))
    msk = nl.ndarray((P, NB, Tp), dtype=mask.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=msk[0:P, 0:NB, 0:Tp], src=mask.ap(pattern=[[NB * Tp, P], [Tp, NB], [1, Tp]], offset=0))
    wt = nl.ndarray((P, NB), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=wt[0:P, 0:NB], src=w.ap(pattern=[[1, P], [P, NB]], offset=0))
    wb = nl.ndarray((P, NB, Tp), dtype=nl.bfloat16, buffer=nl.sbuf)
    for j in range(NB):
        nisa.tensor_scalar(dst=wb[0:P, j, 0:Tp], data=msk[0:P, j, 0:Tp], op0=nl.multiply, operand0=wt[0:P, j:j + 1])
    NT = K // KEY_TILE
    NH = -(-NT // n_prg)  # this program's tiles
    SUB = KEY_TILE // P
    keys = nl.ndarray((P, NH, KEY_TILE), dtype=keys_rm.dtype, buffer=nl.sbuf)  # [dim, tile, key]
    tiles = list(range(pid, NT, n_prg))
    for i, kt in enumerate(tiles):
        kr = nl.ndarray((P, SUB, P), dtype=keys_rm.dtype, buffer=nl.sbuf)  # kr[p, s, :] = key 512 kt + 128 s + p
        nisa.dma_copy(dst=kr[0:P, 0:SUB, 0:P], src=keys_rm.ap(pattern=[[P, P], [P * P, SUB], [1, P]],
                                                                offset=kt * KEY_TILE * P),
                      dge_mode=nisa.dge_mode.hwdge, engine=(nisa.engine.sync, nisa.engine.scalar)[i % 2])
        for sb in range(SUB):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=kr[0:P, sb, 0:P])
            nisa.tensor_copy(dst=keys[0:P, i, sb * P:(sb + 1) * P], src=tp[0:P, 0:P],
                             engine=nisa.scalar_engine if sb % 2 else nisa.vector_engine)
    o = nl.ndarray((Tp, NH, KEY_TILE), dtype=nl.float32, buffer=nl.sbuf)
    for i, kt in enumerate(tiles):
        acc = nl.ndarray((Tp, KEY_TILE), dtype=nl.float32, buffer=nl.psum)
        for j in range(NB):
            s = nl.ndarray((P, KEY_TILE), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=s[0:P, 0:KEY_TILE], stationary=q[0:P, j * P:(j + 1) * P], moving=keys[0:P, i, 0:KEY_TILE])
            r = nl.ndarray((P, KEY_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=r[0:P, 0:KEY_TILE], op=nl.relu, data=s[0:P, 0:KEY_TILE])
            nisa.nc_matmul(dst=acc[0:Tp, 0:KEY_TILE], stationary=wb[0:P, j, 0:Tp], moving=r[0:P, 0:KEY_TILE],
                           accumulate=(j > 0))
        nisa.tensor_copy(dst=o[0:Tp, i, 0:KEY_TILE], src=acc[0:Tp, 0:KEY_TILE])
    for i, kt in enumerate(tiles):
        nisa.dma_copy(dst=out.ap(pattern=[[K, Tp], [1, KEY_TILE]], offset=kt * KEY_TILE), src=o[0:Tp, i, 0:KEY_TILE],
                      dge_mode=nisa.dge_mode.hwdge, engine=(nisa.engine.sync, nisa.engine.scalar)[i % 2])
    return out


def placement_mask_decode(heads: int, tp: int, device=None):
    """index_scores_decode's [128, NB, Tp] placement: row p of query block j is token (128 / heads) j + p // heads."""
    import torch

    nb = tp * heads // P
    p = torch.arange(P, device=device)[:, None, None]
    j = torch.arange(nb, device=device)[None, :, None]
    c = torch.arange(tp, device=device)[None, None, :]
    return (c == (P // heads) * j + p // heads).to(torch.bfloat16)
