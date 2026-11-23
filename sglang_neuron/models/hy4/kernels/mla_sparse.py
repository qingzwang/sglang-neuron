# SPDX-License-Identifier: Apache-2.0
"""The prefill's sparse MLA attention over a rank's selected keys in one NKI kernel.

Past the dense bucket a token attends only the keys its full layer selected on this rank: at most ``cap`` of them,
compacted into the local cache rows ``rows [T, cap]`` (Hy4Graph._compact; a pad slot's row is ``R``, the cache's row
count). A row block of the queries is 128 / H tokens of all H heads; its key set is its tokens' rows side by side
(``slots = (128 / H) * cap``, no deduplication), and token tt attends only its own slots (the additive ``bias``), so
the result is each token's attention over its own selected keys, exactly.

For each row block the slots' cache rows are gathered from HBM by one indirect DMA (pad slots are out of bounds and
skipped: they move no data), transposed on the tensor engine into the score matmul's channel-major operand, and the
block runs the two-pass softmax of mla_dense over its slots (all its scores fit in PSUM). The gather buffers are a
fixed pool, zeroed once, so a skipped slot reads a finite stale row (its weight is exp(MASKED) = 0).

The output is head-major, as mla_dense's, for Attention.merge_head_major. The two programs of a logical core take
alternate row blocks."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
KEY_TILE = 512
NBUF = 3  # gather buffers in flight


def mla_sparse(q, cache, rows, bias, onehot, scale=1.0, tag=""):
    """q [NB, 128, D] bf16; cache [R, D] bf16 (this layer's cache rows); rows [NB, S] int32 (the block's slots' cache
    rows, R for a pad slot); bias [NB, 128 / H, S] bf16; onehot [128 / H, 128] bf16 -> (out [H, T, L] fp32, ml [H, T, 2]
    fp32), with S a multiple of 512, at most 2048, and L = 512."""
    NB, _, D = q.shape
    S = rows.shape[1]
    assert S % KEY_TILE == 0 and S <= 2048, f"mla_sparse: {S} slots a block (a multiple of {KEY_TILE}, at most 2048)"
    TPB = bias.shape[1]
    H = P // TPB
    T = NB * TPB
    L = 512
    SC = S // P  # 128-slot chunks
    n_kt = S // KEY_TILE
    chunks = [(o, min(P, D - o)) for o in range(0, D, P)]
    NC = len(chunks)
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    out = nl.ndarray((H, T, L), dtype=nl.float32, buffer=nl.shared_hbm, name="mla_sparse_out" + tag)
    ml = nl.ndarray((H, T, 2), dtype=nl.float32, buffer=nl.shared_hbm, name="mla_sparse_ml" + tag)

    oh = nl.ndarray((TPB, P), dtype=onehot.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=oh[0:TPB, 0:P], src=onehot.ap(pattern=[[P, TPB], [1, P]], offset=0))
    pool = []
    for i in range(NBUF):
        g = nl.ndarray((P, SC, D), dtype=cache.dtype, buffer=nl.sbuf)
        nisa.memset(dst=g[0:P, 0:SC, 0:D], value=0.0)
        pool.append(g)

    for b in range(pid, NB, n_prg):
        kg = pool[(b // n_prg) % NBUF]
        # the block's slots (block_slots' order), read column-major by the gather: kg[p, m] is score column m * 128 + p
        idx = nl.ndarray((P, SC), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx[0:P, 0:SC], src=rows.ap(pattern=[[1, P], [P, SC]], offset=b * S))
        nisa.dma_copy(dst=kg[0:P, 0:SC, 0:D], src=cache.ap(pattern=[[D, P * SC], [1, D]], offset=0,
                                                           vector_offset=idx[0:P, 0:SC], indirect_dim=0),
                      dge_mode=nisa.dge_mode.swdge, oob_mode=nisa.oob_mode.skip)
        bs = nl.ndarray((TPB, S), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=bs[0:TPB, 0:S], src=bias.ap(pattern=[[S, TPB], [1, S]], offset=b * TPB * S))
        qs = nl.ndarray((P, D), dtype=q.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=qs[0:P, 0:D], src=q.ap(pattern=[[D, P], [1, D]], offset=b * P * D))
        # the block's queries and keys, channel-major
        qT = nl.ndarray((P, NC, P), dtype=q.dtype, buffer=nl.sbuf)
        for c, (o, w) in enumerate(chunks):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:w, 0:P], data=qs[0:P, o:o + w])
            nisa.tensor_copy(dst=qT[0:w, c, 0:P], src=tp[0:w, 0:P],
                             engine=nisa.scalar_engine if c % 2 else nisa.vector_engine)
        kT = nl.ndarray((P, NC, S), dtype=cache.dtype, buffer=nl.sbuf)
        for m in range(SC):
            for c, (o, w) in enumerate(chunks):
                tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp[0:w, 0:P], data=kg[0:P, m, o:o + w])
                nisa.tensor_copy(dst=kT[0:w, c, m * P:(m + 1) * P], src=tp[0:w, 0:P],
                                 engine=nisa.scalar_engine if (m * NC + c) % 2 else nisa.vector_engine)
        # scores of all slots in PSUM: q k^T, + the tokens' bias rows broadcast to their heads
        sp = nl.ndarray((P, S), dtype=nl.float32, buffer=nl.psum)
        for c, (o, w) in enumerate(chunks):
            for kt in range(n_kt):
                k0 = kt * KEY_TILE
                nisa.nc_matmul(dst=sp[0:P, k0:k0 + KEY_TILE], stationary=qT[0:w, c, 0:P],
                               moving=kT[0:w, c, k0:k0 + KEY_TILE], accumulate=(c > 0))
        for kt in range(n_kt):
            k0 = kt * KEY_TILE
            nisa.nc_matmul(dst=sp[0:P, k0:k0 + KEY_TILE], stationary=oh[0:TPB, 0:P],
                           moving=bs[0:TPB, k0:k0 + KEY_TILE], accumulate=True)
        mx = nl.ndarray((P, n_kt), dtype=nl.float32, buffer=nl.sbuf)
        for kt in range(n_kt):
            nisa.tensor_reduce(dst=mx[0:P, kt:kt + 1], op=nl.maximum, data=sp[0:P, kt * KEY_TILE:(kt + 1) * KEY_TILE],
                               axis=1)
        nms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=nms[0:P, 0:1], op=nl.maximum, data=mx[0:P, 0:n_kt], axis=1, negate=True)
        nb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nb[0:P, 0:1], data=nms[0:P, 0:1], op0=nl.multiply, operand0=scale)
        rs = nl.ndarray((P, n_kt), dtype=nl.float32, buffer=nl.sbuf)
        pv = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.psum)
        for kt in range(n_kt):
            k0 = kt * KEY_TILE
            p = nl.ndarray((P, KEY_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=p[0:P, 0:KEY_TILE], op=nl.exp, data=sp[0:P, k0:k0 + KEY_TILE], bias=nb[0:P, 0:1],
                            scale=scale, reduce_op=nl.add, reduce_res=rs[0:P, kt:kt + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            pT = nl.ndarray((P, KEY_TILE // P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            for j in range(KEY_TILE // P):
                tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp[0:P, 0:P], data=p[0:P, j * P:(j + 1) * P])
                nisa.tensor_copy(dst=pT[0:P, j, 0:P], src=tp[0:P, 0:P],
                                 engine=nisa.scalar_engine if j % 2 else nisa.vector_engine)
            for j in range(KEY_TILE // P):
                m = kt * (KEY_TILE // P) + j
                nisa.nc_matmul(dst=pv[0:P, 0:L], stationary=pT[0:P, j, 0:P], moving=kg[0:P, m, 0:L],
                               accumulate=(kt > 0 or j > 0))
        acc = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=acc[0:P, 0:L], src=pv[0:P, 0:L])
        mlt = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mlt[0:P, 0:1], data=nb[0:P, 0:1], op0=nl.multiply, operand0=-1.0)
        nisa.tensor_reduce(dst=mlt[0:P, 1:2], op=nl.add, data=rs[0:P, 0:n_kt], axis=1)
        t0 = TPB * b
        for tt in range(TPB):
            nisa.dma_copy(dst=out.ap(pattern=[[T * L, H], [1, L]], offset=(t0 + tt) * L),
                          src=acc[tt * H:(tt + 1) * H, 0:L])
            nisa.dma_copy(dst=ml.ap(pattern=[[T * 2, H], [1, 2]], offset=(t0 + tt) * 2),
                          src=mlt[tt * H:(tt + 1) * H, 0:2])
    return out, ml


def block_slots(rows, keep, heads: int, pad: int):
    """The kernel's per-block slots from the per-token compaction (rows [T, cap] cache rows, keep [T, cap]) in torch:
    (slots [NB, (128 / heads) cap] int32, the pad slots ``pad``; bias [NB, 128 / heads, slots] bf16: 0 on a token's own
    kept slots, MASKED elsewhere)."""
    import torch

    t, cap = rows.shape
    tpb = P // heads
    nb = t // tpb
    slots = torch.where(keep, rows, torch.full_like(rows, pad)).reshape(nb, tpb * cap).to(torch.int32)
    # the gather fills row k of its [128, S / 128] destination at (k // (S / 128), k % (S / 128)) from its index tile
    # read column-major; the kernel's score column m * 128 + p is destination (p, m): so the slots go in transposed,
    # and the columns come out in the slots' natural order
    s = tpb * cap
    slots = slots.reshape(nb, s // P, P).transpose(1, 2).reshape(nb, s).contiguous()
    own = torch.eye(tpb, dtype=torch.bool, device=rows.device)[None, :, :, None]  # [1, tt, slot token, 1]
    kept = keep.reshape(nb, 1, tpb, cap)
    bias = torch.where(own & kept, torch.zeros((), device=rows.device), torch.full((), -1.0e30, device=rows.device))
    return slots, bias.reshape(nb, tpb, tpb * cap).to(torch.bfloat16)
