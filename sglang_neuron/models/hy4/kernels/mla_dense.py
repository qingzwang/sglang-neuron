# SPDX-License-Identifier: Apache-2.0
"""The prefill's masked MLA attention over a rank's keys in one NKI kernel (flash-style: no score leaves the core).

Per rank, every query head attends this rank's keys (the KV is context parallel: the caller merges the ranks' partial
softmaxes, Attention.merge). The absorbed queries are [T, H, 576] (512 latent + 64 rope channels); a key row is the
compact cache row [576] and its value the row's first 512 channels. The selection of a full layer arrives as an
additive bias per token and key (0, or MASKED for a key the token does not attend).

In the XLA graph the scores of a key tile ([T, H, 1024] fp32, 256 MB a 1K-token chunk) are spilled to HBM and read
back by every softmax step. Here a block of 128 query rows (``128 / H`` tokens of all H heads) runs over the keys in
tiles of 512, nothing but the block's output leaving the core:

1. S [128 rows, 512 keys] in PSUM: the block's queries, transposed once on the tensor engine, are the stationary
   operand of five matmuls (the 576 channels as 4 x 128 + 64) against the keys (transposed once per call); a sixth
   matmul adds the tokens' bias rows (a one-hot [tokens, 128 rows] stationary operand broadcasts a token's row to its
   heads), so the bias costs no per-row DMA and no vector pass;
2. the tile's row max is read from PSUM, and exp(scale * s - m) goes from PSUM to bf16 with its row sum on the scalar
   engine;
3. P is transposed on the tensor engine (four 128-key blocks) and P V accumulates [128 rows, 512] in PSUM, the running
   output, max and sum rescaled as in flash attention.

Up to TWO_PASS_KEYS keys a block keeps all its scores in PSUM and takes the row max first (no rescaling); past it the
tensor engine computes a tile's scores one tile ahead of its online softmax. The loops are unrolled at trace time: a
device loop (nl.fori_loop) inside the large XLA program made the compiler spill and reload the program's live state
around every iteration. The output is head-major, the unnormalized sum of p * value against the running max ``m`` (of
the scaled scores), with ``l`` the sum of p, for the merge (Attention.merge_head_major). The two programs of a logical
core take one half of the row blocks each."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
KEY_TILE = 512
MASKED = -1.0e30
UNROLL = 8
# up to this many keys a row block keeps all its scores in PSUM (4 of its 8 banks) and runs the two-pass softmax
TWO_PASS_KEYS = 2048


def mla_dense(q, kv, kvT, bias, onehot, scale=1.0):
    """q [NB / U, U, 128, D] bf16 (row p of block b = U * i + u is token (128 / H) * b + p // H, head p % H); kv [K, D]
    bf16; kvT [D, K] bf16; bias [NB / U, U, 128 / H, K] bf16; onehot [128 / H, 128] bf16 (row t is 1 on token t's H
    rows) -> (out [H, T, L] fp32, ml [H, T, 2] fp32: the running max of the scaled scores and the sum), with L = 512
    the value channels and K a multiple of 512."""
    NI, U, _, D = q.shape
    K, _ = kv.shape
    TPB = bias.shape[2]
    L = 512
    KC = K // P  # 128-key chunks
    chunks = [(o, min(P, D - o)) for o in range(0, D, P)]
    NC = len(chunks)
    n_kt = K // KEY_TILE
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    H = P // TPB
    T = NI * U * TPB
    # head-major: a head's rows are contiguous, as the merge's reduce-scatter over the heads takes them (a token-major
    # output made XLA relayout it for the collective with element-sized DMAs, ~3.7 s a 1K-token group)
    out = nl.ndarray((H, T, L), dtype=nl.float32, buffer=nl.shared_hbm, name="mla_dense_out")
    ml = nl.ndarray((H, T, 2), dtype=nl.float32, buffer=nl.shared_hbm, name="mla_dense_ml")

    # the keys, transposed (channels on the partitions), and the values (keys on the partitions), for every block
    kT = nl.ndarray((P, NC, K), dtype=kvT.dtype, buffer=nl.sbuf)
    for c, (o, w) in enumerate(chunks):
        nisa.dma_copy(dst=kT[0:w, c, 0:K], src=kvT.ap(pattern=[[K, w], [1, K]], offset=o * K))
    v = nl.ndarray((P, KC, L), dtype=kv.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=v[0:P, 0:KC, 0:L], src=kv.ap(pattern=[[D, P], [P * D, KC], [1, L]], offset=0))
    oh = nl.ndarray((TPB, P), dtype=onehot.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=oh[0:TPB, 0:P], src=onehot.ap(pattern=[[P, TPB], [1, P]], offset=0))

    per = NI // n_prg

    def at(t, pattern, offset, i):
        """``t``'s access pattern at row-block iteration ``i``: its first dimension indexes the iterations."""
        stride = 1
        for n in t.shape[1:]:
            stride *= n
        return t.ap(pattern=pattern, offset=offset + i * stride)

    def store(acc, mlt, u, i):
        """The block's rows (token tt, head h) to out[h, token] and ml[h, token]: one DMA per token of the block."""
        t0 = TPB * (U * i + u)
        for tt in range(TPB):
            nisa.dma_copy(dst=out.ap(pattern=[[T * L, H], [1, L]], offset=(t0 + tt) * L),
                          src=acc[tt * H:(tt + 1) * H, 0:L])
            nisa.dma_copy(dst=ml.ap(pattern=[[T * 2, H], [1, 2]], offset=(t0 + tt) * 2),
                          src=mlt[tt * H:(tt + 1) * H, 0:2])

    def two_pass(u, qT, bs, i):
        """K <= TWO_PASS_KEYS: all the block's scores in PSUM at once, so the softmax takes the row max over every key
        first and needs no rescaling, and P V accumulates over every key in one PSUM tile. Each query chunk is the
        stationary operand of all the tiles' matmuls in a row (one weight load), as is the bias one-hot."""
        sp = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.psum)
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
        # the reduce's negate gives -max; the exp's bias is -scale * max
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
                nisa.nc_matmul(dst=pv[0:P, 0:L], stationary=pT[0:P, j, 0:P],
                               moving=v[0:P, kt * (KEY_TILE // P) + j, 0:L], accumulate=(kt > 0 or j > 0))
        acc = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=acc[0:P, 0:L], src=pv[0:P, 0:L])
        mlt = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mlt[0:P, 0:1], data=nb[0:P, 0:1], op0=nl.multiply, operand0=-1.0)
        nisa.tensor_reduce(dst=mlt[0:P, 1:2], op=nl.add, data=rs[0:P, 0:n_kt], axis=1)
        store(acc, mlt, u, i)

    def row_block(u, i=None):
        qs = nl.ndarray((P, D), dtype=q.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=qs[0:P, 0:D], src=at(q, [[D, P], [1, D]], u * P * D, i))
        bs = nl.ndarray((TPB, K), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=bs[0:TPB, 0:K], src=at(bias, [[K, TPB], [1, K]], u * TPB * K, i))
        qT = nl.ndarray((P, NC, P), dtype=q.dtype, buffer=nl.sbuf)
        for c, (o, w) in enumerate(chunks):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:w, 0:P], data=qs[0:P, o:o + w])
            nisa.tensor_copy(dst=qT[0:w, c, 0:P], src=tp[0:w, 0:P],
                             engine=nisa.scalar_engine if c % 2 else nisa.vector_engine)
        if K <= TWO_PASS_KEYS:
            two_pass(u, qT, bs, i)
            return
        # the running max of the scaled scores, negated (nms = -m): the exp's bias as it stands
        nms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=nms[0:P, 0:1], value=-MASKED)
        lsum = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=lsum[0:P, 0:1], value=0.0)
        acc = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc[0:P, 0:L], value=0.0)

        def scores(kt):
            k0 = kt * KEY_TILE
            sp = nl.ndarray((P, KEY_TILE), dtype=nl.float32, buffer=nl.psum)
            for c, (o, w) in enumerate(chunks):
                nisa.nc_matmul(dst=sp[0:P, 0:KEY_TILE], stationary=qT[0:w, c, 0:P],
                               moving=kT[0:w, c, k0:k0 + KEY_TILE], accumulate=(c > 0))
            # + the tokens' bias rows, broadcast to their heads' rows (unscaled: the exp applies the scale)
            nisa.nc_matmul(dst=sp[0:P, 0:KEY_TILE], stationary=oh[0:TPB, 0:P], moving=bs[0:TPB, k0:k0 + KEY_TILE],
                           accumulate=True)
            return sp

        pending = scores(0)
        for kt in range(n_kt):
            sp = pending
            if kt + 1 < n_kt:
                pending = scores(kt + 1)
            mx = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=mx[0:P, 0:1], op=nl.maximum, data=sp[0:P, 0:KEY_TILE], axis=1)
            # nn = min(nms, -scale * mx): the new running max, negated
            nn = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=nn[0:P, 0:1], data=mx[0:P, 0:1], op0=nl.multiply, operand0=-scale,
                                      op1=nl.minimum, operand1=nms[0:P, 0:1])
            p = nl.ndarray((P, KEY_TILE), dtype=nl.bfloat16, buffer=nl.sbuf)
            rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=p[0:P, 0:KEY_TILE], op=nl.exp, data=sp[0:P, 0:KEY_TILE], bias=nn[0:P, 0:1],
                            scale=scale, reduce_op=nl.add, reduce_res=rs[0:P, 0:1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            # alpha = exp(m_old - m_new) = exp(-nms + nn)
            al = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=al[0:P, 0:1], op=nl.exp, data=nms[0:P, 0:1], bias=nn[0:P, 0:1], scale=-1.0)
            nisa.scalar_tensor_tensor(dst=lsum[0:P, 0:1], data=lsum[0:P, 0:1], op0=nl.multiply,
                                      operand0=al[0:P, 0:1], op1=nl.add, operand1=rs[0:P, 0:1])
            nms = nn
            pT = nl.ndarray((P, KEY_TILE // P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            for j in range(KEY_TILE // P):
                tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=tp[0:P, 0:P], data=p[0:P, j * P:(j + 1) * P])
                nisa.tensor_copy(dst=pT[0:P, j, 0:P], src=tp[0:P, 0:P],
                                 engine=nisa.scalar_engine if j % 2 else nisa.vector_engine)
            pv = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.psum)
            for j in range(KEY_TILE // P):
                nisa.nc_matmul(dst=pv[0:P, 0:L], stationary=pT[0:P, j, 0:P],
                               moving=v[0:P, kt * (KEY_TILE // P) + j, 0:L], accumulate=(j > 0))
            nisa.scalar_tensor_tensor(dst=acc[0:P, 0:L], data=acc[0:P, 0:L], op0=nl.multiply, operand0=al[0:P, 0:1],
                                      op1=nl.add, operand1=pv[0:P, 0:L])
        mlt = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mlt[0:P, 0:1], data=nms[0:P, 0:1], op0=nl.multiply, operand0=-1.0)
        nisa.tensor_copy(dst=mlt[0:P, 1:2], src=lsum[0:P, 0:1])
        store(acc, mlt, u, i)

    # unrolled (no device loop): a device loop inside a large XLA program made the compiler spill and reload the
    # program's live state around its iterations
    for i in range(pid * per, (pid + 1) * per):
        for u in range(U):
            row_block(u, i)
    return out, ml


def onehot_rows(heads: int, device=None):
    """The [128 / heads, 128] one-hot of ``mla_dense``'s bias matmul: row t is 1 on token t's heads' rows."""
    import torch

    t = torch.arange(P // heads, device=device)[:, None]
    p = torch.arange(P, device=device)[None, :]
    return (p // heads == t).to(torch.bfloat16)
