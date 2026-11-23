# SPDX-License-Identifier: Apache-2.0
"""Hy4's decode attention input in one NKI kernel, the query's all-gather included (the giga kernel's attention entry).

From the hyper-connection's attention input to every head's absorbed query, for T <= 8 tokens:

    RMSNorm -> q_a and kv_a (one weight pass, 512-column moving tiles) -> q_a_layernorm, kv_a_layernorm
        -> q_b -> RoPE of the query's and the key's rope channels -> the absorbed query (q_nope @ w_uk)
        -> all-gather of the query over the attention TP (every rank's head)

In the XLA graph the replicated projections (q_a 6144 x 2048, kv_a 6144 x 576, bf16: 32 MB a rank and layer) run on
the tensor engine at 94-240 GB/s (prof_nx/vg7r: ~130 us a layer). Here they stream at the DMA rate: the two programs
of a logical core take half of the columns each, each 128-row block of the normed tokens is the stationary of a
matmul moving 512 columns, and the halves meet in shared HBM. Every later step is small and runs on both programs
alike. Rounding follows the XLA graph: the projections rounded to bf16, the norms in fp32 rounded, RoPE's products and
sum rounded to bf16, the absorbed query rounded."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

P = 128
TILE = 32
W = 512
KB_DMA = 4
DMA_ENGINES = (nisa.engine.sync, nisa.engine.scalar)


def _rmsnorm_rows(dst, src, w_tile, T, N, eps, name):
    """dst [T, N] bf16 = w * bf16(src * rsqrt(mean(src^2) + eps)) (RMSNorm of tokens on partitions, as the graph's)."""
    sq = nl.ndarray((T, N), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq[0:T, 0:N], op=nl.square, data=src[0:T, 0:N], reduce_op=nl.add, reduce_res=ss[0:T, 0:1],
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.tensor_scalar(dst=ss[0:T, 0:1], data=ss[0:T, 0:1], op0=nl.multiply, operand0=1.0 / N, op1=nl.add,
                       operand1=eps)
    nisa.activation(dst=ss[0:T, 0:1], op=nl.rsqrt, data=ss[0:T, 0:1])
    nb = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nb[0:T, 0:N], data=src[0:T, 0:N], op0=nl.multiply, operand0=ss[0:T, 0:1])
    nisa.tensor_tensor(dst=dst[0:T, 0:N], data1=nb[0:T, 0:N], data2=w_tile[0:T, 0:N], op=nl.multiply)


def _rope(dst, src, cs, sn, T, R):
    """dst [T, R] bf16 = bf16(bf16(x * cos) + bf16(rot(x) * sin)), rot(x) = [-x[R/2:], x[:R/2]] (the graph's)."""
    h = R // 2
    a = nl.ndarray((T, R), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=a[0:T, 0:R], data1=src[0:T, 0:R], data2=cs[0:T, 0:R], op=nl.multiply)
    rt = nl.ndarray((T, R), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rt[0:T, 0:h], data=src[0:T, h:R], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_copy(dst=rt[0:T, h:R], src=src[0:T, 0:h])
    b = nl.ndarray((T, R), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=b[0:T, 0:R], data1=rt[0:T, 0:R], data2=sn[0:T, 0:R], op=nl.multiply)
    nisa.tensor_tensor(dst=dst[0:T, 0:R], data1=a[0:T, 0:R], data2=b[0:T, 0:R], op=nl.add)


def attn_in_decode(x, ln_w, q_a, qa_ln_w, q_b, w_uk, kv_a, kva_ln_w, cos, sin, cache, write_rows, gate=None, t_pad=0, hook=None, ranks=(),
                   eps=1e-5, lat_eps=1e-6, qkv_sh=None, tag=""):
    """x [T, H] bf16 (the attention's hyper-connection input, before the norm); ln_w [1, H] bf16; q_a [H, QL] bf16;
    qa_ln_w [1, QL] bf16; q_b [QL, D] bf16 (this rank's one head: D = nope + rope); w_uk [1, nope, L] bf16; kv_a
    [H, L + rope] bf16 (rope rows in halves); kva_ln_w [1, L] bf16; cos, sin [T, rope] bf16; ranks: the attention TP
    (empty: no all-gather, a one-rank test); cache [R, L + rope] bf16 (this layer's latent cache rows) and write_rows
    [1, T] int32 (each token's row, 0 where another rank owns it): the compact rows are written there as soon as they
    are computed (wrap_nki with alias_outputs_in_place) -> (query [T, ranks, L + rope] bf16 (rank r's head r), compact
    [T, L + rope] bf16, q_lat [T, QL] bf16, cache[, gate_x]): with ``gate`` [H, V] bf16 (the output gate's columns of
    this rank) also gate_x = sigmoid(xn @ gate) [T, V] bf16, its projection in the same weight pass.

    ``qkv_sh`` [128, H / 128, CQ + CK + V] bf16 (qkv_shard_layout): q_a and kv_a sharded over the attention TP
    instead of replicated: this rank's CQ = QL / ranks columns of q_a, CK = NKV / ranks of kv_a and its gate columns,
    partition-major. Each rank projects its columns (0.5 MB of weights instead of 32 MB) and an all-gather of the
    [T, CQ + CK] slices assembles the latents on every rank (q_a and kv_a are then not read)."""
    T, H = x.shape
    QL = q_a.shape[1]
    NKV = kv_a.shape[1]
    D = q_b.shape[1]
    NOPE = w_uk.shape[1]
    L = w_uk.shape[2]
    RP = D - NOPE  # rope channels
    KH = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    nr = len(ranks[0]) if ranks else 1
    group = ReplicaGroup([list(g) for g in ranks]) if ranks else None

    TQ = max(t_pad, T)  # the query's rows: padded with zeros to the attention's row blocks
    query = nl.ndarray((TQ, nr, L + RP), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_query" + tag)
    if TQ > T and pid == 0:
        zq = nl.ndarray((TQ - T, nr * (L + RP)), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=zq[0:TQ - T, 0:nr * (L + RP)], value=0.0)
        nisa.dma_copy(dst=query.ap(pattern=[[nr * (L + RP), TQ - T], [1, nr * (L + RP)]], offset=T * nr * (L + RP)),
                      src=zq[0:TQ - T, 0:nr * (L + RP)])
    compact = nl.ndarray((T, NKV), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_compact" + tag)
    q_lat = nl.ndarray((T, QL), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_qlat" + tag)
    NG_ = gate.shape[1] if gate is not None else 0
    PJ = QL + NKV + NG_
    pj = nl.ndarray((T, PJ), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_proj" + tag)  # the halves meet
    gate_x = nl.ndarray((T, max(NG_, 1)), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_gate" + tag)
    qloc = nl.ndarray((T, L + RP), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_qloc" + tag)
    qall = nl.ndarray((nr, T, L + RP), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_qall" + tag)

    SH = qkv_sh is not None
    if SH:
        CS = qkv_sh.shape[2]
        CQ, CK = QL // nr, NKV // nr
        # (the pack's last columns are the gate's: NG_ of them, used when ``gate`` is given)
        KHH = KH // n_prg  # this program's row blocks
        wsh = nl.ndarray((P, KHH, CS), dtype=qkv_sh.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wsh[0:P, 0:KHH, 0:CS], src=qkv_sh.ap(pattern=[[KH * CS, P], [1, KHH * CS]],
                                                                 offset=pid * KHH * CS),
                      dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[pid % 2])
        part = nl.ndarray((n_prg, T, CS), dtype=nl.float32, buffer=nl.shared_hbm, name="attn_in_part" + tag)
        sloc = nl.ndarray((T, CQ + CK), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_sloc" + tag)
        sall = nl.ndarray((nr, T, CQ + CK), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_in_sall" + tag)
    else:
        # ---- this program's columns of [q_a | kv_a], in chunks of <= 512 (one column tile each) ----------------------
        qn, kn = QL // n_prg, NKV // n_prg
        # (weight, its first column, width, the column in pj)
        chunks = [(q_a, pid * qn + s0, min(W, qn - s0), pid * qn + s0) for s0 in range(0, qn, W)]
        chunks += [(kv_a, pid * kn + s0, min(W, kn - s0), QL + pid * kn + s0) for s0 in range(0, kn, W)]
        if gate is not None:
            gn = NG_ // n_prg
            chunks += [(gate, pid * gn + s0, min(W, gn - s0), QL + NKV + pid * gn + s0) for s0 in range(0, gn, W)]
        NC = len(chunks)
        # the weights' first row blocks first: they stream while the norm runs
        bufs = {}

        def load(kb):
            out = []
            for i, (w, c0, width, _) in enumerate(chunks):
                N = w.shape[1]
                b = nl.ndarray((P, KB_DMA, W), dtype=w.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=b[0:P, 0:KB_DMA, 0:width], src=w.ap(pattern=[[N, P], [P * N, KB_DMA], [1, width]],
                                                                        offset=kb * P * N + c0),
                              dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[(kb // KB_DMA + i) % 2])
                out.append(b)
            return out

        bufs[0] = load(0)

    # ---- the tokens transposed (xr[p, kb, t] = x[t, 128 kb + p]), then RMSNorm in that layout ---------------------
    xt = nl.ndarray((T, H), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt[0:T, 0:H], src=x.ap(pattern=[[H, T], [1, H]], offset=0))
    TA = T + T % 2  # a bf16 PSUM slice must be 4-byte aligned
    xps = nl.ndarray((P, KH, TA), dtype=nl.bfloat16, buffer=nl.psum)
    for kb in range(KH):
        nisa.nc_transpose(dst=xps[0:P, kb, 0:T], data=xt[0:T, kb * P:(kb + 1) * P])
    xr = nl.ndarray((P, KH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xr[0:P, 0:KH, 0:T], src=xps[0:P, 0:KH, 0:T])
    sq = nl.ndarray((P, KH, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq[0:P, 0:KH, 0:T], data1=xr[0:P, 0:KH, 0:T], data2=xr[0:P, 0:KH, 0:T], op=nl.multiply)
    ssp = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssp[0:P, 0:T], op=nl.add, data=sq.ap(pattern=[[KH * T, P], [1, T], [T, KH]], offset=0),
                       axis=2)
    ones = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones[0:P, 0:1], value=1.0)
    tps0 = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tps0[0:1, 0:T], stationary=ones[0:P, 0:1], moving=ssp[0:P, 0:T])
    rinv = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rinv[0:1, 0:T], data=tps0[0:1, 0:T], op0=nl.multiply, operand0=1.0 / H, op1=nl.add,
                       operand1=eps)
    nisa.activation(dst=rinv[0:1, 0:T], op=nl.rsqrt, data=rinv[0:1, 0:T])
    onesrow = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onesrow[0:1, 0:P], value=1.0)
    rbp = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=rbp[0:P, 0:T], stationary=onesrow[0:1, 0:P], moving=rinv[0:1, 0:T])
    rbs = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rbs[0:P, 0:T], src=rbp[0:P, 0:T])
    xb = nl.ndarray((P, KH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xb[0:P, 0:KH, 0:T], data1=xr[0:P, 0:KH, 0:T],
                       data2=rbs.ap(pattern=[[T, P], [0, KH], [1, T]], offset=0), op=nl.multiply)
    wpk = nl.ndarray((P, KH), dtype=ln_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=wpk[0:P, 0:KH], src=ln_w.ap(pattern=[[1, P], [P, KH]], offset=0))
    xs = nl.ndarray((P, KH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xs[0:P, 0:KH, 0:T], data1=xb[0:P, 0:KH, 0:T],
                       data2=wpk.ap(pattern=[[KH, P], [1, KH], [0, T]], offset=0), op=nl.multiply)

    TA = T + T % 2
    if SH:
        # ---- this rank's columns: each program half the row blocks, the fp32 halves summed ----------------------------
        pf = nl.ndarray((T, CS), dtype=nl.float32, buffer=nl.sbuf)
        for c0 in range(0, CS, W):  # (one column tile in the served layout: CS = 169)
            cw = min(W, CS - c0)
            pp = nl.ndarray((T, W), dtype=nl.float32, buffer=nl.psum)
            for k in range(KHH):
                nisa.nc_matmul(dst=pp[0:T, 0:cw], stationary=xs[0:P, pid * KHH + k, 0:T],
                               moving=wsh[0:P, k, c0:c0 + cw], accumulate=(k > 0))
            nisa.tensor_copy(dst=pf[0:T, c0:c0 + cw], src=pp[0:T, 0:cw])
        nisa.dma_copy(dst=part.ap(pattern=[[CS, T], [1, CS]], offset=pid * T * CS), src=pf[0:T, 0:CS])
        nisa.core_barrier(part, (0, 1))
        pa = nl.ndarray((T, n_prg, CS), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pa[0:T, 0:n_prg, 0:CS], src=part.ap(pattern=[[CS, T], [T * CS, n_prg], [1, CS]], offset=0))
        sb = nl.ndarray((T, CS), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sb[0:T, 0:CS], op=nl.add, data=pa.ap(pattern=[[n_prg * CS, T], [1, CS], [CS, n_prg]],
                                                                       offset=0), axis=2)
        if pid == 0:
            nisa.dma_copy(dst=sloc.ap(pattern=[[CQ + CK, T], [1, CQ + CK]], offset=0), src=sb[0:T, 0:CQ + CK])
        nisa.core_barrier(sloc, (0, 1))
        if group is not None:
            ncc.all_gather(srcs=[sloc], dsts=[sall], replica_group=group, collective_dim=0)
        else:
            cs_ = nl.ndarray((T, CQ + CK), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs_[0:T, 0:CQ + CK], src=sloc.ap(pattern=[[CQ + CK, T], [1, CQ + CK]], offset=0))
            nisa.dma_copy(dst=sall.ap(pattern=[[CQ + CK, T], [1, CQ + CK]], offset=0), src=cs_[0:T, 0:CQ + CK])
        nisa.core_barrier(sall, (0, 1))
        # the latents in column order: q_lat[t, CQ r + j] = sall[r, t, j], kv[t, CK r + j] = sall[r, t, CQ + j]
        pr = nl.ndarray((T, PJ), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=pr.ap(pattern=[[PJ, T], [CQ, nr], [1, CQ]], offset=0),
                      src=sall.ap(pattern=[[CQ + CK, T], [T * (CQ + CK), nr], [1, CQ]], offset=0))
        nisa.dma_copy(dst=pr.ap(pattern=[[PJ, T], [CK, nr], [1, CK]], offset=QL),
                      src=sall.ap(pattern=[[CQ + CK, T], [T * (CQ + CK), nr], [1, CK]], offset=CQ))
        if gate is not None:
            nisa.tensor_copy(dst=pr[0:T, QL + NKV:PJ], src=sb[0:T, CQ + CK:CS])
    else:
        # ---- the projections ---------------------------------------------------------------------------------------------
        ps = [nl.ndarray((P, W), dtype=nl.float32, buffer=nl.psum) for _ in range(NC)]
        for kb in range(KH):
            if kb % KB_DMA == 0 and kb not in bufs:
                bufs[kb] = load(kb)
            b = bufs[kb - kb % KB_DMA]
            for c, (w, c0, width, _) in enumerate(chunks):
                nisa.nc_matmul(dst=ps[c][c * TILE:c * TILE + T, 0:width], stationary=xs[0:P, kb, 0:T],
                               moving=b[c][0:P, kb % KB_DMA, 0:width], tile_position=(0, c * TILE),
                               tile_size=(P, TILE), accumulate=None)
        for c, (w, c0, width, o0) in enumerate(chunks):
            ob = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob[c * TILE:c * TILE + T, 0:width], src=ps[c][c * TILE:c * TILE + T, 0:width])
            nisa.dma_copy(dst=pj.ap(pattern=[[PJ, T], [1, width]], offset=o0),
                          src=ob[c * TILE:c * TILE + T, 0:width], dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[c % 2])
        nisa.core_barrier(pj, (0, 1))
        pr = nl.ndarray((T, PJ), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=pr[0:T, 0:PJ], src=pj.ap(pattern=[[PJ, T], [1, PJ]], offset=0))
    if gate is not None and pid == 0:
        gs_ = nl.ndarray((T, NG_), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.activation(dst=gs_[0:T, 0:NG_], op=nl.sigmoid, data=pr[0:T, QL + NKV:PJ])
        nisa.dma_copy(dst=gate_x.ap(pattern=[[NG_, T], [1, NG_]], offset=0), src=gs_[0:T, 0:NG_])

    # ---- the latents' norms, q_b, RoPE, the absorbed query (both programs alike) -------------------------------------
    qw = nl.ndarray((T, QL), dtype=qa_ln_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=qw[0:T, 0:QL], src=qa_ln_w.ap(pattern=[[0, T], [1, QL]], offset=0))
    ql = nl.ndarray((T, QL), dtype=nl.bfloat16, buffer=nl.sbuf)
    prq = nl.ndarray((T, QL), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=prq[0:T, 0:QL], src=pr[0:T, 0:QL])
    _rmsnorm_rows(ql, prq, qw, T, QL, lat_eps, "q")
    kw = nl.ndarray((T, L), dtype=kva_ln_w.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=kw[0:T, 0:L], src=kva_ln_w.ap(pattern=[[0, T], [1, L]], offset=0))
    cs = nl.ndarray((T, RP), dtype=cos.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=cs[0:T, 0:RP], src=cos.ap(pattern=[[RP, T], [1, RP]], offset=0))
    sn = nl.ndarray((T, RP), dtype=sin.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sn[0:T, 0:RP], src=sin.ap(pattern=[[RP, T], [1, RP]], offset=0))
    prl = nl.ndarray((T, L), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=prl[0:T, 0:L], src=pr[0:T, QL:QL + L])
    prk = nl.ndarray((T, RP), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=prk[0:T, 0:RP], src=pr[0:T, QL + L:QL + L + RP])
    cl = nl.ndarray((T, L), dtype=nl.bfloat16, buffer=nl.sbuf)
    _rmsnorm_rows(cl, prl, kw, T, L, lat_eps, "kv")
    ck = nl.ndarray((T, RP), dtype=nl.bfloat16, buffer=nl.sbuf)
    _rope(ck, prk, cs, sn, T, RP)
    cp = nl.ndarray((T, L + RP), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cp[0:T, 0:L], src=cl[0:T, 0:L])
    nisa.tensor_copy(dst=cp[0:T, L:L + RP], src=ck[0:T, 0:RP])
    # the cache rows now, ahead of the query's work: the attention's key gather waits on them
    if pid == 0:
        for t in range(T):
            r = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=r[0:1, 0:1], src=write_rows.ap(pattern=[[T, 1], [1, 1]], offset=t))
            nisa.dma_copy(dst=cache.ap(pattern=[[NKV, 1], [1, NKV]], offset=0, scalar_offset=r, indirect_dim=0),
                          src=cp[t:t + 1, 0:NKV])
    # q = q_lat @ q_b: q_lat transposed as the stationary
    KQ = QL // P
    qlp = nl.ndarray((P, KQ, TA), dtype=nl.bfloat16, buffer=nl.psum)
    for k in range(KQ):
        nisa.nc_transpose(dst=qlp[0:P, k, 0:T], data=ql[0:T, k * P:(k + 1) * P])
    qlT = nl.ndarray((P, KQ, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qlT[0:P, 0:KQ, 0:T], src=qlp[0:P, 0:KQ, 0:T])
    qbw = nl.ndarray((P, KQ, D), dtype=q_b.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=qbw[0:P, 0:KQ, 0:D], src=q_b.ap(pattern=[[D, P], [P * D, KQ], [1, D]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    qp = nl.ndarray((T, D), dtype=nl.float32, buffer=nl.psum)
    for k in range(KQ):
        nisa.nc_matmul(dst=qp[0:T, 0:D], stationary=qlT[0:P, k, 0:T], moving=qbw[0:P, k, 0:D], accumulate=None)
    qv = nl.ndarray((T, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qv[0:T, 0:D], src=qp[0:T, 0:D])
    qq = nl.ndarray((T, L + RP), dtype=nl.bfloat16, buffer=nl.sbuf)
    qpe = nl.ndarray((T, RP), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=qpe[0:T, 0:RP], src=qv[0:T, NOPE:D])
    qr = nl.ndarray((T, RP), dtype=nl.bfloat16, buffer=nl.sbuf)
    _rope(qr, qpe, cs, sn, T, RP)
    nisa.tensor_copy(dst=qq[0:T, L:L + RP], src=qr[0:T, 0:RP])
    # q_abs = q_nope @ w_uk: q_nope transposed in 128-row chunks
    nch = [(o, min(P, NOPE - o)) for o in range(0, NOPE, P)]
    uk = nl.ndarray((P, len(nch), L), dtype=w_uk.dtype, buffer=nl.sbuf)
    for i, (o, n) in enumerate(nch):
        nisa.dma_copy(dst=uk[0:n, i, 0:L], src=w_uk.ap(pattern=[[L, n], [1, L]], offset=o * L),
                      dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[i % 2])
    qnp = nl.ndarray((P, len(nch), TA), dtype=nl.bfloat16, buffer=nl.psum)
    for i, (o, n) in enumerate(nch):
        nisa.nc_transpose(dst=qnp[0:n, i, 0:T], data=qv[0:T, o:o + n])
    qnT = nl.ndarray((P, len(nch), T), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i, (o, n) in enumerate(nch):
        nisa.tensor_copy(dst=qnT[0:n, i, 0:T], src=qnp[0:n, i, 0:T])
    ap = nl.ndarray((T, L), dtype=nl.float32, buffer=nl.psum)
    for i, (o, n) in enumerate(nch):
        nisa.nc_matmul(dst=ap[0:T, 0:L], stationary=qnT[0:n, i, 0:T], moving=uk[0:n, i, 0:L], accumulate=None)
    nisa.tensor_copy(dst=qq[0:T, 0:L], src=ap[0:T, 0:L])

    # ---- outputs; the query gathered over the ranks ------------------------------------------------------------------
    if pid == 0:
        nisa.dma_copy(dst=compact.ap(pattern=[[NKV, T], [1, NKV]], offset=0), src=cp[0:T, 0:NKV])
        nisa.dma_copy(dst=q_lat.ap(pattern=[[QL, T], [1, QL]], offset=0), src=ql[0:T, 0:QL])
        nisa.dma_copy(dst=qloc.ap(pattern=[[L + RP, T], [1, L + RP]], offset=0), src=qq[0:T, 0:L + RP])
    nisa.core_barrier(qloc, (0, 1))
    if hook is not None:
        hook()  # the caller's prefetch DMAs, issued to run during the all-gather
    if group is not None:
        ncc.all_gather(srcs=[qloc], dsts=[qall], replica_group=group, collective_dim=0)
    else:
        cq = nl.ndarray((T, L + RP), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=cq[0:T, 0:L + RP], src=qloc.ap(pattern=[[L + RP, T], [1, L + RP]], offset=0))
        nisa.dma_copy(dst=qall.ap(pattern=[[L + RP, T], [1, L + RP]], offset=0), src=cq[0:T, 0:L + RP])
    nisa.core_barrier(qall, (0, 1))
    # the gathered query into the output (a collective cannot write a kernel output), each program half the ranks
    half = nr // n_prg if nr >= n_prg else nr
    r0 = pid * half if nr >= n_prg else 0
    if nr >= n_prg or pid == 0:
        gq = nl.ndarray((T, half, L + RP), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=gq[0:T, 0:half, 0:L + RP],
                      src=qall.ap(pattern=[[L + RP, T], [T * (L + RP), half], [1, L + RP]], offset=r0 * T * (L + RP)))
        # token-major [T, rank, L + rope], the attention's layout (an XLA transpose of [rank, T, ...] cost ~30 us)
        nisa.dma_copy(dst=query.ap(pattern=[[nr * (L + RP), T], [L + RP, half], [1, L + RP]], offset=r0 * (L + RP)),
                      src=gq[0:T, 0:half, 0:L + RP])
    if gate is not None:
        return query, compact, q_lat, cache, gate_x
    return query, compact, q_lat, cache


def qkv_shard_layout(q_a, kv_a, gate, rank: int, ranks: int):
    """attn_in_decode's ``qkv_sh`` for ``rank`` of ``ranks``: [128, H / 128, CQ + CK + V] bf16 from q_a [H, QL], kv_a
    [H, NKV] (replicated) and this rank's gate [H, V]: its CQ = QL / ranks q_a columns, CK = NKV / ranks kv_a columns
    and the gate, partition-major (pack[p, kb, :] = row 128 kb + p)."""
    import torch

    H, QL = q_a.shape
    NKV = kv_a.shape[1]
    cq, ck = QL // ranks, NKV // ranks
    w = torch.cat([q_a[:, rank * cq:(rank + 1) * cq], kv_a[:, rank * ck:(rank + 1) * ck], gate], dim=1)
    return w.reshape(H // P, P, -1).permute(1, 0, 2).contiguous()
