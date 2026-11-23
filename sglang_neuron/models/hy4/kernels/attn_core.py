# SPDX-License-Identifier: Apache-2.0
"""Hy4's decode attention core in one NKI kernel, its three collectives included.

From every head's absorbed query to the all-reduced attention output, for a decode of T <= 8 rows (padded to whole
row blocks of 2 tokens x 64 heads):

    each head over this rank's selected keys (mla_sparse's body: the keys gathered, transposed, two-pass softmax)
        -> all-gather of the ranks' softmax statistics -> each head's partial rescaled (the sinks counted once)
        -> reduce-scatter over the heads (this rank keeps its own head) -> the value up-projection, the sigmoid gate,
        o_proj -> all-reduce

In the XLA graph these are ~25 ops around three collectives (prof_nx/vg7d: ~75 us from the query's all-gather to the
statistics' all-gather alone). Rounding follows the graph's (Attention.merge's small-row path, Attention.output): the
partials and their rescaling in fp32, the attended latent rounded to bf16, the values and the gated product bf16."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from sglang_neuron.models.hy4.kernels.mla_sparse import mla_sparse

P = 128
W = 512


def attn_core_decode(q, cache, slots, bias, onehot, sinks, w_uv, gate_x, o, ranks=(), scale=1.0, t_real=0, pre=None,
                     hooks=None, tag=""):
    """q [NB, 128, D] bf16 (every head's query, 2 tokens a row block); cache [R, D] bf16; slots [NB, S] int32 and bias
    [NB, 2, S] bf16 (mla_sparse.block_slots); onehot [2, 128] bf16; sinks [1, H] fp32 (every head's); w_uv [V, L]
    bf16 (this rank's head); gate_x [T, V] bf16 (sigmoid(x @ gate), the graph's); o [V, Hd] bf16; ranks: the attention
    TP (one head a rank; empty: no collectives, a one-rank test); t_real: the real rows (<= 2 NB) -> [T, Hd] bf16, the
    all-reduced attention output."""
    NB, _, D = q.shape
    H = P // 2  # heads (2 tokens a row block)
    T = t_real or 2 * NB
    V, L = w_uv.shape
    Hd = o.shape[1]
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    nr = len(ranks[0]) if ranks else 1
    group = ReplicaGroup([list(g) for g in ranks]) if ranks else None
    hp = H // nr if ranks else H  # heads a rank keeps after the reduce-scatter

    out_final = nl.ndarray((T, Hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_core_out" + tag)
    ml_all = nl.ndarray((nr, H, 2 * NB, 2), dtype=nl.float32, buffer=nl.shared_hbm, name="attn_core_ml_all" + tag)
    sc = nl.ndarray((H, 2 * NB, L), dtype=nl.float32, buffer=nl.shared_hbm, name="attn_core_scaled" + tag)
    att = nl.ndarray((hp, 2 * NB, L), dtype=nl.float32, buffer=nl.shared_hbm, name="attn_core_att" + tag)
    red = nl.ndarray((T, Hd), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="attn_core_red" + tag)

    HH = Hd // n_prg
    h0 = pid * HH
    KV = V // P
    if pre is not None:
        wv, owk = pre  # prefetched by the caller (load_out_weights)
    else:
        # the output projections' weights first: they stream during the attention
        wv, owk = load_out_weights(w_uv, o)

    # ---- this rank's partial softmax of every head (mla_sparse: out [H, 2NB, L] fp32, ml [H, 2NB, 2]) -------------
    out, ml = mla_sparse(q, cache, slots, bias, onehot, scale=scale, tag=tag)
    nisa.core_barrier(out, (0, 1))
    nisa.core_barrier(ml, (0, 1))
    TT = 2 * NB

    # ---- the statistics of every rank; each head's partial rescaled ------------------------------------------------
    hooks = hooks or {}
    if "ml" in hooks:
        hooks["ml"]()  # the caller's prefetch DMAs, during the statistics' all-gather
    if group is not None:
        ncc.all_gather(srcs=[ml], dsts=[ml_all], replica_group=group, collective_dim=0)
    else:
        cp = nl.ndarray((H, TT, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cp[0:H, 0:TT, 0:2], src=ml.ap(pattern=[[TT * 2, H], [2, TT], [1, 2]], offset=0))
        nisa.dma_copy(dst=ml_all.ap(pattern=[[TT * 2, H], [2, TT], [1, 2]], offset=0), src=cp[0:H, 0:TT, 0:2])
    nisa.core_barrier(ml_all, (0, 1))
    # heads on partitions: mla[h, r, t, 0 / 1]
    mla = nl.ndarray((H, nr, TT, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=mla[0:H, 0:nr, 0:TT, 0:2], src=ml_all.ap(pattern=[[TT * 2, H], [H * TT * 2, nr], [2, TT], [1, 2]],
                                                                offset=0))
    snk = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=snk[0:H, 0:1], src=sinks.ap(pattern=[[1, H], [1, 1]], offset=0))
    big = nl.ndarray((H, TT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=big[0:H, 0:TT], op=nl.maximum,
                       data=mla.ap(pattern=[[nr * TT * 2, H], [2, TT], [TT * 2, nr]], offset=0), axis=2)
    nisa.tensor_scalar(dst=big[0:H, 0:TT], data=big[0:H, 0:TT], op0=nl.maximum, operand0=snk[0:H, 0:1])
    # total = sum_r exp(m_r - M) l_r + exp(sink - M)
    e = nl.ndarray((H, nr, TT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=e[0:H, 0:nr, 0:TT], data1=mla.ap(pattern=[[nr * TT * 2, H], [TT * 2, nr], [2, TT]], offset=0),
                       data2=big.ap(pattern=[[TT, H], [0, nr], [1, TT]], offset=0), op=nl.subtract)
    nisa.activation(dst=e[0:H, 0:nr, 0:TT], op=nl.exp, data=e[0:H, 0:nr, 0:TT])
    nisa.tensor_tensor(dst=e[0:H, 0:nr, 0:TT], data1=e[0:H, 0:nr, 0:TT],
                       data2=mla.ap(pattern=[[nr * TT * 2, H], [TT * 2, nr], [2, TT]], offset=1), op=nl.multiply)
    tot = nl.ndarray((H, TT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=tot[0:H, 0:TT], op=nl.add, data=e.ap(pattern=[[nr * TT, H], [1, TT], [TT, nr]], offset=0),
                       axis=2)
    es = nl.ndarray((H, TT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=es[0:H, 0:TT], data=big[0:H, 0:TT], op0=nl.subtract, operand0=snk[0:H, 0:1], op1=nl.multiply,
                       operand1=-1.0)
    nisa.activation(dst=es[0:H, 0:TT], op=nl.exp, data=es[0:H, 0:TT])
    nisa.tensor_tensor(dst=tot[0:H, 0:TT], data1=tot[0:H, 0:TT], data2=es[0:H, 0:TT], op=nl.add)
    # this rank's factor: exp(m - M) / total
    ms = nl.ndarray((H, TT, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ms[0:H, 0:TT, 0:2], src=ml.ap(pattern=[[TT * 2, H], [2, TT], [1, 2]], offset=0))
    f = nl.ndarray((H, TT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=f[0:H, 0:TT], data1=ms.ap(pattern=[[TT * 2, H], [2, TT]], offset=0), data2=big[0:H, 0:TT],
                       op=nl.subtract)
    nisa.activation(dst=f[0:H, 0:TT], op=nl.exp, data=f[0:H, 0:TT])
    rt = nl.ndarray((H, TT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rt[0:H, 0:TT], data=tot[0:H, 0:TT])
    nisa.tensor_tensor(dst=f[0:H, 0:TT], data1=f[0:H, 0:TT], data2=rt[0:H, 0:TT], op=nl.multiply)
    # the scaled partial [H, TT, L], each program half of the tokens
    tpp = TT // n_prg
    t0 = pid * tpp
    po = nl.ndarray((H, tpp, L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=po[0:H, 0:tpp, 0:L], src=out.ap(pattern=[[TT * L, H], [L, tpp], [1, L]], offset=t0 * L))
    nisa.tensor_tensor(dst=po[0:H, 0:tpp, 0:L], data1=po[0:H, 0:tpp, 0:L],
                       data2=f.ap(pattern=[[TT, H], [1, tpp], [0, L]], offset=t0), op=nl.multiply)
    nisa.dma_copy(dst=sc.ap(pattern=[[TT * L, H], [L, tpp], [1, L]], offset=t0 * L), src=po[0:H, 0:tpp, 0:L])
    nisa.core_barrier(sc, (0, 1))
    if "rs" in hooks:
        hooks["rs"]()
    if group is not None:
        ncc.reduce_scatter(srcs=[sc], dsts=[att], op=nl.add, replica_group=group, collective_dim=0)
    else:
        cpa = nl.ndarray((H, TT, L), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=cpa[0:H, 0:TT, 0:L], src=sc.ap(pattern=[[TT * L, H], [L, TT], [1, L]], offset=0))
        nisa.dma_copy(dst=att.ap(pattern=[[TT * L, H], [L, TT], [1, L]], offset=0), src=cpa[0:H, 0:TT, 0:L])
    nisa.core_barrier(att, (0, 1))

    # ---- the output: values = bf16(att) @ w_uv^T, gated, @ o (this program's columns), all-reduced -----------------
    # (one head a rank: att[0] is [TT, L])
    KL = L // P
    av = nl.ndarray((T, L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=av[0:T, 0:L], src=att.ap(pattern=[[L, T], [1, L]], offset=0))
    ab = nl.ndarray((T, L), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ab[0:T, 0:L], src=av[0:T, 0:L])
    TA = T + T % 2
    abp = nl.ndarray((P, KL, TA), dtype=nl.bfloat16, buffer=nl.psum)
    for k in range(KL):
        nisa.nc_transpose(dst=abp[0:P, k, 0:T], data=ab[0:T, k * P:(k + 1) * P])
    abT = nl.ndarray((P, KL, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=abT[0:P, 0:KL, 0:T], src=abp[0:P, 0:KL, 0:T])
    # w_uv^T [L, V] by blocks: wvT[:, k, :] = w_uv[:, 128 k: 128 k + 128]^T
    wvT = nl.ndarray((P, KL, V), dtype=w_uv.dtype, buffer=nl.sbuf)
    for k in range(KL):
        for j in range(KV):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=wv[0:P, j, k * P:(k + 1) * P])
            nisa.tensor_copy(dst=wvT[0:P, k, j * P:(j + 1) * P], src=tp[0:P, 0:P])
    vp = nl.ndarray((T, V), dtype=nl.float32, buffer=nl.psum)
    for k in range(KL):
        nisa.nc_matmul(dst=vp[0:T, 0:V], stationary=abT[0:P, k, 0:T], moving=wvT[0:P, k, 0:V], accumulate=None)
    vb = nl.ndarray((T, V), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vb[0:T, 0:V], src=vp[0:T, 0:V])
    gx = nl.ndarray((T, V), dtype=gate_x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gx[0:T, 0:V], src=gate_x.ap(pattern=[[V, T], [1, V]], offset=0))
    gd = nl.ndarray((T, V), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=gd[0:T, 0:V], data1=vb[0:T, 0:V], data2=gx[0:T, 0:V], op=nl.multiply)
    gdp = nl.ndarray((P, KV, TA), dtype=nl.bfloat16, buffer=nl.psum)
    for j in range(KV):
        nisa.nc_transpose(dst=gdp[0:P, j, 0:T], data=gd[0:T, j * P:(j + 1) * P])
    gdT = nl.ndarray((P, KV, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gdT[0:P, 0:KV, 0:T], src=gdp[0:P, 0:KV, 0:T])
    ob = nl.ndarray((T, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
    for g in range(HH // W):
        op_ = nl.ndarray((T, W), dtype=nl.float32, buffer=nl.psum)
        for j in range(KV):
            nisa.nc_matmul(dst=op_[0:T, 0:W], stationary=gdT[0:P, j, 0:T], moving=owk[0:P, j, g * W:(g + 1) * W],
                           accumulate=None)
        nisa.tensor_copy(dst=ob[0:T, g * W:(g + 1) * W], src=op_[0:T, 0:W])
    nisa.dma_copy(dst=red.ap(pattern=[[Hd, T], [1, HH]], offset=h0), src=ob[0:T, 0:HH])
    nisa.core_barrier(red, (0, 1))
    if "ar" in hooks:
        hooks["ar"]()
    if group is not None:
        ncc.all_reduce(srcs=[red], dsts=[out_final], replica_group=group, op=nl.add)
    else:
        cpo = nl.ndarray((T, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=cpo[0:T, 0:HH], src=red.ap(pattern=[[Hd, T], [1, HH]], offset=h0))
        nisa.dma_copy(dst=out_final.ap(pattern=[[Hd, T], [1, HH]], offset=h0), src=cpo[0:T, 0:HH])
    return out_final


def load_out_weights(w_uv, o):
    """(wv [128, V / 128, L], owk [128, V / 128, Hd / n_prg]): the output projections' weights of this program, in
    SBUF (attn_core_decode's ``pre``)."""
    V, L = w_uv.shape
    Hd = o.shape[1]
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    KV = V // P
    HH = Hd // n_prg
    wv = nl.ndarray((P, KV, L), dtype=w_uv.dtype, buffer=nl.sbuf)  # wv[p, j, :] = w_uv[128 j + p, :]
    nisa.dma_copy(dst=wv[0:P, 0:KV, 0:L], src=w_uv.ap(pattern=[[L, P], [P * L, KV], [1, L]], offset=0),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    owk = nl.ndarray((P, KV, HH), dtype=o.dtype, buffer=nl.sbuf)  # owk[p, j, :] = o[128 j + p, this half]
    nisa.dma_copy(dst=owk[0:P, 0:KV, 0:HH], src=o.ap(pattern=[[Hd, P], [P * Hd, KV], [1, HH]], offset=pid * HH),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
    return wv, owk
