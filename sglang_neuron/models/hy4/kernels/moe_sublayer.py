# SPDX-License-Identifier: Apache-2.0
"""Hy4's decode MoE sublayer in one NKI kernel, its all-reduce included (the giga kernel's MoE half).

From the hyper-connection's MLP input to the all-reduced MLP output, for T <= 8 tokens:

    RMSNorm -> router logits, sigmoid, biased top-8, renormalized weights -> shared expert (this rank's rows)
        -> the hit local experts (moe_wide's 512-column moving tiles) -> bf16 -> all-reduce over the attention TP

In the XLA graph this is ~15 ops a layer around two kernels and an XLA all-reduce, each with its own start-up (an NKI
kernel's preamble alone is ~15-30 us, prof_nx/kcap_mw_pm1). Here the start-up is paid once, the router and shared
expert weights stream from the kernel's first instruction (their rows laid out partition-major at load, so every DMA
moves long contiguous runs: a 512 B router row split into small packets and loaded at ~140 GB/s,
prof_nx/kcap_mblk3n), and the routing runs while they stream. The hyper-connection's post update is left to the next
sublayer's fused post + pre (hc_decode).

The two programs of a logical core split the up projections by intermediate columns (each expert's gate / up, the
shared expert's rows) and the down projections by hidden columns: after its gate / up a program hands its half of the
activation to the other through shared HBM (a core barrier), then computes the down projection's hidden half over
all of the intermediate dimension, so no partial outputs are summed. The shared expert's down projection uses the
routed experts' diagonal layout (four copies of its activation a tile, unscaled) and seeds their accumulator. Both
programs issue the all-reduce of the bf16 result (routed + shared summed in fp32 and rounded, as the XLA graph)."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from sglang_neuron.models.hy4.kernels.moe_wide import GW, KB_DMA, NT, TILE, W

P = 128
DMA_ENGINES = (nisa.engine.sync, nisa.engine.scalar)


def moe_sublayer_decode(x, ln_w, router_pm, router_bias, base, gate_up, gate_up_scale, down, down_scale, sh_gate_up_pm,
                        sh_down, ranks=(), top_k=8, scaling=1.0, limit=10.0, eps=1e-6, tag=""):
    """x [T, H] bf16 (the MLP's hyper-connection input, before the norm); ln_w [1, H] bf16; router_pm [128, KH * NE]
    bf16 (router_partition_major); router_bias [1, NE] fp32; base [1, 1] int32 (this rank's first expert);
    gate_up [E, H, 2I] / gate_up_scale / down / down_scale (moe_wide's); sh_gate_up_pm [n_prg, 128, KH * 2 RH] bf16
    (shared_partition_major); sh_down [RS, H] bf16; ranks: the all-reduce's group (every attention-TP rank; empty:
    none, a one-rank test) -> the MLP output [T, H] bf16, all-reduced."""
    T, H = x.shape
    KH = H // P
    NE = router_bias.shape[1]
    E, I, _ = down.shape  # noqa: E741
    RS = sh_down.shape[0]
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    RH = RS // n_prg
    IH = I // n_prg
    i0 = pid * IH
    GB = 2 * I // P
    HB = H // P
    KA = I // P  # the down's row blocks (all of I)
    KI = IH // P  # this program's activation blocks
    HH = H // n_prg  # this program's hidden columns
    h0 = pid * HH
    HBH = HH // P
    NG = HH // W  # its 512-column groups (6)
    R = -(-NG // NT)  # their rounds over the four tiles (2)
    S = GW * T
    group = ReplicaGroup([list(g) for g in ranks]) if ranks else None

    out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="moe_sub_out" + tag)
    red = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="moe_sub_red" + tag)
    hx = nl.ndarray((n_prg, P, KI * T), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="moe_sub_hx" + tag)
    sx = nl.ndarray((RS, T), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="moe_sub_sx" + tag)

    # ---- the router's and the shared expert's weights first: they stream while the norm runs -------------------
    rw = nl.ndarray((P, KH, NE), dtype=router_pm.dtype, buffer=nl.sbuf)
    half = KH // 2
    for j, eng in enumerate(DMA_ENGINES):
        nisa.dma_copy(dst=rw[0:P, j * half:(j + 1) * half, 0:NE],
                      src=router_pm.ap(pattern=[[KH * NE, P], [NE, half], [1, NE]], offset=j * half * NE),
                      dge_mode=nisa.dge_mode.hwdge, engine=eng)
    sw = nl.ndarray((P, KH, 2 * RH), dtype=sh_gate_up_pm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sw[0:P, 0:KH, 0:2 * RH],
                  src=sh_gate_up_pm.ap(pattern=[[KH * 2 * RH, P], [2 * RH, KH], [1, 2 * RH]],
                                       offset=pid * P * KH * 2 * RH),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    sd = nl.ndarray((RS, HH), dtype=sh_down.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sd[0:RS, 0:HH], src=sh_down.ap(pattern=[[H, RS], [1, HH]], offset=h0),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)

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

    # ---- router logits [T, NE] and the shared expert's gate / up [T, 2 RH] -----------------------------------------
    lg = nl.ndarray((T, NE), dtype=nl.float32, buffer=nl.psum)
    sgu = nl.ndarray((T, 2 * RH), dtype=nl.float32, buffer=nl.psum)
    for kb in range(KH):
        nisa.nc_matmul(dst=lg[0:T, 0:NE], stationary=xs[0:P, kb, 0:T], moving=rw[0:P, kb, 0:NE], accumulate=None)
        nisa.nc_matmul(dst=sgu[0:T, 0:2 * RH], stationary=xs[0:P, kb, 0:T], moving=sw[0:P, kb, 0:2 * RH],
                       accumulate=None)
    # ---- router logits [T, NE] and the shared expert's gate / up [T, 2 RH] -----------------------------------------
    lg = nl.ndarray((T, NE), dtype=nl.float32, buffer=nl.psum)
    sgu = nl.ndarray((T, 2 * RH), dtype=nl.float32, buffer=nl.psum)
    for kb in range(KH):
        nisa.nc_matmul(dst=lg[0:T, 0:NE], stationary=xs[0:P, kb, 0:T], moving=rw[0:P, kb, 0:NE], accumulate=None)
        nisa.nc_matmul(dst=sgu[0:T, 0:2 * RH], stationary=xs[0:P, kb, 0:T], moving=sw[0:P, kb, 0:2 * RH],
                       accumulate=None)
    # the shared expert's activation [T, RH] of this program's rows, handed to the other program: shT [RS, T]
    sg = nl.ndarray((T, RH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg[0:T, 0:RH], op=nl.silu, data=sgu[0:T, 0:RH])
    sh = nl.ndarray((T, RH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sh[0:T, 0:RH], data1=sg[0:T, 0:RH], data2=sgu[0:T, RH:2 * RH], op=nl.multiply)
    TA = T + T % 2
    tps = nl.ndarray((RH, TA), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=tps[0:RH, 0:T], data=sh[0:T, 0:RH])
    shTm = nl.ndarray((RH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=shTm[0:RH, 0:T], src=tps[0:RH, 0:T])
    nisa.dma_copy(dst=sx.ap(pattern=[[T, RH], [1, T]], offset=pid * RH * T), src=shTm[0:RH, 0:T])

    # ---- the choice: sigmoid, + bias, top 8, the chosen scores renormalized, each local expert's weight -------------
    sc = nl.ndarray((T, NE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sc[0:T, 0:NE], op=nl.sigmoid, data=lg[0:T, 0:NE])
    rb = nl.ndarray((T, NE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=rb[0:T, 0:NE], src=router_bias.ap(pattern=[[0, T], [1, NE]], offset=0))
    sb = nl.ndarray((T, NE), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sb[0:T, 0:NE], data1=sc[0:T, 0:NE], data2=rb[0:T, 0:NE], op=nl.add)
    v8 = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.max8(dst=v8[0:T, 0:top_k], src=sb[0:T, 0:NE])
    i8 = nl.ndarray((T, top_k), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=i8[0:T, 0:top_k], data=sb[0:T, 0:NE], vals=v8[0:T, 0:top_k])
    ch = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=ch[0:T, 0:top_k], data=sc[0:T, 0:NE], indices=i8[0:T, 0:top_k])
    ssum = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=ssum[0:T, 0:1], op=nl.add, data=ch[0:T, 0:top_k], axis=1)
    nisa.tensor_scalar(dst=ssum[0:T, 0:1], data=ssum[0:T, 0:1], op0=nl.add, operand0=1e-20)
    rs = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rs[0:T, 0:1], data=ssum[0:T, 0:1])
    nisa.tensor_scalar(dst=rs[0:T, 0:1], data=rs[0:T, 0:1], op0=nl.multiply, operand0=scaling)
    chn = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=chn[0:T, 0:top_k], data=ch[0:T, 0:top_k], op0=nl.multiply, operand0=rs[0:T, 0:1])
    idf = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idf[0:T, 0:top_k], src=i8[0:T, 0:top_k])
    bf = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bf[0:T, 0:1], src=base.ap(pattern=[[0, T], [1, 1]], offset=0))
    rel = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rel[0:T, 0:top_k], data=idf[0:T, 0:top_k], op0=nl.subtract, operand0=bf[0:T, 0:1])
    A = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    for e in range(E):
        m = nl.ndarray((T, top_k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=m[0:T, 0:top_k], data=rel[0:T, 0:top_k], op0=nl.equal, operand0=float(e))
        nisa.tensor_tensor(dst=m[0:T, 0:top_k], data1=m[0:T, 0:top_k], data2=chn[0:T, 0:top_k], op=nl.multiply)
        nisa.tensor_reduce(dst=A[0:T, e:e + 1], op=nl.add, data=m[0:T, 0:top_k], axis=1)
    # affT [E, T] (the expert loop reads its expert's row) and the hit experts' list
    affT = nl.ndarray((E, T), dtype=nl.float32, buffer=nl.private_hbm, name="moe_sub_aff" + tag)
    nisa.dma_copy(dst=affT.ap(pattern=[[1, T], [T, E]], offset=0), src=A[0:T, 0:E])
    hitm = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hitm[0:T, 0:E], data=A[0:T, 0:E], op0=nl.greater, operand0=0.0)
    onet = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onet[0:T, 0:1], value=1.0)
    cnt_ps = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt_ps[0:1, 0:E], stationary=onet[0:T, 0:1], moving=hitm[0:T, 0:E])
    hf = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=hf[0:P, 0:E], value=0.0)
    nisa.tensor_scalar(dst=hf[0:1, 0:E], data=cnt_ps[0:1, 0:E], op0=nl.minimum, operand0=1.0)
    nz = nl.ndarray((P, E + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=nz[0:P, 0:E + 1], src=hf[0:P, 0:E], index_offset=0, padding_val=0)
    hits = nl.ndarray((1, E + 1), dtype=nl.int32, buffer=nl.private_hbm, name="moe_sub_hits" + tag)
    nisa.dma_copy(dst=hits.ap(pattern=[[E + 1, 1], [1, E + 1]], offset=0), src=nz[0:1, 0:E + 1])
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, nz[0:1, E:E + 1])
    slot = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=slot[0:1, 0:1], value=0)

    # ---- the shared expert's down projection seeds the accumulator, in the diagonal layout ----------------------
    nisa.core_barrier(sx, (0, 1))
    sh4 = nl.ndarray((RS, S), dtype=nl.bfloat16, buffer=nl.sbuf)  # four copies of the activation, a tile's columns
    nisa.dma_copy(dst=sh4.ap(pattern=[[S, RS], [T, GW], [1, T]], offset=0),
                  src=sx.ap(pattern=[[T, RS], [0, GW], [1, T]], offset=0))
    acc = nl.ndarray((P, R, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc[0:P, 0:R, 0:W], value=0.0)
    spd = nl.ndarray((P, R, W), dtype=nl.float32, buffer=nl.psum)
    for grp in range(NG):
        r, c = grp // NT, grp % NT
        nisa.nc_matmul(dst=spd[c * TILE:c * TILE + S, r, 0:W], stationary=sh4[0:RS, 0:S],
                       moving=sd[0:RS, grp * W:(grp + 1) * W], tile_position=(0, c * TILE), tile_size=(RS, TILE),
                       accumulate=None)
    for grp in range(NG):
        r, c = grp // NT, grp % NT
        nisa.tensor_copy(dst=acc[c * TILE:c * TILE + S, r, 0:W], src=spd[c * TILE:c * TILE + S, r, 0:W])

    def tiled(dst, stationary, moving, c):
        nisa.nc_matmul(dst=dst, stationary=stationary, moving=moving, tile_position=(0, c * TILE),
                       tile_size=(P, TILE), accumulate=None)

    # ---- the hit local experts -------------------------------------------------------------------------------------
    def expert_body(it):
        e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e[0:1, 0:1], src=hits.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=slot,
                                                     indirect_dim=1))
        gs = nl.ndarray((P, KH, NT, GW), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(2):
            nisa.dma_copy(dst=gs.ap(pattern=[[KH * NT * GW, P], [NT * GW, KH], [GW, 2], [1, GW]], offset=g * 2 * GW),
                          src=gate_up_scale.ap(pattern=[[0, P], [GB, KH], [GW, 2], [1, GW]],
                                               offset=(g * I + i0) // P, scalar_offset=e, indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[g])
        aw = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=aw[0:P, 0:T], src=affT.ap(pattern=[[0, P], [1, T]], offset=0, scalar_offset=e,
                                                       indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        ds = nl.ndarray((P, KA, HBH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds[0:P, 0:KA, 0:HBH], src=down_scale.ap(pattern=[[0, P], [HB, KA], [1, HBH]],
                                                                    offset=h0 // P, scalar_offset=e, indirect_dim=0),
                      dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
        # gate / up of this program's intermediate columns (tiles 0, 1 the gate's halves, 2, 3 the up's)
        gu = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.psum)
        # the down's weights (all of I, this program's hidden half, 3 KB rows) interleaved with the gate / up's on the
        # two DMA queues: both stream at once and the queues never idle between them (issued after, the down's 6 MB
        # waited for the gate / up's 12 MB and ran at ~180 GB/s)
        wd = nl.ndarray((P, KA, HH), dtype=down.dtype, buffer=nl.sbuf)
        NGRP = KH // KB_DMA
        wd_at = {g: [ki for ki in range(KA) if ki * NGRP // KA == g] for g in range(NGRP)}
        for kb in range(KH):
            if kb % KB_DMA == 0:
                wts = nl.ndarray((P, KB_DMA, 2, IH), dtype=gate_up.dtype, buffer=nl.sbuf)
                for g in range(2):
                    nisa.dma_copy(dst=wts[0:P, 0:KB_DMA, g, 0:IH],
                                  src=gate_up.ap(pattern=[[2 * I, P], [P * 2 * I, KB_DMA], [1, IH]],
                                                 offset=kb * P * 2 * I + g * I + i0, scalar_offset=e, indirect_dim=0),
                                  dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[(kb // KB_DMA + g) % 2])
                for ki in wd_at[kb // KB_DMA]:
                    nisa.dma_copy(dst=wd[0:P, ki, 0:HH], src=down.ap(pattern=[[H, P], [1, HH]], offset=ki * P * H + h0,
                                                                       scalar_offset=e, indirect_dim=0),
                                  dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[ki % 2])
            xsc = nl.ndarray((P, NT, S), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=xsc.ap(pattern=[[NT * S, P], [S, NT], [T, GW], [1, T]], offset=0),
                               data1=xs.ap(pattern=[[KH * T, P], [0, NT], [0, GW], [1, T]], offset=kb * T),
                               data2=gs.ap(pattern=[[KH * NT * GW, P], [GW, NT], [1, GW], [0, T]],
                                           offset=kb * NT * GW), op=nl.multiply)
            for c in range(NT):
                tiled(gu[c * TILE:c * TILE + S, 0:W], xsc[0:P, c, 0:S],
                      wts[0:P, kb % KB_DMA, c // 2, (c % 2) * W:(c % 2 + 1) * W], c)
        gsb = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gsb[0:P, 0:W], src=gu[0:P, 0:W])
        tq = nl.ndarray((P, GW, P), dtype=nl.float32, buffer=nl.psum)
        for q in range(GW):
            nisa.nc_transpose(dst=tq[0:P, q, 0:P], data=gsb[0:P, q * P:(q + 1) * P])
        hT = nl.ndarray((P, KI, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for q in range(GW):
            gt = nl.ndarray((P, 2, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=gt[0:P, 0:2, 0:T],
                               data=tq.ap(pattern=[[GW * P, P], [TILE, 2], [1, T]], offset=q * P + q * T),
                               op0=nl.minimum, operand0=limit)
            nisa.activation(dst=gt[0:P, 0:2, 0:T], op=nl.silu, data=gt[0:P, 0:2, 0:T])
            up = nl.ndarray((P, 2, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=up[0:P, 0:2, 0:T],
                               data=tq.ap(pattern=[[GW * P, P], [TILE, 2], [1, T]], offset=q * P + 2 * TILE + q * T),
                               op0=nl.minimum, operand0=limit, op1=nl.maximum, operand1=-limit)
            nisa.tensor_tensor(dst=gt[0:P, 0:2, 0:T], data1=gt[0:P, 0:2, 0:T], data2=up[0:P, 0:2, 0:T],
                               op=nl.multiply)
            nisa.tensor_tensor(dst=hT.ap(pattern=[[KI * T, P], [GW * T, 2], [1, T]], offset=q * T),
                               data1=gt[0:P, 0:2, 0:T], data2=aw.ap(pattern=[[T, P], [0, 2], [1, T]], offset=0),
                               op=nl.multiply)
        # the activation's halves meet: hA[p, prg * KI + ki, t]
        nisa.dma_copy(dst=hx.ap(pattern=[[KI * T, P], [1, KI * T]], offset=pid * P * KI * T), src=hT[0:P, 0:KI, 0:T])
        nisa.core_barrier(hx, (0, 1))
        hA = nl.ndarray((P, KA, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=hA[0:P, 0:KA, 0:T], src=hx.ap(pattern=[[KI * T, P], [P * KI * T, n_prg], [1, KI * T]],
                                                         offset=0))
        nisa.core_barrier(hx, (0, 1))  # both have read before either writes the next expert's
        hs = nl.ndarray((P, KA, HBH * T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hs.ap(pattern=[[KA * HBH * T, P], [HBH * T, KA], [T, HBH], [1, T]], offset=0),
                           data1=hA.ap(pattern=[[KA * T, P], [T, KA], [0, HBH], [1, T]], offset=0),
                           data2=ds.ap(pattern=[[KA * HBH, P], [HBH, KA], [1, HBH], [0, T]], offset=0), op=nl.multiply)
        pd = nl.ndarray((P, R, W), dtype=nl.float32, buffer=nl.psum)
        for ki in range(KA):
            for grp in range(NG):
                r, c = grp // NT, grp % NT
                tiled(pd[c * TILE:c * TILE + S, r, 0:W], hs[0:P, ki, grp * S:(grp + 1) * S],
                      wd[0:P, ki, grp * W:(grp + 1) * W], c)
        for grp in range(NG):
            r, c = grp // NT, grp % NT
            nisa.tensor_tensor(dst=acc[c * TILE:c * TILE + S, r, 0:W], data1=acc[c * TILE:c * TILE + S, r, 0:W],
                               data2=pd[c * TILE:c * TILE + S, r, 0:W], op=nl.add)
        nisa.tensor_scalar(dst=slot[0:1, 0:1], data=slot[0:1, 0:1], op0=nl.add, operand0=1)

    nl.fori_loop(0, n_reg, expert_body)

    # ---- bf16, this program's hidden half of the result (the diagonal), all-reduced --------------------------------
    accb = nl.ndarray((P, R, W), dtype=nl.bfloat16, buffer=nl.sbuf)
    for grp in range(NG):
        r, c = grp // NT, grp % NT
        nisa.tensor_copy(dst=accb[c * TILE:c * TILE + S, r, 0:W], src=acc[c * TILE:c * TILE + S, r, 0:W])
    # tile c's groups c, 4 + c, ... in one DMA a (tile, block)
    k = 0
    for c in range(NT):
        rc = len(range(c, NG, NT))
        for j in range(GW):
            nisa.dma_copy(dst=red.ap(pattern=[[H, T], [NT * W, rc], [1, P]], offset=h0 + c * W + j * P),
                          src=accb[c * TILE + j * T:c * TILE + (j + 1) * T, 0:rc, j * P:(j + 1) * P],
                          dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[k % 2])
            k += 1
    nisa.core_barrier(red, (0, 1))
    if group is not None:
        ncc.all_reduce(srcs=[red], dsts=[out], replica_group=group, op=nl.add)
    else:
        cp = nl.ndarray((T, HH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=cp[0:T, 0:HH], src=red.ap(pattern=[[H, T], [1, HH]], offset=h0))
        nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, HH]], offset=h0), src=cp[0:T, 0:HH])
    return out


def router_partition_major(router):
    """The router [H, NE] as [128, H / 128 * NE]: [p, kb * NE + j] = router[128 kb + p, j]."""
    H, NE = router.shape
    return router.view(H // P, P, NE).permute(1, 0, 2).contiguous().view(P, -1)


def shared_partition_major(gate_up, n_prg: int = 2):
    """A shared expert's gate_up [H, 2 Rs] (gate then up columns) as [n_prg, 128, H / 128 * 2 Rs / n_prg]: each
    program's gate rows then its up rows (moe_block.shared_gate_up_layout), partition-major."""
    from sglang_neuron.models.hy4.kernels.moe_block import shared_gate_up_layout

    lay = shared_gate_up_layout(gate_up, n_prg)  # [H, n_prg, 2 RH]
    H = lay.shape[0]
    return lay.view(H // P, P, n_prg, -1).permute(2, 1, 0, 3).contiguous().view(n_prg, P, -1)
