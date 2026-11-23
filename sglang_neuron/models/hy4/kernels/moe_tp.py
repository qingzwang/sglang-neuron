# SPDX-License-Identifier: Apache-2.0
"""Hy4's decode routed experts sharded TP-16 x EP-4: every hit expert's work spread over 16 ranks.

With EP over the 64 ranks (4 whole experts a rank) a decode step waits at every MoE all-reduce for the rank that
drew the most of its tokens' experts: at 7 rows (a DFly verify) ~3 experts on the slowest rank against ~0.9 on the
mean one, ~130 us a layer (prof_nx/vg7p vs the served step). Here the ranks form 4 EP groups of 16, each group holding
64 experts, and each rank of a group holds 1/16 of each: the expert's gate and up columns [128 j, 128 j + 128) and
its down rows [128 j, 128 j + 128), j the rank's index in its group. A hit expert is then 2.4 MB of weights on each
of its group's 16 ranks instead of 37.7 MB on one, and the hits of a group spread over its ranks evenly.

The slices of 128 columns are one FP8 scale block wide, so two experts' gate and up slices (4 blocks, 4 scales) fill
one 512-column moving tile and moe_wide's diagonal (4 pre-scaled copies of the tokens a column tile) applies, as in
the down projection (4 column blocks of H). The hit experts are processed 8 slots a device-loop iteration, each slot's
weights a hardware-DGE DMA at the expert from the hit list, a slot past the hits out of range (skipped: no data moves;
its affinity is zero). The two programs split each chunk's gate / up by slots and the down by hidden columns, the
activations meeting in shared HBM.

Layouts (built at load, tp_layouts): gate_up [EL, 128, KH, 256] fp8 partition-major (gate slice, then up slice);
gate_up_scale [EL, KH, 2] fp32; down [EL, 128, H] fp8; down_scale [EL, H / 128] fp32."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
TILE = 32
NT = P // TILE
W = 512
CH = 8  # slots a chunk
DMA_ENGINES = (nisa.engine.sync, nisa.engine.scalar)


def moe_tp_decode(xT, affT, hits, n_chunks, gate_up, gate_up_scale, down, down_scale, limit=10.0, tag=""):
    """xT [H, T] bf16 (the normed tokens, T <= 8); affT [EL, T] fp32 (each local expert's weight of each token);
    hits [1, EL + CH] int32 (the hit local experts first, then EL: out of range); n_chunks [1, 1] int32 (ceil(hits /
    CH)) -> [T, H] fp32, this rank's partial of the routed experts' output."""
    H, T = xT.shape
    EL, _, KH, C2 = gate_up.shape
    SW = C2 // 2  # a slice's columns of the gate (and of the up): 128
    KI = down.shape[1] // P  # the down's row blocks of a slice: 1
    HB = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    SP = CH // n_prg  # slots a program's gate / up takes
    NQ = SP // 2  # their pairs (one moving tile each)
    HH = H // n_prg
    h0 = pid * HH
    HBH = HH // P
    NG = HH // W
    R = -(-NG // NT)
    S = 4 * T
    KC = 12  # row blocks a weight DMA

    out = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_tp_out" + tag)
    xs = nl.ndarray((P, KH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs[0:P, 0:KH, 0:T], src=xT.ap(pattern=[[T, P], [P * T, KH], [1, T]], offset=0))
    acc = nl.ndarray((P, R, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc[0:P, 0:R, 0:W], value=0.0)
    nc = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nc[0:1, 0:1], src=n_chunks.ap(pattern=[[1, 1], [1, 1]], offset=0))
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, nc)
    tp_routed(xs, affT, hits, n_reg, gate_up, gate_up_scale, down, down_scale, acc, limit=limit, tag=tag)
    # the diagonal of this program's hidden half
    k = 0
    for c in range(NT):
        rc = len(range(c, NG, NT))
        if rc == 0:
            continue
        for j in range(4):
            nisa.dma_copy(dst=out.ap(pattern=[[H, T], [NT * W, rc], [1, P]], offset=h0 + c * W + j * P),
                          src=acc[c * TILE + j * T:c * TILE + (j + 1) * T, 0:rc, j * P:(j + 1) * P],
                          dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[k % 2])
            k += 1
    return out


def tp_routed(xs, affT, hits, n_reg, gate_up, gate_up_scale, down, down_scale, acc, limit=10.0, tag=""):
    """The hit experts' slices over the tokens xs [128, KH, T] (normed, transposed), added into acc [128, R, 512]
    (the diagonal layout of this program's hidden half: tile c's rows 32 c + T j + t, round r, column block j); n_reg
    a register holding the chunks (ceil(hits / CH)). The device loop of moe_tp_decode."""
    KH = xs.shape[1]
    T = xs.shape[2]
    H = KH * P
    EL, _, _, C2 = gate_up.shape
    HB = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    SP = CH // n_prg
    HH = H // n_prg
    h0 = pid * HH
    HBH = HH // P
    NG = HH // W
    R = -(-NG // NT)
    S = 4 * T
    KC = 12
    hx = nl.ndarray((n_prg, P, SP * T), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="moe_tp_hx" + tag)

    # the slots' buffers persist over the loop, zeroed once: a skipped (out-of-range) slot keeps finite values
    # slot-major: each slot's run on a partition contiguous (12 KB of weights, 384 B of scales a DMA row)
    w = nl.ndarray((P, SP, KH, C2), dtype=gate_up.dtype, buffer=nl.sbuf)
    nisa.memset(dst=w[0:P, 0:SP, 0:KH, 0:C2], value=0.0, engine=nisa.gpsimd_engine)  # beside the vector's
    gs = nl.ndarray((P, SP, KH, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=gs[0:P, 0:SP, 0:KH, 0:2], value=0.0)
    wd = nl.ndarray((P, CH, HH), dtype=down.dtype, buffer=nl.sbuf)
    nisa.memset(dst=wd[0:P, 0:CH, 0:HH], value=0.0)
    ds = nl.ndarray((P, CH, HBH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ds[0:P, 0:CH, 0:HBH], value=0.0)
    base = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=base[0:1, 0:1], value=0)


    def tiled(dst, stationary, moving, c):
        nisa.nc_matmul(dst=dst, stationary=stationary, moving=moving, tile_position=(0, c * TILE),
                       tile_size=(P, TILE), accumulate=None)

    def chunk(it):
        # the chunk's experts: ix[s] = hits[base + s]
        ix = nl.ndarray((1, CH), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ix[0:1, 0:CH], src=hits.ap(pattern=[[CH, 1], [1, CH]], offset=0, scalar_offset=base,
                                                       indirect_dim=1))
        # this program's gate / up slots (pid * SP ..): weights, scales; then every slot's down weights and scales
        for s in range(SP):
            e = ix[0:1, pid * SP + s:pid * SP + s + 1]
            for k0 in range(0, KH, KC):
                nisa.dma_copy(dst=w[0:P, s, k0:k0 + KC, 0:C2],
                              src=gate_up.ap(pattern=[[KH * C2, P], [C2, KC], [1, C2]], offset=k0 * C2,
                                             scalar_offset=e, indirect_dim=0),
                              dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[(s + k0 // KC) % 2],
                              oob_mode=nisa.oob_mode.skip)
            nisa.dma_copy(dst=gs[0:P, s, 0:KH, 0:2],
                          src=gate_up_scale.ap(pattern=[[0, P], [2, KH], [1, 2]], offset=0, scalar_offset=e,
                                               indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[s % 2], oob_mode=nisa.oob_mode.skip)
        aw = nl.ndarray((P, CH, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=aw[0:P, 0:CH, 0:T], value=0.0)  # a skipped slot's weight stays 0
        for s in range(CH):
            e = ix[0:1, s:s + 1]
            nisa.dma_copy(dst=wd[0:P, s, 0:HH], src=down.ap(pattern=[[H, P], [1, HH]], offset=h0, scalar_offset=e,
                                                              indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[s % 2], oob_mode=nisa.oob_mode.skip)
            nisa.dma_copy(dst=ds[0:P, s, 0:HBH], src=down_scale.ap(pattern=[[0, P], [1, HBH]], offset=h0 // P,
                                                                     scalar_offset=e, indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[(s + 1) % 2], oob_mode=nisa.oob_mode.skip)
            nisa.dma_copy(dst=aw[0:P, s, 0:T], src=affT.ap(pattern=[[0, P], [1, T]], offset=0, scalar_offset=e,
                                                            indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[s % 2], oob_mode=nisa.oob_mode.skip)
        # ---- gate / up: slot s on column tile s, its stationary the tokens twice (gate scale, up scale) ----------
        xsc = nl.ndarray((P, SP, KH, 2 * T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for s in range(SP):
            nisa.tensor_tensor(dst=xsc.ap(pattern=[[SP * KH * 2 * T, P], [2 * T, KH], [T, 2], [1, T]],
                                          offset=s * KH * 2 * T),
                               data1=xs.ap(pattern=[[KH * T, P], [T, KH], [0, 2], [1, T]], offset=0),
                               data2=gs.ap(pattern=[[SP * KH * 2, P], [2, KH], [1, 2], [0, T]], offset=s * KH * 2),
                               op=nl.multiply)
        gu = nl.ndarray((P, C2), dtype=nl.float32, buffer=nl.psum)
        for kb in range(KH):
            for s in range(SP):
                tiled(gu[s * TILE:s * TILE + 2 * T, 0:C2], xsc[0:P, s, kb, 0:2 * T], w[0:P, s, kb, 0:C2], s)
        gsb = nl.ndarray((P, C2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gsb[0:P, 0:C2], src=gu[0:P, 0:C2])
        tq = nl.ndarray((P, 2, P), dtype=nl.float32, buffer=nl.psum)
        for j in range(2):
            nisa.nc_transpose(dst=tq[0:P, j, 0:P], data=gsb[0:P, j * P:(j + 1) * P])
        # slot s: gate = tq[:, 0, 32 s + t], up = tq[:, 1, 32 s + T + t]
        gt = nl.ndarray((P, SP, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gt[0:P, 0:SP, 0:T], data=tq.ap(pattern=[[2 * P, P], [TILE, SP], [1, T]], offset=0),
                           op0=nl.minimum, operand0=limit)
        nisa.activation(dst=gt[0:P, 0:SP, 0:T], op=nl.silu, data=gt[0:P, 0:SP, 0:T])
        up = nl.ndarray((P, SP, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up[0:P, 0:SP, 0:T], data=tq.ap(pattern=[[2 * P, P], [TILE, SP], [1, T]], offset=P + T),
                           op0=nl.minimum, operand0=limit, op1=nl.maximum, operand1=-limit)
        nisa.tensor_tensor(dst=gt[0:P, 0:SP, 0:T], data1=gt[0:P, 0:SP, 0:T], data2=up[0:P, 0:SP, 0:T], op=nl.multiply)
        hT = nl.ndarray((P, SP, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hT[0:P, 0:SP, 0:T], data1=gt[0:P, 0:SP, 0:T],
                           data2=aw.ap(pattern=[[CH * T, P], [T, SP], [1, T]], offset=pid * SP * T), op=nl.multiply)
        # ---- the activations meet: hA[p, slot, t] for all CH slots ----------------------------------------------------
        nisa.dma_copy(dst=hx.ap(pattern=[[SP * T, P], [1, SP * T]], offset=pid * P * SP * T), src=hT[0:P, 0:SP, 0:T])
        nisa.core_barrier(hx, (0, 1))
        hA = nl.ndarray((P, CH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=hA[0:P, 0:CH, 0:T], src=hx.ap(pattern=[[SP * T, P], [P * SP * T, n_prg], [1, SP * T]],
                                                         offset=0))
        nisa.core_barrier(hx, (0, 1))
        # ---- down: per slot, its activation in 4 copies scaled by 4 column blocks' scales ---------------------------
        hs = nl.ndarray((P, CH, HBH * T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hs.ap(pattern=[[CH * HBH * T, P], [HBH * T, CH], [T, HBH], [1, T]], offset=0),
                           data1=hA.ap(pattern=[[CH * T, P], [T, CH], [0, HBH], [1, T]], offset=0),
                           data2=ds.ap(pattern=[[CH * HBH, P], [HBH, CH], [1, HBH], [0, T]], offset=0), op=nl.multiply)
        pd = nl.ndarray((P, R, W), dtype=nl.float32, buffer=nl.psum)
        for s in range(CH):
            for grp in range(NG):
                r, c = grp // NT, grp % NT
                tiled(pd[c * TILE:c * TILE + S, r, 0:W], hs[0:P, s, grp * S:(grp + 1) * S],
                      wd[0:P, s, grp * W:(grp + 1) * W], c)
        for grp in range(NG):
            r, c = grp // NT, grp % NT
            nisa.tensor_tensor(dst=acc[c * TILE:c * TILE + S, r, 0:W], data1=acc[c * TILE:c * TILE + S, r, 0:W],
                               data2=pd[c * TILE:c * TILE + S, r, 0:W], op=nl.add)
        nisa.tensor_scalar(dst=base[0:1, 0:1], data=base[0:1, 0:1], op0=nl.add, operand0=CH)

    nl.fori_loop(0, n_reg, chunk)


def tp_layouts(gate_up, gate_up_scale, down, down_scale, j: int, tp: int = 16):
    """The EP group's experts (gate_up [EL, H, 2I] / gate_up_scale [EL, H/128, 2I/128] / down [EL, I, H] /
    down_scale [EL, I/128, H/128], the trn2-halved layouts of weights.py) as rank ``j`` of the group's slices, in
    moe_tp_decode's layouts."""
    import torch

    EL, H, I2 = gate_up.shape
    I = I2 // 2  # noqa: E741
    sw = I // tp
    KH = H // P
    c0 = j * sw
    gu = torch.cat([gate_up[:, :, c0:c0 + sw], gate_up[:, :, I + c0:I + c0 + sw]], dim=2)  # [EL, H, 2 sw]
    gu = gu.view(torch.int8).view(EL, KH, P, 2 * sw).permute(0, 2, 1, 3).contiguous().view(gate_up.dtype)
    gus = torch.stack([gate_up_scale[:, :, c0 // P], gate_up_scale[:, :, (I + c0) // P]], dim=2).contiguous()
    dn = down[:, c0:c0 + sw, :].contiguous()
    dns = down_scale[:, c0 // P, :].contiguous()
    return gu, gus, dn, dns
