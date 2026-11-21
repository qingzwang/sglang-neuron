# SPDX-License-Identifier: Apache-2.0
"""Hy4's decode routed experts in one NKI kernel, token-major: the hit local experts only, each weight block one
cheap matmul.

moe_decode.py makes each 128 x 128 FP8 weight block the stationary operand against the few decode tokens and scales
and accumulates its product on the vector engine: per expert and program 1152 blocks, each a weight load, a matmul
and a vector instruction of almost no work (~0.2 ms an expert, instruction bound). Here the block scales go into the
activations instead, which are tiny: for each 128-row block of a weight, one vector instruction scales the tokens'
activation rows by all the block's column-block scales at once (zero-stride operands broadcast them), and each weight
block is then the moving operand of one matmul against its scaled activations (stationary, T columns: a cheap weight
load), accumulating token-major in PSUM over the row blocks with no per-block vector work. The accumulation flags are
left to the compiler (accumulate=None): several 128-column accumulations share a PSUM bank, and an explicit
overwrite (accumulate=False) of one of them broke the others' sums.

Per rank and step, for T <= 128 tokens, over the hit local experts (``hit`` lists them first, ``n_hit`` counts them;
a device loop walks them):

    out[p] = sum over the hit experts e of  (silu(min(x Wg_e, L)) * clamp(x Wu_e, -L, L) * aff[:, e]) Wd_e

The two programs of a logical core take one half of I each; ``out`` [2, T, H] fp32 holds each program's partial,
which the caller sums."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def moe_decode_fp8_tm(xT, affT, hit, n_hit, gate_up, gate_up_scale, down, down_scale, limit=10.0):
    """xT [H, T] bf16 (the tokens' activations, transposed); affT [E, T] fp32 (zero off a token's experts); hit [1, E]
    int32; n_hit [1, 1] int32 -> [2, T, H] fp32 partials (token-major)."""
    H, T = xT.shape
    E, I, _ = down.shape  # noqa: E741
    KH = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    IH = I // n_prg
    i0 = pid * IH
    OB = IH // P  # this program's column blocks of the gate, and of the up
    KI = IH // P  # its row blocks of the down
    HB = H // P  # column blocks of the down
    GB = 2 * I // P

    out = nl.ndarray((n_prg, T, H), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_decode_tm_out")
    acc = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc[0:T, 0:H], value=0.0)
    xs = nl.ndarray((P, KH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs[0:P, 0:KH, 0:T], src=xT.ap(pattern=[[T, P], [P * T, KH], [1, T]], offset=0))

    nh = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nh[0:1, 0:1], src=n_hit.ap(pattern=[[1, 1], [1, 1]], offset=0))
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, nh)
    slot = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=slot[0:1, 0:1], value=0)

    def expert_body(it):
        e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e[0:1, 0:1], src=hit.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=slot,
                                                    indirect_dim=1))
        # each token's weight on this expert, one token a partition
        a = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a[0:T, 0:1], src=affT.ap(pattern=[[1, T], [1, 1]], offset=0, scalar_offset=e,
                                                     indirect_dim=0))
        # the gate / up block scales of this program's columns on every partition: [kb, g, ob]
        gs = nl.ndarray((P, KH, 2, OB), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(2):
            nisa.dma_copy(dst=gs[0:P, 0:KH, g, 0:OB],
                          src=gate_up_scale.ap(pattern=[[0, P], [GB, KH], [1, OB]], offset=(g * I + i0) // P,
                                               scalar_offset=e, indirect_dim=0))
        # gate (g = 0) and up (g = 1), token-major [T, IH] in PSUM
        gu = nl.ndarray((T, 2, IH), dtype=nl.float32, buffer=nl.psum)
        for kb in range(KH):
            wt = nl.ndarray((P, 2, IH), dtype=gate_up.dtype, buffer=nl.sbuf)
            for g in range(2):
                nisa.dma_copy(dst=wt[0:P, g, 0:IH],
                              src=gate_up.ap(pattern=[[2 * I, P], [1, IH]], offset=kb * P * 2 * I + g * I + i0,
                                             scalar_offset=e, indirect_dim=0))
            # the tokens' row block kb times each of its column blocks' scales: xsc [feature, g, ob, token]
            xsc = nl.ndarray((P, 2, OB, T), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=xsc[0:P, 0:2, 0:OB, 0:T],
                               data1=xs.ap(pattern=[[KH * T, P], [0, 2], [0, OB], [1, T]], offset=kb * T),
                               data2=gs.ap(pattern=[[KH * 2 * OB, P], [OB, 2], [1, OB], [0, T]], offset=kb * 2 * OB),
                               op=nl.multiply)
            for g in range(2):
                for ob in range(OB):
                    nisa.nc_matmul(dst=gu[0:T, g, ob * P:(ob + 1) * P], stationary=xsc[0:P, g, ob, 0:T],
                                   moving=wt[0:P, g, ob * P:(ob + 1) * P], accumulate=None)
        # silu(min(gate, L)) * clamp(up, -L, L) * affinity -> h [T, IH] bf16, transposed hT [i in block, ki, T]
        gt = nl.ndarray((T, IH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gt[0:T, 0:IH], data=gu[0:T, 0, 0:IH], op0=nl.minimum, operand0=limit)
        nisa.activation(dst=gt[0:T, 0:IH], op=nl.silu, data=gt[0:T, 0:IH])
        up = nl.ndarray((T, IH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up[0:T, 0:IH], data=gu[0:T, 1, 0:IH], op0=nl.minimum, operand0=limit, op1=nl.maximum,
                           operand1=-limit)
        nisa.tensor_tensor(dst=gt[0:T, 0:IH], data1=gt[0:T, 0:IH], data2=up[0:T, 0:IH], op=nl.multiply)
        h = nl.ndarray((T, IH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h[0:T, 0:IH], data=gt[0:T, 0:IH], op0=nl.multiply, operand0=a[0:T, 0:1])
        hT = nl.ndarray((P, KI, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ki in range(KI):
            tp = nl.ndarray((P, T), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:T], data=h[0:T, ki * P:(ki + 1) * P])
            nisa.tensor_copy(dst=hT[0:P, ki, 0:T], src=tp[0:P, 0:T])
        # the down: this program's rows, scaled activations hs [i in block, ki, hb, T]
        ds = nl.ndarray((P, KI, HB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds[0:P, 0:KI, 0:HB],
                      src=down_scale.ap(pattern=[[0, P], [HB, KI], [1, HB]], offset=(i0 // P) * HB,
                                        scalar_offset=e, indirect_dim=0))
        wd = nl.ndarray((P, KI, H), dtype=down.dtype, buffer=nl.sbuf)
        for ki in range(KI):
            nisa.dma_copy(dst=wd[0:P, ki, 0:H], src=down.ap(pattern=[[H, P], [1, H]], offset=(i0 + ki * P) * H,
                                                              scalar_offset=e, indirect_dim=0))
        hs = nl.ndarray((P, KI, HB, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ki in range(KI):
            nisa.tensor_tensor(dst=hs[0:P, ki, 0:HB, 0:T],
                               data1=hT.ap(pattern=[[KI * T, P], [0, HB], [1, T]], offset=ki * T),
                               data2=ds.ap(pattern=[[KI * HB, P], [1, HB], [0, T]], offset=ki * HB),
                               op=nl.multiply)
        for hb in range(HB):
            pd = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.psum)
            for ki in range(KI):
                nisa.nc_matmul(dst=pd[0:T, 0:P], stationary=hs[0:P, ki, hb, 0:T], moving=wd[0:P, ki, hb * P:(hb + 1) * P],
                               accumulate=None)
            nisa.tensor_tensor(dst=acc[0:T, hb * P:(hb + 1) * P], data1=acc[0:T, hb * P:(hb + 1) * P],
                               data2=pd[0:T, 0:P], op=nl.add)
        nisa.tensor_scalar(dst=slot[0:1, 0:1], data=slot[0:1, 0:1], op0=nl.add, operand0=1)

    nl.fori_loop(0, n_reg, expert_body)
    nisa.dma_copy(dst=out.ap(pattern=[[H, T], [1, H]], offset=pid * T * H), src=acc[0:T, 0:H])
    return out


TILE = 32  # a tensor-engine column tile: up to T <= 32 tokens
WEIGHT_DMA_ENGINES = (nisa.engine.sync, nisa.engine.scalar)  # the weight DMAs' trigger engines, in turn


def moe_decode_fp8_ct(xT, affT, hit, n_hit, gate_up, gate_up_scale, down, down_scale, limit=10.0):
    """moe_decode_fp8_tm on the tensor engine's column tiles: the stationary operand is only T <= 32 columns wide, so
    four matmuls (of four column blocks) run at once, one in each 32-column tile of the array (~3.3x the matmul rate
    of the full array at this width, measured). Tile c's products land in PSUM partitions 32 c .. 32 c + T, so the
    gate / up / down results are laid out [32 c + t, q, column] for column block 4 q + c; the activation runs in that
    layout, and its transpose gives the down projection's row block 4 q + c in columns 32 c .. 32 c + T. Same
    arguments and result as moe_decode_fp8_tm ([2, T, H] fp32 partials).

    At this rate the kernel is bound by issuing the weight DMAs: their descriptors come from the hardware DGE of the
    sync and scalar engines in turn (the GpSimd's software DGE took ~0.7 us a DMA)."""
    H, T = xT.shape
    E, I, _ = down.shape  # noqa: E741
    KH = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    IH = I // n_prg
    i0 = pid * IH
    OB = IH // P
    KI = IH // P
    HB = H // P
    GB = 2 * I // P
    NT = P // TILE  # column tiles
    OQ = OB // NT  # gate / up column-block quads
    HQ = HB // NT  # down column-block quads

    out = nl.ndarray((n_prg, T, H), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_decode_ct_out")
    acc = nl.ndarray((P, HQ, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc[0:P, 0:HQ, 0:P], value=0.0)
    xs = nl.ndarray((P, KH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs[0:P, 0:KH, 0:T], src=xT.ap(pattern=[[T, P], [P * T, KH], [1, T]], offset=0))

    nh = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nh[0:1, 0:1], src=n_hit.ap(pattern=[[1, 1], [1, 1]], offset=0))
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, nh)
    slot = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=slot[0:1, 0:1], value=0)

    def tiled(dst, stationary, moving, c):
        nisa.nc_matmul(dst=dst, stationary=stationary, moving=moving, tile_position=(0, c * TILE),
                       tile_size=(P, TILE), accumulate=None)

    def expert_body(it):
        e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e[0:1, 0:1], src=hit.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=slot,
                                                    indirect_dim=1))
        # each token's weight on this expert, at partition 32 c + t of every tile c
        a = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=a[0:P, 0:1], value=0.0)
        for c in range(NT):
            nisa.dma_copy(dst=a[c * TILE:c * TILE + T, 0:1], src=affT.ap(pattern=[[1, T], [1, 1]], offset=0,
                                                                           scalar_offset=e, indirect_dim=0))
        gs = nl.ndarray((P, KH, 2, OB), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(2):
            nisa.dma_copy(dst=gs[0:P, 0:KH, g, 0:OB],
                          src=gate_up_scale.ap(pattern=[[0, P], [GB, KH], [1, OB]], offset=(g * I + i0) // P,
                                               scalar_offset=e, indirect_dim=0))
        # gate / up: [32 c + t, g, q, column] for column block 4 q + c
        gu = nl.ndarray((P, 2, OQ, P), dtype=nl.float32, buffer=nl.psum)
        for kb in range(KH):
            wt = nl.ndarray((P, 2, IH), dtype=gate_up.dtype, buffer=nl.sbuf)
            for g in range(2):
                nisa.dma_copy(dst=wt[0:P, g, 0:IH],
                              src=gate_up.ap(pattern=[[2 * I, P], [1, IH]], offset=kb * P * 2 * I + g * I + i0,
                                             scalar_offset=e, indirect_dim=0),
                              dge_mode=nisa.dge_mode.hwdge, engine=WEIGHT_DMA_ENGINES[(2 * kb + g) % 2])
            xsc = nl.ndarray((P, 2, OB, T), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=xsc[0:P, 0:2, 0:OB, 0:T],
                               data1=xs.ap(pattern=[[KH * T, P], [0, 2], [0, OB], [1, T]], offset=kb * T),
                               data2=gs.ap(pattern=[[KH * 2 * OB, P], [OB, 2], [1, OB], [0, T]], offset=kb * 2 * OB),
                               op=nl.multiply)
            for g in range(2):
                for q in range(OQ):
                    for c in range(NT):
                        ob = q * NT + c
                        tiled(gu[c * TILE:c * TILE + T, g, q, 0:P], xsc[0:P, g, ob, 0:T],
                              wt[0:P, g, ob * P:(ob + 1) * P], c)
        # silu(min(gate, L)) * clamp(up, -L, L) * affinity, in the tiled layout (rows past a tile's T are unused)
        gt = nl.ndarray((P, OQ, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gt[0:P, 0:OQ, 0:P], data=gu[0:P, 0, 0:OQ, 0:P], op0=nl.minimum, operand0=limit)
        nisa.activation(dst=gt[0:P, 0:OQ, 0:P], op=nl.silu, data=gt[0:P, 0:OQ, 0:P])
        up = nl.ndarray((P, OQ, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up[0:P, 0:OQ, 0:P], data=gu[0:P, 1, 0:OQ, 0:P], op0=nl.minimum, operand0=limit,
                           op1=nl.maximum, operand1=-limit)
        nisa.tensor_tensor(dst=gt[0:P, 0:OQ, 0:P], data1=gt[0:P, 0:OQ, 0:P], data2=up[0:P, 0:OQ, 0:P], op=nl.multiply)
        h = nl.ndarray((P, OQ, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h[0:P, 0:OQ, 0:P], data=gt[0:P, 0:OQ, 0:P], op0=nl.multiply, operand0=a[0:P, 0:1])
        # hT[:, q, 32 c + t] is row block 4 q + c of the down's contraction
        hT = nl.ndarray((P, OQ, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for q in range(OQ):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=h[0:P, q, 0:P])
            nisa.tensor_copy(dst=hT[0:P, q, 0:P], src=tp[0:P, 0:P])
        ds = nl.ndarray((P, KI, HB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds[0:P, 0:KI, 0:HB],
                      src=down_scale.ap(pattern=[[0, P], [HB, KI], [1, HB]], offset=(i0 // P) * HB,
                                        scalar_offset=e, indirect_dim=0))
        wd = nl.ndarray((P, KI, H), dtype=down.dtype, buffer=nl.sbuf)
        for ki in range(KI):
            nisa.dma_copy(dst=wd[0:P, ki, 0:H], src=down.ap(pattern=[[H, P], [1, H]], offset=(i0 + ki * P) * H,
                                                              scalar_offset=e, indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=WEIGHT_DMA_ENGINES[ki % 2])
        hs = nl.ndarray((P, KI, HB, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ki in range(KI):
            q, c = ki // NT, ki % NT
            nisa.tensor_tensor(dst=hs[0:P, ki, 0:HB, 0:T],
                               data1=hT.ap(pattern=[[OQ * P, P], [0, HB], [1, T]], offset=q * P + c * TILE),
                               data2=ds.ap(pattern=[[KI * HB, P], [1, HB], [0, T]], offset=ki * HB),
                               op=nl.multiply)
        # down: [32 c + t, q, column] for column block 4 q + c of H
        pd = nl.ndarray((P, HQ, P), dtype=nl.float32, buffer=nl.psum)
        for ki in range(KI):
            for q in range(HQ):
                for c in range(NT):
                    hb = q * NT + c
                    tiled(pd[c * TILE:c * TILE + T, q, 0:P], hs[0:P, ki, hb, 0:T], wd[0:P, ki, hb * P:(hb + 1) * P], c)
        nisa.tensor_tensor(dst=acc[0:P, 0:HQ, 0:P], data1=acc[0:P, 0:HQ, 0:P], data2=pd[0:P, 0:HQ, 0:P], op=nl.add)
        nisa.tensor_scalar(dst=slot[0:1, 0:1], data=slot[0:1, 0:1], op0=nl.add, operand0=1)

    nl.fori_loop(0, n_reg, expert_body)
    # acc[32 c + t, q, column] -> out[pid, t, (4 q + c) * 128 + column]
    for c in range(NT):
        nisa.dma_copy(dst=out.ap(pattern=[[H, T], [NT * P, HQ], [1, P]], offset=pid * T * H + c * P),
                      src=acc[c * TILE:c * TILE + T, 0:HQ, 0:P])
    return out
