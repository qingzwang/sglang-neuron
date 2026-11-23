# SPDX-License-Identifier: Apache-2.0
"""Hy4's decode routed experts with 512-column moving tiles: the tensor engine at the weights' DMA rate, for T <= 8
tokens.

moe_decode2.moe_decode_fp8_ct folds each 128 x 128 block's FP8 scale into the tokens (the stationary operand), so
each matmul moves one 128-column block: ~270 GB/s of weights a program on the tensor engine (kbench/gemv_rate.py),
below the ~395 GB/s a program's DMA streams, and an expert took ~93 us for the ~48 us its 37.7 MB of weights take to
load. A matmul moving 512 columns runs at ~720 GB/s, but its columns span 4 column blocks of different scales. So each
32-column tile's stationary is the tokens 4 times over, copy j scaled by column block j's scale (4 T <= 32 columns):
row block j of the product is the tokens against all 512 columns at block j's scale, and its j-th 128 columns (the
"diagonal") are the wanted ones. The products accumulate in PSUM over the row blocks with no vector work; the
tensor engine computes 4x the products it keeps, which at T <= 8 tokens it has to spare.

* gate / up: tiles 0, 1 the program's gate columns [512 c, 512 c + 512), tiles 2, 3 its up columns likewise (each
  weight row a contiguous 1 KB a DMA descriptor). The [128, 512] result transposed by 128-column block q puts block
  q's columns on the partitions, copy q's tokens at columns 32 c + T q + t: the gate and the up of a column sit on
  one partition, and the activation runs there, giving the down's stationary (the contraction rows) directly.
* down: contraction row block 4 cg + q (the activation's block); the 48 column blocks of H as 12 groups of 4,
  three rounds over the tiles (tile c of round r: group 4 r + c), each group's stationary the activation scaled by its
  4 blocks' scales; per expert the [128, 3, 512] diagonal-bearing product adds into the accumulator, whose diagonal
  goes out at the end.

Same arguments and result as moe_decode_fp8_ct ([2, T, H] fp32 partials, which the caller sums)."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
TILE = 32
NT = P // TILE
W = 512  # moving columns a matmul
GW = W // P  # column blocks a matmul (4)
MAX_T = TILE // GW  # tokens: 4 scaled copies in a 32-column tile
KB_DMA = 4
DMA_ENGINES = (nisa.engine.sync, nisa.engine.scalar)


def moe_decode_fp8_wide(xT, affT, hit, n_hit, gate_up, gate_up_scale, down, down_scale, limit=10.0):
    """xT [H, T] bf16 (T <= 8); affT [E, T] fp32; hit [1, E] int32 (the hit local experts first); n_hit [1, 1] int32;
    gate_up [E, H, 2I] / gate_up_scale [E, H/128, 2I/128] / down [E, I, H] / down_scale [E, I/128, H/128] (trn2-halved
    e4m3) -> [2, T, H] fp32."""
    H, T = xT.shape
    E, I, _ = down.shape  # noqa: E741
    KH = H // P
    PM = len(gate_up.shape) == 3 and gate_up.shape[1] != H  # partition-major [E, n_prg * P * KH * 2, IH]
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    IH = I // n_prg
    i0 = pid * IH
    GB = 2 * I // P
    HB = H // P
    KI = IH // P  # the down's row blocks of this program (2 a tile)
    CW = IH // NT  # a tile's intermediate columns (256)
    CB = CW // P  # their column blocks (2)
    NG = H // W  # the down's 512-column groups (12)
    R = NG // NT  # their rounds over the tiles (3)
    S = GW * T  # stationary columns a tile

    out = nl.ndarray((n_prg, T, H), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_wide_out")
    acc = nl.ndarray((P, R, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc[0:P, 0:R, 0:W], value=0.0)
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
        # this expert's weight of each token, on every partition: aw[p, t]
        aw = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=aw[0:P, 0:T], src=affT.ap(pattern=[[0, P], [1, T]], offset=0, scalar_offset=e,
                                                       indirect_dim=0))
        # gs[p, kb, c, j]: the scale of row block kb, tile c's column block j: tiles 0, 1 the gate's blocks 4 c + j,
        # tiles 2, 3 the up's blocks 4 (c - 2) + j
        gs = nl.ndarray((P, KH, NT, GW), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(2):
            nisa.dma_copy(dst=gs.ap(pattern=[[KH * NT * GW, P], [NT * GW, KH], [GW, 2], [1, GW]], offset=g * 2 * GW),
                          src=gate_up_scale.ap(pattern=[[0, P], [GB, KH], [GW, 2], [1, GW]],
                                               offset=(g * I + i0) // P, scalar_offset=e, indirect_dim=0))
        # ds[p, ki, b]: the down's scales, on every partition
        ds = nl.ndarray((P, KI, HB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds[0:P, 0:KI, 0:HB], src=down_scale.ap(pattern=[[0, P], [HB, KI], [1, HB]],
                                                                   offset=(i0 // P) * HB, scalar_offset=e,
                                                                   indirect_dim=0))
        # ---- gate / up ----------------------------------------------------------------------------------------------
        gu = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.psum)
        for kb in range(KH):
            if kb % KB_DMA == 0:
                wts = nl.ndarray((P, KB_DMA, 2, IH), dtype=gate_up.dtype, buffer=nl.sbuf)
                if PM:
                    # partition-major (partition_major_gate_up): a partition's KB_DMA row blocks of both halves are
                    # one contiguous 8 KB run, whole DMA packets
                    nisa.dma_copy(dst=wts[0:P, 0:KB_DMA, 0:2, 0:IH],
                                  src=gate_up.ap(pattern=[[KH * 2 * IH, P], [2 * IH, KB_DMA], [IH, 2], [1, IH]],
                                                 offset=(pid * P * KH + kb) * 2 * IH, scalar_offset=e,
                                                 indirect_dim=0),
                                  dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[(kb // KB_DMA) % 2])
                else:
                    for g in range(2):
                        nisa.dma_copy(dst=wts[0:P, 0:KB_DMA, g, 0:IH],
                                      src=gate_up.ap(pattern=[[2 * I, P], [P * 2 * I, KB_DMA], [1, IH]],
                                                     offset=kb * P * 2 * I + g * I + i0, scalar_offset=e,
                                                     indirect_dim=0),
                                      dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[(kb // KB_DMA + g) % 2])
            # the stationaries of the four tiles: xsc[p, c, j, t] = x[kb rows p, t] * scale(kb, c, j)
            xsc = nl.ndarray((P, NT, S), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=xsc.ap(pattern=[[NT * S, P], [S, NT], [T, GW], [1, T]], offset=0),
                               data1=xs.ap(pattern=[[KH * T, P], [0, NT], [0, GW], [1, T]], offset=kb * T),
                               data2=gs.ap(pattern=[[KH * NT * GW, P], [GW, NT], [1, GW], [0, T]],
                                           offset=kb * NT * GW), op=nl.multiply)
            for c in range(NT):
                tiled(gu[c * TILE:c * TILE + S, 0:W], xsc[0:P, c, 0:S], wts[0:P, kb % KB_DMA, c // 2, (c % 2) * W:(c % 2 + 1) * W], c)
        gsb = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gsb[0:P, 0:W], src=gu[0:P, 0:W])
        # block q transposed: tq[col, 32 c + T j + t]; the diagonal is j = q
        tq = nl.ndarray((P, GW, P), dtype=nl.float32, buffer=nl.psum)
        for q in range(GW):
            nisa.nc_transpose(dst=tq[0:P, q, 0:P], data=gsb[0:P, q * P:(q + 1) * P])
        # the gate's block 4 cg + q is tq[:, q, 32 cg + T q + t], its up's tq[:, q, 32 (2 + cg) + T q + t]: one partition
        hT = nl.ndarray((P, KI, T), dtype=nl.bfloat16, buffer=nl.sbuf)  # [col, row block 4 cg + q, t]
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
        # ---- down ---------------------------------------------------------------------------------------------------
        wd = nl.ndarray((P, KI, H), dtype=down.dtype, buffer=nl.sbuf)
        for ki in range(KI):
            nisa.dma_copy(dst=wd[0:P, ki, 0:H], src=down.ap(pattern=[[H, P], [1, H]], offset=(i0 + ki * P) * H,
                                                              scalar_offset=e, indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=DMA_ENGINES[ki % 2])
        # hs[p, ki, b, t] = h[row block ki, t] * down scale (ki, b): group g's stationary is blocks 4 g .. 4 g + 3
        hs = nl.ndarray((P, KI, HB * T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hs.ap(pattern=[[KI * HB * T, P], [HB * T, KI], [T, HB], [1, T]], offset=0),
                           data1=hT.ap(pattern=[[KI * T, P], [T, KI], [0, HB], [1, T]], offset=0),
                           data2=ds.ap(pattern=[[KI * HB, P], [HB, KI], [1, HB], [0, T]], offset=0), op=nl.multiply)
        pd = nl.ndarray((P, R, W), dtype=nl.float32, buffer=nl.psum)
        for ki in range(KI):
            for r in range(R):
                for c in range(NT):
                    grp = r * NT + c
                    tiled(pd[c * TILE:c * TILE + S, r, 0:W], hs[0:P, ki, grp * S:(grp + 1) * S],
                          wd[0:P, ki, grp * W:(grp + 1) * W], c)
        nisa.tensor_tensor(dst=acc[0:P, 0:R, 0:W], data1=acc[0:P, 0:R, 0:W], data2=pd[0:P, 0:R, 0:W], op=nl.add)
        nisa.tensor_scalar(dst=slot[0:1, 0:1], data=slot[0:1, 0:1], op0=nl.add, operand0=1)

    nl.fori_loop(0, n_reg, expert_body)
    # the diagonal: acc[32 c + T j + t, r, 128 j + col] -> out[pid, t, (4 r + c) * 512 + 128 j + col]
    for c in range(NT):
        for j in range(GW):
            nisa.dma_copy(dst=out.ap(pattern=[[H, T], [NT * W, R], [1, P]], offset=pid * T * H + c * W + j * P),
                          src=acc[c * TILE + j * T:c * TILE + (j + 1) * T, 0:R, j * P:(j + 1) * P])
    return out


def partition_major_gate_up(gate_up, n_prg: int = 2):
    """The experts' gate_up [E, H, 2I] as moe_decode_fp8_wide's partition-major layout [E, n_prg * 128 * KH * 2,
    I / n_prg]: element [e, ((p_rg * 128 + p) * KH + kb) * 2 + g, j] = gate_up[e, 128 kb + p, g I + p_rg I / n_prg + j],
    so each program's partition reads its row blocks of the gate and the up as one contiguous run."""
    E, H, I2 = gate_up.shape
    I = I2 // 2  # noqa: E741
    KH = H // P
    v = gate_up.view(E, KH, P, 2, n_prg, I // n_prg)  # [e, kb, p, g, prg, j]
    return v.permute(0, 4, 2, 1, 3, 5).contiguous().view(E, n_prg * P * KH * 2, I // n_prg)
