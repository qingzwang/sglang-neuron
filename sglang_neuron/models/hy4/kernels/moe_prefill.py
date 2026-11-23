# SPDX-License-Identifier: Apache-2.0
"""Hy4's prefill routed experts in one NKI kernel: each local expert over the tokens routed to it only, in blocks of
128 tokens, the block-FP8 weights dequantized in SBUF (never written back to HBM).

In the XLA graph every local expert ran over every token of the chunk (~32 of 1024 tokens are routed to an expert on
average) and its weights were dequantized to bf16 in HBM and read back by the matmul: most of a layer's tensor-engine
time and ~1.5 GB of its HBM traffic.

The caller lists the blocks (``moe_blocks``): block b holds up to 128 tokens routed to expert ``bexp[b]``; ``tok``
gives each slot's token (T for an empty slot: the gather and the scatter skip it, out of bounds) and ``aff`` its
routing weight; ``nblk`` counts the blocks. The kernel walks them in a device loop. For each block:

1. its tokens' rows are gathered from HBM (one indirect DMA) and transposed on the tensor engine (feature-major);
2. gate / up: for each 128-feature row block of the expert's weights, the FP8 rows are scaled to bf16 by their block
   scales (one instruction a row block: the scales broadcast over their columns by a zero-stride operand) and are the
   moving operand against the tokens (stationary): the products accumulate token-major in PSUM;
3. silu(min(gate, L)) * clamp(up, -L, L) times the token's weight, transposed for the down projection;
4. down: the same, accumulated token-major in PSUM, and scatter-added into the output rows (DMA read-modify-write).

The expert weights are loaded by the sync engine's hardware DGE (the GpSimd's software DGE took ~0.7 us a DMA and
bound the kernel). The two programs of a logical core take one half of the intermediate dimension each, and each adds into its own half
of ``out`` ([2, T, H] fp32), which the caller sums."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
COLS = 512  # a PSUM bank of fp32 columns


def moe_prefill_fp8(x, tok, tok_out, aff, bexp, nblk, gate_up, gate_up_scale, down, down_scale, limit=10.0):
    """x [T, H] bf16; tok [NB, 128] int32 (T marks an empty slot); tok_out [NB, 2, 128] int32 (the slot's row of
    ``out`` viewed [2 T, H] for each program: token + program * T, 2 T for an empty slot); aff [NB, 128] fp32;
    bexp [NB, 1] int32; nblk [1, 1] int32; gate_up [E, H, 2I] (trn2-halved e4m3, gate columns then up);
    gate_up_scale [E, H / 128, 2I / 128]; down [E, I, H]; down_scale [E, I / 128, H / 128] -> out [2 T, H] fp32."""
    T, H = x.shape
    NB = tok.shape[0]
    E, I, _ = down.shape  # noqa: E741
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    IH = I // n_prg
    i0 = pid * IH
    KH = H // P  # feature blocks of the gate / up contraction
    KI = IH // P  # this program's intermediate blocks (the down contraction)
    GB = 2 * I // P  # column blocks of gate_up
    OB = IH // P  # this program's column blocks of the gate and of the up
    HC = H // COLS  # down output column chunks
    HALF = HC // 2
    out = nl.ndarray((2 * T, H), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_prefill_out")

    # zero this program's half of the output (the blocks scatter-add into it)
    zero = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero[0:P, 0:H], value=0.0)
    for tb in range(T // P):
        nisa.dma_copy(dst=out.ap(pattern=[[H, P], [1, H]], offset=(pid * T + tb * P) * H), src=zero[0:P, 0:H])

    nb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nb[0:1, 0:1], src=nblk.ap(pattern=[[1, 1], [1, 1]], offset=0))
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, nb)
    slot = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=slot[0:1, 0:1], value=0)

    def dequant(dst, src, sc, part_stride, offset, nblocks, j):
        """dst = src * the scale of each 128-column block (``sc``: per partition ``part_stride`` apart, block k at
        ``offset + k``). The vector engine (~1.3 ns an element a partition) takes three of four calls in one
        instruction, the scales broadcast over their columns by a zero-stride operand; the scalar engine (~3.7 ns)
        the fourth, a block at a time (its scale operand is per partition). The GpSimd engine (~6-8 ns) takes none."""
        if j % 4 == 3:
            for k in range(nblocks):
                nisa.activation(dst=dst[0:P, k * P:(k + 1) * P], op=nl.copy, data=src[0:P, k * P:(k + 1) * P],
                                scale=sc.ap(pattern=[[part_stride, P], [1, 1]], offset=offset + k))
        else:
            nisa.tensor_tensor(dst=dst, data1=src, data2=sc.ap(pattern=[[part_stride, P], [1, nblocks], [0, P]],
                                                                 offset=offset), op=nl.multiply,
                               engine=nisa.vector_engine)

    def block(it):
        # the block's expert, tokens and weights
        e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e[0:1, 0:1], src=bexp.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=slot,
                                                     indirect_dim=0))
        ti = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti[0:P, 0:1], src=tok.ap(pattern=[[1, P], [1, 1]], offset=0, scalar_offset=slot,
                                                     indirect_dim=0))
        to = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=to[0:P, 0:1], src=tok_out.ap(pattern=[[1, P], [1, 1]], offset=pid * P, scalar_offset=slot,
                                                         indirect_dim=0))
        af = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=af[0:P, 0:1], src=aff.ap(pattern=[[1, P], [1, 1]], offset=0, scalar_offset=slot,
                                                     indirect_dim=0))
        # 1. gather the tokens' rows and transpose them: xT [feature in block, kb, token]
        xr = nl.ndarray((P, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xr[0:P, 0:H], src=x.ap(pattern=[[H, P], [1, H]], offset=0, vector_offset=ti[0:P, 0:1],
                                                    indirect_dim=0), dge_mode=nisa.dge_mode.swdge,
                      oob_mode=nisa.oob_mode.skip)
        xT = nl.ndarray((P, KH, P), dtype=x.dtype, buffer=nl.sbuf)
        for kb in range(KH):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=xr[0:P, kb * P:(kb + 1) * P])
            nisa.tensor_copy(dst=xT[0:P, kb, 0:P], src=tp[0:P, 0:P],
                             engine=nisa.scalar_engine if kb % 2 else nisa.vector_engine)
        # the gate / up block scales of this program's columns, on every partition: [kb, g, ob]
        gs = nl.ndarray((P, KH, 2, OB), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(2):
            nisa.dma_copy(dst=gs[0:P, 0:KH, g, 0:OB],
                          src=gate_up_scale.ap(pattern=[[0, P], [GB, KH], [1, OB]], offset=(g * I + i0) // P,
                                               scalar_offset=e, indirect_dim=0))
        # 2. gate / up, token-major [token, g, col] in PSUM (2 * IH / 512 banks)
        gu = nl.ndarray((P, 2, IH), dtype=nl.float32, buffer=nl.psum)
        for kb in range(KH):
            w8 = nl.ndarray((P, 2, IH), dtype=gate_up.dtype, buffer=nl.sbuf)
            for g in range(2):
                nisa.dma_copy(dst=w8[0:P, g, 0:IH],
                              src=gate_up.ap(pattern=[[2 * I, P], [1, IH]], offset=kb * P * 2 * I + g * I + i0,
                                             scalar_offset=e, indirect_dim=0), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            wb = nl.ndarray((P, 2, IH), dtype=nl.bfloat16, buffer=nl.sbuf)
            dequant(wb.reshape((P, 2 * IH)), w8.reshape((P, 2 * IH)), gs, KH * 2 * OB, kb * 2 * OB, 2 * OB, kb)
            for g in range(2):
                for c in range(IH // COLS):
                    nisa.nc_matmul(dst=gu[0:P, g, c * COLS:(c + 1) * COLS], stationary=xT[0:P, kb, 0:P],
                                   moving=wb[0:P, g, c * COLS:(c + 1) * COLS], accumulate=(kb > 0))
        # 3. silu(min(gate, L)) * clamp(up, -L, L) * affinity -> h [token, IH] bf16, transposed hT [i in block, ki, token]
        gt = nl.ndarray((P, IH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gt[0:P, 0:IH], data=gu[0:P, 0, 0:IH], op0=nl.minimum, operand0=limit)
        nisa.activation(dst=gt[0:P, 0:IH], op=nl.silu, data=gt[0:P, 0:IH])
        up = nl.ndarray((P, IH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up[0:P, 0:IH], data=gu[0:P, 1, 0:IH], op0=nl.minimum, operand0=limit, op1=nl.maximum,
                           operand1=-limit)
        nisa.tensor_tensor(dst=gt[0:P, 0:IH], data1=gt[0:P, 0:IH], data2=up[0:P, 0:IH], op=nl.multiply)
        h = nl.ndarray((P, IH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h[0:P, 0:IH], data=gt[0:P, 0:IH], op0=nl.multiply, operand0=af[0:P, 0:1])
        hT = nl.ndarray((P, KI, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ki in range(KI):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=h[0:P, ki * P:(ki + 1) * P])
            nisa.tensor_copy(dst=hT[0:P, ki, 0:P], src=tp[0:P, 0:P],
                             engine=nisa.scalar_engine if ki % 2 else nisa.vector_engine)
        # the down block scales of this program's rows, on every partition: [ki, ob]
        ds = nl.ndarray((P, KI, H // P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds[0:P, 0:KI, 0:H // P],
                      src=down_scale.ap(pattern=[[0, P], [H // P, KI], [1, H // P]], offset=(i0 // P) * (H // P),
                                        scalar_offset=e, indirect_dim=0))
        # 4. down, token-major, in two halves of the output columns (HALF PSUM banks each), scatter-added
        for half in range(2):
            c0 = half * HALF * COLS
            W = HALF * COLS
            acc = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.psum)
            for ki in range(KI):
                d8 = nl.ndarray((P, W), dtype=down.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=d8[0:P, 0:W], src=down.ap(pattern=[[H, P], [1, W]], offset=(i0 + ki * P) * H + c0,
                                                              scalar_offset=e, indirect_dim=0), dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
                db = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
                dequant(db[0:P, 0:W], d8[0:P, 0:W], ds, KI * (H // P), ki * (H // P) + c0 // P, W // P, ki)
                for c in range(HALF):
                    nisa.nc_matmul(dst=acc[0:P, c * COLS:(c + 1) * COLS], stationary=hT[0:P, ki, 0:P],
                                   moving=db[0:P, c * COLS:(c + 1) * COLS], accumulate=(ki > 0))
            y = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y[0:P, 0:W], src=acc[0:P, 0:W])
            rows = out.ap(pattern=[[H, P], [1, W]], offset=c0, vector_offset=to[0:P, 0:1], indirect_dim=0)
            nisa.dma_compute(dst=rows, srcs=[rows, y[0:P, 0:W]], reduce_op=nl.add, oob_mode=nisa.oob_mode.skip)
        nisa.tensor_scalar(dst=slot[0:1, 0:1], data=slot[0:1, 0:1], op0=nl.add, operand0=1)

    nl.fori_loop(0, n_reg, block)
    return out


def moe_blocks(aff, P: int = 128, top_k: int = 0):
    """The block list of ``moe_prefill_fp8`` from the routing weights ``aff`` [T, E] (zero off a token's experts), in
    torch (the caller's graph): (tok [NB, P], tok_out [NB, 2, P], aff [NB, P], bexp [NB, 1], nblk [1, 1]). Expert e's
    routed tokens fill ceil(count_e / P) consecutive blocks; NB = T E / P + E bounds the blocks of any routing.""" 
    import torch

    T, E = aff.shape
    # (with ``top_k``: a token routes to at most top_k of the E experts, so the blocks are bounded by T min(k, E) / P
    # + E; the block list's gathers scale with NB: 576 -> 128 blocks for the TP layout's 64 local experts)
    NB = T * (min(top_k, E) if top_k else E) // P + E
    routed = aff > 0
    count = routed.sum(0)
    nbe = (count + P - 1) // P
    start = torch.cumsum(nbe, 0) - nbe
    ar = torch.arange(T, device=aff.device)
    # each expert's routed tokens first, in token order
    order = torch.argsort((~routed).to(torch.int32) * T + ar[:, None], dim=0)
    b = torch.arange(NB, device=aff.device)
    # the block's expert: the last expert with blocks that starts at or before it (experts without tokens own none)
    owns = (b[:, None] >= start[None, :]) & (nbe[None, :] > 0)
    be = torch.where(owns, torch.arange(E, device=aff.device)[None, :], torch.full_like(owns, -1, dtype=torch.int64))
    be = be.amax(1).clamp(min=0)
    pos = (b - start[be])[:, None] * P + torch.arange(P, device=aff.device)[None, :]
    live = (b < nbe.sum())[:, None] & (pos >= 0) & (pos < count[be][:, None])
    tokidx = order.t().reshape(-1)[be[:, None] * T + pos.clamp(min=0, max=T - 1)]
    tok = torch.where(live, tokidx, torch.full_like(tokidx, T))
    a = torch.where(live, aff[tok.clamp(max=T - 1), be[:, None].expand(-1, P)], torch.zeros_like(pos, dtype=aff.dtype))
    tok_out = torch.stack([torch.where(live, tok, torch.full_like(tok, 2 * T)),
                           torch.where(live, tok + T, torch.full_like(tok, 2 * T))], 1)
    return (tok.to(torch.int32), tok_out.to(torch.int32), a.contiguous(), be.to(torch.int32).reshape(-1, 1),
            nbe.sum().to(torch.int32).reshape(1, 1))
