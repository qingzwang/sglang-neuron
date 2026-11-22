# SPDX-License-Identifier: Apache-2.0
"""Hy4's decode MoE block in one NKI kernel: the router, the expert choice, this rank's routed experts and its shard
of the shared expert (the giga kernel's first block).

In the XLA graph the block was a chain of small ops around the routed-expert kernel (the router's fp32 matmul, the
sigmoid, a top-8, a gather, the renormalization, the local affinities, the hit order, the shared expert's two
matmuls and its activation), each paying a fixed ~10-20 us on a decode step's few tokens. Here, for T <= 32 tokens:

1. the router logits [T, E] accumulate in PSUM (the tokens stationary, the router's rows moving; bf16 inputs with
   fp32 accumulation, as x.float() @ router.float());
2. sigmoid, + the choice bias, the top 8 (max8 / find_index8: one instruction each), the chosen scores gathered
   (nc_n_gather) and renormalized, and each local expert's weight of each token;
3. the hit local experts (compacted by nonzero_with_count) run in one device loop, the expert index read from the
   hit list: the column-tiled body of moe_decode_fp8_ct (four matmuls at once on the tensor engine's 32-column tiles, the
   block scales in the activations);
4. the shared expert's rows of this program, computed with the router (one weight load of the tokens a row block),
   its down projection column-tiled into the accumulator before the routed experts add theirs.

The two programs of a logical core take one half of the routed experts' intermediate dimension and of the shared
expert's rows each; ``out`` [2, T, H] fp32 holds their partials, which the caller sums (and all-reduces over the
ranks)."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
TILE = 32
NT = P // TILE
KB_DMA = 4


def moe_block_decode(xT, router, router_bias, base, sel, gate_up, gate_up_scale, down, down_scale, sh_gate_up,
                     sh_down, top_k=8, scaling=1.0, limit=10.0):
    """xT [H, T] bf16 (the normed tokens, transposed); router [H, NE] bf16; router_bias [1, NE] fp32; base [1, 1]
    int32 (this rank's first expert); sel [T, 128] fp32 (sel[t, 32 c + t] = 1: a token's rows of the column tiles);
    gate_up [E, H, 2I] / gate_up_scale [E, H/128, 2I/128] / down [E, I, H] / down_scale [E, I/128, H/128] (this
    rank's E experts, trn2-halved e4m3); sh_gate_up [H, n_prg, 2 Rs / n_prg] bf16 (each program's gate rows then its up rows, adjacent: shared_gate_up_layout); sh_down [Rs, H] bf16 ->
    [2, T, H] fp32 partials."""
    H, T = xT.shape
    NE = router.shape[1]
    E, I, _ = down.shape  # noqa: E741
    RS = sh_down.shape[0]
    KH = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    IH = I // n_prg
    i0 = pid * IH
    OB = IH // P
    KI = IH // P
    HB = H // P
    GB = 2 * I // P
    OQ = OB // NT
    HQ = HB // NT
    RH = RS // n_prg  # this program's shared rows
    r0 = pid * RH

    out = nl.ndarray((n_prg, T, H), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_block_out")
    acc = nl.ndarray((P, HQ, P), dtype=nl.float32, buffer=nl.sbuf)  # initialized by the shared expert's output
    xs = nl.ndarray((P, KH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xs[0:P, 0:KH, 0:T], src=xT.ap(pattern=[[T, P], [P * T, KH], [1, T]], offset=0))

    def tiled(dst, stationary, moving, c):
        nisa.nc_matmul(dst=dst, stationary=stationary, moving=moving, tile_position=(0, c * TILE),
                       tile_size=(P, TILE), accumulate=None)

    # ---- 1. the router logits [T, NE] and the shared expert's gate / up [T, 2 RH], one row block at a time ---------
    lg = nl.ndarray((T, NE), dtype=nl.float32, buffer=nl.psum)
    sgu = nl.ndarray((T, 2 * RH), dtype=nl.float32, buffer=nl.psum)
    for kb in range(KH):
        if kb % KB_DMA == 0:
            rw = nl.ndarray((P, KB_DMA, NE), dtype=router.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=rw[0:P, 0:KB_DMA, 0:NE], src=router.ap(pattern=[[NE, P], [P * NE, KB_DMA], [1, NE]],
                                                                     offset=kb * P * NE),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            sw = nl.ndarray((P, KB_DMA, 2 * RH), dtype=sh_gate_up.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sw[0:P, 0:KB_DMA, 0:2 * RH],
                          src=sh_gate_up.ap(pattern=[[n_prg * 2 * RH, P], [P * n_prg * 2 * RH, KB_DMA], [1, 2 * RH]],
                                            offset=kb * P * n_prg * 2 * RH + pid * 2 * RH),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.scalar)
        nisa.nc_matmul(dst=lg[0:T, 0:NE], stationary=xs[0:P, kb, 0:T], moving=rw[0:P, kb % KB_DMA, 0:NE],
                       accumulate=None)
        nisa.nc_matmul(dst=sgu[0:T, 0:2 * RH], stationary=xs[0:P, kb, 0:T], moving=sw[0:P, kb % KB_DMA, 0:2 * RH],
                       accumulate=None)
    # the shared expert, finished before the routed experts (only the tokens feed it)
    sg = nl.ndarray((T, RH), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sg[0:T, 0:RH], op=nl.silu, data=sgu[0:T, 0:RH])
    sh = nl.ndarray((T, RH), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sh[0:T, 0:RH], data1=sg[0:T, 0:RH], data2=sgu[0:T, RH:2 * RH], op=nl.multiply)
    tps = nl.ndarray((RH, T), dtype=nl.bfloat16, buffer=nl.psum)
    nisa.nc_transpose(dst=tps[0:RH, 0:T], data=sh[0:T, 0:RH])
    shT = nl.ndarray((RH, T), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=shT[0:RH, 0:T], src=tps[0:RH, 0:T])
    sd = nl.ndarray((RH, H), dtype=sh_down.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sd[0:RH, 0:H], src=sh_down.ap(pattern=[[H, RH], [1, H]], offset=r0 * H),
                  dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    spd = nl.ndarray((P, HQ, P), dtype=nl.float32, buffer=nl.psum)
    for q in range(HQ):
        for c in range(NT):
            hb = q * NT + c
            tiled(spd[c * TILE:c * TILE + T, q, 0:P], shT[0:RH, 0:T], sd[0:RH, hb * P:(hb + 1) * P], c)
    nisa.tensor_copy(dst=acc[0:P, 0:HQ, 0:P], src=spd[0:P, 0:HQ, 0:P])
    # ---- 2. the choice ----------------------------------------------------------------------------------------------
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
    # each chosen expert relative to this rank's first: rel[t, k] = id - base
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
    # the weights in the column tiles' layout [32 c + t, e], and whether any token chose each expert
    st = nl.ndarray((T, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=st[0:T, 0:P], src=sel.ap(pattern=[[P, T], [1, P]], offset=0))
    at_ps = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=at_ps[0:P, 0:E], stationary=st[0:T, 0:P], moving=A[0:T, 0:E])
    at = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=at[0:P, 0:E], src=at_ps[0:P, 0:E])
    hitm = nl.ndarray((T, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hitm[0:T, 0:E], data=A[0:T, 0:E], op0=nl.greater, operand0=0.0)
    onet = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=onet[0:T, 0:1], value=1.0)
    cnt_ps = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=cnt_ps[0:1, 0:E], stationary=onet[0:T, 0:1], moving=hitm[0:T, 0:E])
    cntf = nl.ndarray((1, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=cntf[0:1, 0:E], data=cnt_ps[0:1, 0:E], op0=nl.minimum, operand0=1.0)
    # the hit experts' list (nonzero_with_count on partition 0) and the weights [e, 32 c + t], for the expert loop
    hf = nl.ndarray((P, E), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=hf[0:P, 0:E], value=0.0)
    nisa.tensor_copy(dst=hf[0:1, 0:E], src=cntf[0:1, 0:E])
    nz = nl.ndarray((P, E + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(dst=nz[0:P, 0:E + 1], src=hf[0:P, 0:E], index_offset=0, padding_val=0)
    hits = nl.ndarray((1, E + 1), dtype=nl.int32, buffer=nl.private_hbm, name="moe_block_hits")
    nisa.dma_copy(dst=hits.ap(pattern=[[E + 1, 1], [1, E + 1]], offset=0), src=nz[0:1, 0:E + 1])
    atT = nl.ndarray((E, P), dtype=nl.float32, buffer=nl.private_hbm, name="moe_block_aff")
    nisa.dma_copy(dst=atT.ap(pattern=[[1, P], [P, E]], offset=0), src=at[0:P, 0:E])
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, nz[0:1, E:E + 1])
    slot = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=slot[0:1, 0:1], value=0)

    # ---- 3. the hit routed experts, in one device loop (the expert index from the hit list) --------------------------
    def body(it):
        e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e[0:1, 0:1], src=hits.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=slot,
                                                     indirect_dim=1))
        # this expert's weight of each token, in the column tiles' layout [32 c + t]
        ae = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ae[0:P, 0:1], src=atT.ap(pattern=[[1, P], [1, 1]], offset=0, scalar_offset=e,
                                                      indirect_dim=0))
        gs = nl.ndarray((P, KH, 2, OB), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(2):
            nisa.dma_copy(dst=gs[0:P, 0:KH, g, 0:OB],
                          src=gate_up_scale.ap(pattern=[[0, P], [GB, KH], [1, OB]],
                                               offset=(g * I + i0) // P, scalar_offset=e,
                                               indirect_dim=0))
        gu = nl.ndarray((P, 2, OQ, P), dtype=nl.float32, buffer=nl.psum)
        for kb in range(KH):
            if kb % KB_DMA == 0:
                wts = nl.ndarray((P, KB_DMA, 2, IH), dtype=gate_up.dtype, buffer=nl.sbuf)
                for g in range(2):
                    nisa.dma_copy(dst=wts[0:P, 0:KB_DMA, g, 0:IH],
                                  src=gate_up.ap(pattern=[[2 * I, P], [P * 2 * I, KB_DMA], [1, IH]],
                                                 offset=kb * P * 2 * I + g * I + i0, scalar_offset=e,
                                                 indirect_dim=0),
                                  dge_mode=nisa.dge_mode.hwdge,
                                  engine=nisa.engine.sync if (kb // KB_DMA + g) % 2 == 0 else nisa.engine.scalar)
            xsc = nl.ndarray((P, 2, OB, T), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=xsc[0:P, 0:2, 0:OB, 0:T],
                               data1=xs.ap(pattern=[[KH * T, P], [0, 2], [0, OB], [1, T]], offset=kb * T),
                               data2=gs.ap(pattern=[[KH * 2 * OB, P], [OB, 2], [1, OB], [0, T]],
                                           offset=kb * 2 * OB), op=nl.multiply)
            for g in range(2):
                for q in range(OQ):
                    for c in range(NT):
                        ob = q * NT + c
                        tiled(gu[c * TILE:c * TILE + T, g, q, 0:P], xsc[0:P, g, ob, 0:T],
                              wts[0:P, kb % KB_DMA, g, ob * P:(ob + 1) * P], c)
        gt = nl.ndarray((P, OQ, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gt[0:P, 0:OQ, 0:P], data=gu[0:P, 0, 0:OQ, 0:P], op0=nl.minimum, operand0=limit)
        nisa.activation(dst=gt[0:P, 0:OQ, 0:P], op=nl.silu, data=gt[0:P, 0:OQ, 0:P])
        up = nl.ndarray((P, OQ, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up[0:P, 0:OQ, 0:P], data=gu[0:P, 1, 0:OQ, 0:P], op0=nl.minimum, operand0=limit,
                           op1=nl.maximum, operand1=-limit)
        nisa.tensor_tensor(dst=gt[0:P, 0:OQ, 0:P], data1=gt[0:P, 0:OQ, 0:P], data2=up[0:P, 0:OQ, 0:P],
                           op=nl.multiply)
        h = nl.ndarray((P, OQ, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h[0:P, 0:OQ, 0:P], data=gt[0:P, 0:OQ, 0:P], op0=nl.multiply,
                           operand0=ae[0:P, 0:1])
        hT = nl.ndarray((P, OQ, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for q in range(OQ):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=h[0:P, q, 0:P])
            nisa.tensor_copy(dst=hT[0:P, q, 0:P], src=tp[0:P, 0:P])
        ds = nl.ndarray((P, KI, HB), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds[0:P, 0:KI, 0:HB],
                      src=down_scale.ap(pattern=[[0, P], [HB, KI], [1, HB]],
                                        offset=(i0 // P) * HB, scalar_offset=e, indirect_dim=0))
        wd = nl.ndarray((P, KI, H), dtype=down.dtype, buffer=nl.sbuf)
        for ki in range(KI):
            nisa.dma_copy(dst=wd[0:P, ki, 0:H], src=down.ap(pattern=[[H, P], [1, H]],
                                                              offset=(i0 + ki * P) * H, scalar_offset=e,
                                                              indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge,
                          engine=nisa.engine.sync if ki % 2 == 0 else nisa.engine.scalar)
        hs = nl.ndarray((P, KI, HB, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        for ki in range(KI):
            q, c = ki // NT, ki % NT
            nisa.tensor_tensor(dst=hs[0:P, ki, 0:HB, 0:T],
                               data1=hT.ap(pattern=[[OQ * P, P], [0, HB], [1, T]], offset=q * P + c * TILE),
                               data2=ds.ap(pattern=[[KI * HB, P], [1, HB], [0, T]], offset=ki * HB),
                               op=nl.multiply)
        pd = nl.ndarray((P, HQ, P), dtype=nl.float32, buffer=nl.psum)
        for ki in range(KI):
            for q in range(HQ):
                for c in range(NT):
                    hb = q * NT + c
                    tiled(pd[c * TILE:c * TILE + T, q, 0:P], hs[0:P, ki, hb, 0:T],
                          wd[0:P, ki, hb * P:(hb + 1) * P], c)
        nisa.tensor_tensor(dst=acc[0:P, 0:HQ, 0:P], data1=acc[0:P, 0:HQ, 0:P], data2=pd[0:P, 0:HQ, 0:P],
                           op=nl.add)
        nisa.tensor_scalar(dst=slot[0:1, 0:1], data=slot[0:1, 0:1], op0=nl.add, operand0=1)

    nl.fori_loop(0, n_reg, body)

    # acc[32 c + t, q, column] -> out[pid, t, (4 q + c) * 128 + column]
    for c in range(NT):
        nisa.dma_copy(dst=out.ap(pattern=[[H, T], [NT * P, HQ], [1, P]], offset=pid * T * H + c * P),
                      src=acc[c * TILE:c * TILE + T, 0:HQ, 0:P])
    return out


def tile_rows(T: int, device=None):
    """sel [T, 128] fp32 of moe_block_decode: token t's row in each of the four 32-row column tiles."""
    import torch

    sel = torch.zeros(T, P, device=device)
    t = torch.arange(T, device=device)
    for c in range(NT):
        sel[t, c * TILE + t] = 1.0
    return sel


def shared_gate_up_layout(gate_up, n_prg: int = 2):
    """A shared expert's [H, 2 R] (gate then up) as moe_block_decode's [H, n_prg, 2 R / n_prg]: each program's gate
    rows then its up rows, adjacent (one matmul a row block)."""
    H, R2 = gate_up.shape
    rh = R2 // 2 // n_prg
    gate, up = gate_up[:, :R2 // 2], gate_up[:, R2 // 2:]
    return torch_cat_programs(gate, up, rh, n_prg)


def torch_cat_programs(gate, up, rh, n_prg):
    import torch

    return torch.stack([torch.cat([gate[:, p * rh:(p + 1) * rh], up[:, p * rh:(p + 1) * rh]], dim=1)
                        for p in range(n_prg)], dim=1).contiguous()
