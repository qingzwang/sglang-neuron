# SPDX-License-Identifier: Apache-2.0
"""Hy4's prefill routed experts on the TP-16 x EP-4 layout (moe_tp.py): this rank's 1/16 slice (128 gate / up
columns, 128 down rows) of each of its EP group's experts, over the tokens routed to the expert only.

The block list is moe_prefill's (``moe_blocks`` over the group's experts); the two programs of a logical core take
alternate blocks (program p blocks p, p + 2, ...: a slice is one 128-row block of the down contraction, too narrow to
split), each scatter-adding into its own half of ``out`` ([2 T, H] fp32, summed by the caller). For each block:

1. its tokens' rows gathered (one indirect DMA, an empty slot out of bounds and skipped), transposed feature-major;
2. the slice's gate / up weights (one 12 KB run a partition, partition-major) dequantized to bf16 in SBUF, the moving
   operand of 48 matmuls into one PSUM bank [token, 256];
3. silu(min(gate, L)) * clamp(up, -L, L) times the token's weight, transposed;
4. down: the fp8 rows the moving operand directly, the block scales (one a 128-column block of H, the slice being one
   row block) applied while PSUM is read out; scatter-added into the output rows.

An odd block count leaves program 1 one empty block (all slots out of bounds: nothing is written)."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
COLS = 512
QW = 1536  # down output columns a PSUM pass (3 banks)


def moe_prefill_tp(x, tok, tok_out, aff, bexp, nit, gate_up, gate_up_scale, down, down_scale, limit=10.0):
    """x [T, H] bf16; tok / tok_out / aff / bexp: moe_prefill.moe_blocks over the group's EL experts; nit [1, 1]
    int32: ceil(blocks / 2); gate_up [EL, 128, H / 128, 256] fp8; gate_up_scale [EL, H / 128, 2]; down [EL, 128, H]
    fp8; down_scale [EL, H / 128] -> out [2 T, H] fp32 (the two programs' halves)."""
    T, H = x.shape
    EL, _, KH, C2 = gate_up.shape
    SW = C2 // 2
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    out = nl.ndarray((2 * T, H), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_prefill_tp_out")

    zero = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero[0:P, 0:H], value=0.0)
    for tb in range(T // P):
        nisa.dma_copy(dst=out.ap(pattern=[[H, P], [1, H]], offset=(pid * T + tb * P) * H), src=zero[0:P, 0:H])

    nb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=nb[0:1, 0:1], src=nit.ap(pattern=[[1, 1], [1, 1]], offset=0))
    n_reg = nisa.register_alloc()
    nisa.register_load(n_reg, nb)
    slot = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)  # 2 it (this program's block: 2 it + pid)
    nisa.memset(dst=slot[0:1, 0:1], value=0)
    KC = KH // 4

    def block(it):
        e = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e[0:1, 0:1], src=bexp.ap(pattern=[[1, 1], [1, 1]], offset=pid, scalar_offset=slot,
                                                     indirect_dim=0))
        ti = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti[0:P, 0:1], src=tok.ap(pattern=[[1, P], [1, 1]], offset=pid * P, scalar_offset=slot,
                                                     indirect_dim=0))
        to = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=to[0:P, 0:1], src=tok_out.ap(pattern=[[1, P], [1, 1]], offset=pid * 2 * P + pid * P,
                                                         scalar_offset=slot, indirect_dim=0))
        af = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=af[0:P, 0:1], src=aff.ap(pattern=[[1, P], [1, 1]], offset=pid * P, scalar_offset=slot,
                                                     indirect_dim=0))
        # the slice's weights and scales (hardware DGE, both queues)
        w8 = nl.ndarray((P, KH, C2), dtype=gate_up.dtype, buffer=nl.sbuf)
        for q in range(4):
            nisa.dma_copy(dst=w8[0:P, q * KC:(q + 1) * KC, 0:C2],
                          src=gate_up.ap(pattern=[[KH * C2, P], [C2, KC], [1, C2]], offset=q * KC * C2,
                                         scalar_offset=e, indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=(nisa.engine.sync, nisa.engine.scalar)[q % 2])
        d8 = nl.ndarray((P, H), dtype=down.dtype, buffer=nl.sbuf)
        for q in range(2):
            nisa.dma_copy(dst=d8[0:P, q * (H // 2):(q + 1) * (H // 2)],
                          src=down.ap(pattern=[[H, P], [1, H // 2]], offset=q * (H // 2), scalar_offset=e,
                                      indirect_dim=0),
                          dge_mode=nisa.dge_mode.hwdge, engine=(nisa.engine.scalar, nisa.engine.sync)[q])
        gs = nl.ndarray((P, KH, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=gs[0:P, 0:KH, 0:2], src=gate_up_scale.ap(pattern=[[0, P], [2, KH], [1, 2]], offset=0,
                                                                     scalar_offset=e, indirect_dim=0))
        ds = nl.ndarray((P, KH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds[0:P, 0:KH], src=down_scale.ap(pattern=[[0, P], [1, KH]], offset=0, scalar_offset=e,
                                                             indirect_dim=0))
        # 1. the tokens, feature-major
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
        # 2. gate / up: dequantized a quarter at a time (vector), [token, 256] in one PSUM bank
        wb = nl.ndarray((P, KH, C2), dtype=nl.bfloat16, buffer=nl.sbuf)
        for q in range(4):
            nisa.tensor_tensor(dst=wb.ap(pattern=[[KH * C2, P], [C2, KC], [SW, 2], [1, SW]], offset=q * KC * C2),
                               data1=w8.ap(pattern=[[KH * C2, P], [C2, KC], [SW, 2], [1, SW]], offset=q * KC * C2),
                               data2=gs.ap(pattern=[[KH * 2, P], [2, KC], [1, 2], [0, SW]], offset=q * KC * 2),
                               op=nl.multiply, engine=nisa.vector_engine)
        gu = nl.ndarray((P, C2), dtype=nl.float32, buffer=nl.psum)
        for kb in range(KH):
            nisa.nc_matmul(dst=gu[0:P, 0:C2], stationary=xT[0:P, kb, 0:P], moving=wb[0:P, kb, 0:C2],
                           accumulate=(kb > 0))
        # 3. the activation, transposed
        gt = nl.ndarray((P, SW), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gt[0:P, 0:SW], data=gu[0:P, 0:SW], op0=nl.minimum, operand0=limit)
        nisa.activation(dst=gt[0:P, 0:SW], op=nl.silu, data=gt[0:P, 0:SW])
        up = nl.ndarray((P, SW), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up[0:P, 0:SW], data=gu[0:P, SW:C2], op0=nl.minimum, operand0=limit, op1=nl.maximum,
                           operand1=-limit)
        nisa.tensor_tensor(dst=gt[0:P, 0:SW], data1=gt[0:P, 0:SW], data2=up[0:P, 0:SW], op=nl.multiply)
        h = nl.ndarray((P, SW), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h[0:P, 0:SW], data=gt[0:P, 0:SW], op0=nl.multiply, operand0=af[0:P, 0:1])
        tph = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tph[0:P, 0:P], data=h[0:P, 0:P])
        hT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hT[0:P, 0:P], src=tph[0:P, 0:P])
        # 4. down, fp8 moving; the column blocks' scales while PSUM is read; scatter-added
        for q in range(H // QW):
            c0 = q * QW
            acc = nl.ndarray((P, QW), dtype=nl.float32, buffer=nl.psum)
            for c in range(QW // COLS):
                nisa.nc_matmul(dst=acc[0:P, c * COLS:(c + 1) * COLS], stationary=hT[0:P, 0:P],
                               moving=d8[0:P, c0 + c * COLS:c0 + (c + 1) * COLS])
            y = nl.ndarray((P, QW), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y.ap(pattern=[[QW, P], [P, QW // P], [1, P]], offset=0),
                               data1=acc.ap(pattern=[[QW, P], [P, QW // P], [1, P]], offset=0),
                               data2=ds.ap(pattern=[[KH, P], [1, QW // P], [0, P]], offset=c0 // P), op=nl.multiply)
            rows = out.ap(pattern=[[H, P], [1, QW]], offset=c0, vector_offset=to[0:P, 0:1], indirect_dim=0)
            nisa.dma_compute(dst=rows, srcs=[rows, y[0:P, 0:QW]], reduce_op=nl.add, oob_mode=nisa.oob_mode.skip)
        nisa.tensor_scalar(dst=slot[0:1, 0:1], data=slot[0:1, 0:1], op0=nl.add, operand0=n_prg)

    nl.fori_loop(0, n_reg, block)
    return out
