# SPDX-License-Identifier: Apache-2.0
"""Hy4's decode routed experts in one NKI kernel: only the local experts some token of the step was routed to.

Per rank and step, for T <= 128 decode tokens, over the rank's E local block-FP8 experts:

    out[p] = sum over the hit experts e of  (silu(min(x Wg_e, L)) * clamp(x Wu_e, -L, L) * aff[:, e]) Wd_e

``hit`` lists the hit local experts first and ``n_hit`` counts them; the kernel walks them in a device loop
(``nl.fori_loop`` over a register), so an expert no token chose costs nothing.

Weights are trn2-halved e4m3 (weights.py) in the transposed layouts: gate_up [E, H, 2I] (gate columns then up), down
[E, I, H], with their 128 x 128 block scales in the same blocking, gate_up_scale [E, H / 128, 2I / 128] and down_scale
[E, I / 128, H / 128]. Every matmul is one weight block: the FP8 block [128 rows, 128 columns] is the stationary
operand and the tokens' activations [128 rows, T] stream past it, so the block's product [128 columns, T] in PSUM is
scaled by the block's one scale and accumulated in SBUF (fp32). The results come out transposed, [columns, T], which
is the layout the down projection streams, so no activation is transposed but the input (by the caller).

The two programs of a logical core take one half of I each: their gate / up columns, the activation, and the down
projection's contraction over their half; ``out`` [2, H, T] fp32 holds each program's partial, transposed, which the
caller sums."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def moe_decode_fp8(xT, affT, hit, n_hit, gate_up, gate_up_scale, down, down_scale, limit=10.0):
    """xT [H, T] bf16 (the tokens' activations, transposed); affT [E, T] fp32 (zero off a token's experts); hit [1, E]
    int32; n_hit [1, 1] int32 -> [2, H, T] fp32 partials."""
    H, T = xT.shape
    E, I, _ = down.shape  # noqa: E741
    KH = H // P
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    IH = I // n_prg  # this program's half of I
    OB = IH // P  # its output blocks of the gate, and of the up
    KI = IH // P  # its contraction tiles of the down
    OH = H // P  # output blocks of the down
    i0 = pid * IH
    GB = 2 * I // P  # column blocks of gate_up

    out = nl.ndarray((n_prg, H, T), dtype=nl.float32, buffer=nl.shared_hbm, name="moe_decode_out")
    acc = nl.ndarray((P, OH, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=acc[0:P, 0:OH, 0:T], value=0.0)
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
        # the expert's affinity of each token, on every partition
        a = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a[0:P, 0:T], src=affT.ap(pattern=[[0, P], [1, T]], offset=0, scalar_offset=e,
                                                     indirect_dim=0))
        # every block scale of this program's gate / up columns, on every partition: [kb, g, ob]
        gs = nl.ndarray((P, KH, 2, OB), dtype=nl.float32, buffer=nl.sbuf)
        for g in range(2):
            nisa.dma_copy(dst=gs[0:P, 0:KH, g, 0:OB],
                          src=gate_up_scale.ap(pattern=[[0, P], [GB, KH], [1, OB]], offset=(g * I + i0) // P,
                                               scalar_offset=e, indirect_dim=0))
        # gate (g = 0) and up (g = 1) of this program's columns, transposed: [col in block, g, ob, t]
        gu = nl.ndarray((P, 2, OB, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gu[0:P, 0:2, 0:OB, 0:T], value=0.0)
        for kb in range(KH):
            wt = nl.ndarray((P, 2, IH), dtype=gate_up.dtype, buffer=nl.sbuf)
            for g in range(2):
                nisa.dma_copy(dst=wt[0:P, g, 0:IH],
                              src=gate_up.ap(pattern=[[2 * I, P], [1, IH]], offset=kb * P * 2 * I + g * I + i0,
                                             scalar_offset=e, indirect_dim=0))
            for g in range(2):
                for ob in range(OB):
                    ps = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=ps[0:P, 0:T], stationary=wt[0:P, g, ob * P:(ob + 1) * P],
                                   moving=xs[0:P, kb, 0:T])
                    nisa.scalar_tensor_tensor(dst=gu[0:P, g, ob, 0:T], data=ps[0:P, 0:T], op0=nl.multiply,
                                              operand0=gs[0:P, kb, g, ob:ob + 1], op1=nl.add,
                                              operand1=gu[0:P, g, ob, 0:T])
        # silu(min(gate, L)) * clamp(up, -L, L) * affinity, as the down's bf16 input [col in block, ob, t]
        gt = nl.ndarray((P, OB, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gt[0:P, 0:OB, 0:T], data=gu[0:P, 0, 0:OB, 0:T], op0=nl.minimum, operand0=limit)
        nisa.activation(dst=gt[0:P, 0:OB, 0:T], op=nl.silu, data=gt[0:P, 0:OB, 0:T])
        up = nl.ndarray((P, OB, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=up[0:P, 0:OB, 0:T], data=gu[0:P, 1, 0:OB, 0:T], op0=nl.minimum, operand0=limit,
                           op1=nl.maximum, operand1=-limit)
        hb = nl.ndarray((P, OB, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=hb[0:P, 0:OB, 0:T], data1=gt[0:P, 0:OB, 0:T], data2=up[0:P, 0:OB, 0:T],
                           op=nl.multiply)
        hf = nl.ndarray((P, OB, T), dtype=nl.float32, buffer=nl.sbuf)
        for ob in range(OB):
            nisa.tensor_tensor(dst=hf[0:P, ob, 0:T], data1=hb[0:P, ob, 0:T], data2=a[0:P, 0:T], op=nl.multiply)
        hT = nl.ndarray((P, KI, T), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=hT[0:P, 0:KI, 0:T], src=hf[0:P, 0:OB, 0:T])
        # down: this program's rows of I (KI tiles) against every column block of H
        ds = nl.ndarray((P, KI, OH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ds[0:P, 0:KI, 0:OH],
                      src=down_scale.ap(pattern=[[0, P], [OH, KI], [1, OH]], offset=(i0 // P) * OH,
                                        scalar_offset=e, indirect_dim=0))
        for kb in range(KI):
            wd = nl.ndarray((P, H), dtype=down.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wd[0:P, 0:H], src=down.ap(pattern=[[H, P], [1, H]], offset=(i0 + kb * P) * H,
                                                        scalar_offset=e, indirect_dim=0))
            for ob in range(OH):
                ps = nl.ndarray((P, T), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps[0:P, 0:T], stationary=wd[0:P, ob * P:(ob + 1) * P], moving=hT[0:P, kb, 0:T])
                nisa.scalar_tensor_tensor(dst=acc[0:P, ob, 0:T], data=ps[0:P, 0:T], op0=nl.multiply,
                                          operand0=ds[0:P, kb, ob:ob + 1], op1=nl.add, operand1=acc[0:P, ob, 0:T])
        nisa.tensor_scalar(dst=slot[0:1, 0:1], data=slot[0:1, 0:1], op0=nl.add, operand0=1)

    nl.fori_loop(0, n_reg, expert_body)
    nisa.dma_copy(dst=out.ap(pattern=[[T, P], [P * T, OH], [1, T]], offset=pid * H * T), src=acc[0:P, 0:OH, 0:T])
    return out
