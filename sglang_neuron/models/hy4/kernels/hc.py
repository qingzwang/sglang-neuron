# SPDX-License-Identifier: Apache-2.0
"""Hy4's hyper-connections (iHC) of a prefill in NKI kernels: each 128-token tile of the residual streams is read once
and written once, with every intermediate in SBUF.

The streams are [T, hc_mult = 4, H] bf16 (6 MB a 128-token tile at H = 6144). In the XLA graph each sublayer's
mixing materialized fp32 copies of all four streams in HBM ([1K, 4, 6144] fp32 is 100 MB) and read them back for each
step: ~10 GB of HBM traffic a 4-layer group of a 1K-token prefill, the largest DMA load of the program.

``hc_pre`` (a sublayer's input): per token, the RMS of the flattened streams, the mixes (the flattened streams against
the rows of ``fn``, on the tensor engine: each 128-feature block of the tile is transposed once and is the stationary
operand of one matmul against fn's columns, kept as two bf16 halves hi + lo for ~fp32 accuracy), the sigmoid gates,
and the sublayer input ``sum_i pre_i * stream_i``; with 2 * hc_mult rows, also the post weights.

``hc_post`` (a sublayer's output mixed back): ``stream_i + post_i * out`` in fp32, stored bf16.

The two programs of a logical core take alternate tiles."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
NORM_CHUNK = 2048


def hc_pre(streams, fnc, affine, rms_eps=1e-6, hc_eps=1e-6, magnitude=1.0):
    """streams [T, n * H] bf16 (flattened); fnc [n * H, 2 R] bf16 (fn transposed as hi then lo halves, R rows: n pre
    gates, then n post gates if R = 2 n); affine [2, R] fp32 (each row's scale, then its base) -> (x [T, H] bf16,
    post [T, n] fp32; post is zeros when R = n)."""
    T, W = streams.shape
    R = fnc.shape[1] // 2
    n = 4
    H = W // n
    FC = W // P  # 128-feature blocks
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    x_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="hc_pre_x")
    post_out = nl.ndarray((T, n), dtype=nl.float32, buffer=nl.shared_hbm, name="hc_pre_post")

    fn = nl.ndarray((P, FC, 2 * R), dtype=fnc.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=fn[0:P, 0:FC, 0:2 * R], src=fnc.ap(pattern=[[2 * R, P], [P * 2 * R, FC], [1, 2 * R]], offset=0))
    aff = nl.ndarray((P, 2, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aff[0:P, 0:2, 0:R], src=affine.ap(pattern=[[0, P], [R, 2], [1, R]], offset=0))

    for tb in range(pid, T // P, n_prg):
        t0 = tb * P
        st = nl.ndarray((P, W), dtype=streams.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=st[0:P, 0:W], src=streams.ap(pattern=[[W, P], [1, W]], offset=t0 * W))
        # sum of squares -> 1 / rms
        ssq = nl.ndarray((P, W // NORM_CHUNK), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(W // NORM_CHUNK):
            junk = nl.ndarray((P, NORM_CHUNK), dtype=nl.float32, buffer=nl.sbuf)  # fp32: the reduce sums its values
            nisa.activation(dst=junk[0:P, 0:NORM_CHUNK], op=nl.square, data=st[0:P, c * NORM_CHUNK:(c + 1) * NORM_CHUNK],
                            reduce_op=nl.add, reduce_res=ssq[0:P, c:c + 1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        ms = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms[0:P, 0:1], op=nl.add, data=ssq[0:P, 0:W // NORM_CHUNK], axis=1)
        nisa.tensor_scalar(dst=ms[0:P, 0:1], data=ms[0:P, 0:1], op0=nl.multiply, operand0=1.0 / W, op1=nl.add,
                           operand1=rms_eps)
        rinv = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rinv[0:P, 0:1], op=nl.rsqrt, data=ms[0:P, 0:1])
        # the mixes [128 tokens, R] on the tensor engine
        mx = nl.ndarray((P, 2 * R), dtype=nl.float32, buffer=nl.psum)
        for c in range(FC):
            tp = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=tp[0:P, 0:P], data=st[0:P, c * P:(c + 1) * P])
            sT = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sT[0:P, 0:P], src=tp[0:P, 0:P],
                             engine=nisa.scalar_engine if c % 2 else nisa.vector_engine)
            # fn's hi and lo halves in one matmul (a matmul costs the same at 8 or 16 columns), summed after
            nisa.nc_matmul(dst=mx[0:P, 0:2 * R], stationary=sT[0:P, 0:P], moving=fn[0:P, c, 0:2 * R],
                           accumulate=(c > 0))
        mx2 = nl.ndarray((P, 2 * R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mx2[0:P, 0:2 * R], src=mx[0:P, 0:2 * R])
        mix = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mix[0:P, 0:R], data1=mx2[0:P, 0:R], data2=mx2[0:P, R:2 * R], op=nl.add)
        # z = mix / rms * scale + base; gates = sigmoid(z)
        nisa.tensor_scalar(dst=mix[0:P, 0:R], data=mix[0:P, 0:R], op0=nl.multiply, operand0=rinv[0:P, 0:1])
        nisa.tensor_tensor(dst=mix[0:P, 0:R], data1=mix[0:P, 0:R], data2=aff[0:P, 0, 0:R], op=nl.multiply)
        nisa.tensor_tensor(dst=mix[0:P, 0:R], data1=mix[0:P, 0:R], data2=aff[0:P, 1, 0:R], op=nl.add)
        gate = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=gate[0:P, 0:R], op=nl.sigmoid, data=mix[0:P, 0:R])
        pre = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pre[0:P, 0:n], data=gate[0:P, 0:n], op0=nl.add, operand0=hc_eps)
        pw = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        if R == 2 * n:
            nisa.tensor_scalar(dst=pw[0:P, 0:n], data=gate[0:P, n:2 * n], op0=nl.multiply, operand0=magnitude,
                               op1=nl.add, operand1=hc_eps)
        else:
            nisa.memset(dst=pw[0:P, 0:n], value=0.0)
        nisa.dma_copy(dst=post_out.ap(pattern=[[n, P], [1, n]], offset=t0 * n), src=pw[0:P, 0:n])
        # x = sum_i pre_i * stream_i (fp32), stored bf16
        xa = nl.ndarray((P, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=xa[0:P, 0:H], data=st[0:P, 0:H], op0=nl.multiply, operand0=pre[0:P, 0:1])
        for i in range(1, n):
            nisa.scalar_tensor_tensor(dst=xa[0:P, 0:H], data=st[0:P, i * H:(i + 1) * H], op0=nl.multiply,
                                      operand0=pre[0:P, i:i + 1], op1=nl.add, operand1=xa[0:P, 0:H])
        xb = nl.ndarray((P, H), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xb[0:P, 0:H], src=xa[0:P, 0:H], engine=nisa.scalar_engine)
        nisa.dma_copy(dst=x_out.ap(pattern=[[H, P], [1, H]], offset=t0 * H), src=xb[0:P, 0:H])
    return x_out, post_out


def hc_post(streams, out, post):
    """streams [T, n * H] bf16; out [T, H] bf16; post [T, n] fp32 -> [T, n * H] bf16: stream_i + post_i * out."""
    T, W = streams.shape
    H = out.shape[1]
    n = W // H
    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    new = nl.ndarray((T, W), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="hc_post_streams")
    for tb in range(pid, T // P, n_prg):
        t0 = tb * P
        st = nl.ndarray((P, W), dtype=streams.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=st[0:P, 0:W], src=streams.ap(pattern=[[W, P], [1, W]], offset=t0 * W))
        o = nl.ndarray((P, H), dtype=out.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=o[0:P, 0:H], src=out.ap(pattern=[[H, P], [1, H]], offset=t0 * H))
        pw = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pw[0:P, 0:n], src=post.ap(pattern=[[n, P], [1, n]], offset=t0 * n))
        nw = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(n):
            nisa.scalar_tensor_tensor(dst=nw[0:P, i * H:(i + 1) * H], data=o[0:P, 0:H], op0=nl.multiply,
                                      operand0=pw[0:P, i:i + 1], op1=nl.add, operand1=st[0:P, i * H:(i + 1) * H])
        nisa.dma_copy(dst=new.ap(pattern=[[W, P], [1, W]], offset=t0 * W), src=nw[0:P, 0:W])
    return new
