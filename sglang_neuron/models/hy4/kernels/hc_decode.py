# SPDX-License-Identifier: Apache-2.0
"""Hy4's hyper-connections of a decode step in one NKI kernel: a sublayer's output mixed back into the streams (post)
and the next sublayer's input mixed from them (pre), fused.

A decode step has few tokens (T <= 16), so the prefill kernels' token-per-partition layout leaves the engines nearly
idle, and in the XLA graph each of a layer's four mixings is a chain of small ops (~20 us each, ~6 ms a step). Here
a token's flattened streams [n * H = 24576] spread over all 128 partitions, 192 elements each: partition p holds
elements p * 192 .. p * 192 + 191, so stream i is partitions 32 i .. 32 i + 31 and the sublayer output [H], viewed
[32, 192], is broadcast to the four groups.

* post: ``s_i + post_i * out`` is one instruction (post_i a per-partition scalar of its group);
* the norm and the mixes: per-partition partial sums (of the squares, and of the products with the R rows of fn laid
  out alike), summed over the partitions by one matmul against ones;
* the gates on one partition, then x = sum_i pre_i * s_i: one matmul whose stationary operand places pre_i on the
  rows of group i, summing the groups into [32, 192] (the output [H]).

Everything is fp32 but the streams and outputs (bf16), as the XLA form. The two programs of a logical core take
alternate tokens."""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


def hc_decode(streams, out, post, fn, affine, ones, place, rms_eps=1e-6, hc_eps=1e-6, magnitude=1.0):
    """streams [T, n * H] bf16; out [T, H] bf16 (the sublayer's output); post [T, n] fp32 (its post weights); fn
    [R, n * H] fp32 (the next mixer's rows); affine [2, R] fp32 (scale, base); ones [128, 1] fp32; place [128, n, 32]
    fp32 (place[p, i, r] = 1 where p = 32 i + r) -> (streams' [T, n * H] bf16, x [T, H] bf16, post' [T, n] fp32: zeros
    when R = n)."""
    T, W = streams.shape
    H = out.shape[1]
    n = W // H
    R = fn.shape[0]
    F = W // P  # elements a partition holds of a token's streams
    G = P // n  # partitions of a stream
    new = nl.ndarray((T, W), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="hc_decode_streams")
    x_out = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.shared_hbm, name="hc_decode_x")
    post_out = nl.ndarray((T, n), dtype=nl.float32, buffer=nl.shared_hbm, name="hc_decode_post")

    # fn laid out as the streams: fnl[p, j, f] = fn[j, p * F + f]
    fnl = nl.ndarray((P, R, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=fnl[0:P, 0:R, 0:F], src=fn.ap(pattern=[[F, P], [W, R], [1, F]], offset=0))
    aff = nl.ndarray((1, 2, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=aff[0:1, 0:2, 0:R], src=affine.ap(pattern=[[2 * R, 1], [R, 2], [1, R]], offset=0))
    one = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=one[0:P, 0:1], src=ones.ap(pattern=[[1, P], [1, 1]], offset=0))
    plc = nl.ndarray((P, n, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=plc[0:P, 0:n, 0:G], src=place.ap(pattern=[[n * G, P], [G, n], [1, G]], offset=0))

    n_prg = nl.num_programs(axes=0)
    pid = nl.program_id(axis=0)
    for t in range(pid, T, n_prg):
        s = nl.ndarray((P, F), dtype=streams.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=s[0:P, 0:F], src=streams.ap(pattern=[[F, P], [1, F]], offset=t * W))
        o = nl.ndarray((P, F), dtype=out.dtype, buffer=nl.sbuf)
        pw = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        for i in range(n):
            # the output [H] as [32, 192] on every stream's group, and the group's post weight
            nisa.dma_copy(dst=o[i * G:(i + 1) * G, 0:F], src=out.ap(pattern=[[F, G], [1, F]], offset=t * H))
            nisa.dma_copy(dst=pw[i * G:(i + 1) * G, 0:1], src=post.ap(pattern=[[0, G], [1, 1]], offset=t * n + i))
        # post: s + post_i * out (fp32), rounded to the bf16 streams
        sf = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=sf[0:P, 0:F], data=o[0:P, 0:F], op0=nl.multiply, operand0=pw[0:P, 0:1],
                                  op1=nl.add, operand1=s[0:P, 0:F])
        sb = nl.ndarray((P, F), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sb[0:P, 0:F], src=sf[0:P, 0:F])
        nisa.dma_copy(dst=new.ap(pattern=[[F, P], [1, F]], offset=t * W), src=sb[0:P, 0:F])
        # the mixer reads the rounded streams
        nisa.tensor_copy(dst=sf[0:P, 0:F], src=sb[0:P, 0:F])
        # partials [p, 0] = sum of squares, [p, 1 + j] = sum of s * fn_j
        part = nl.ndarray((P, 1 + R), dtype=nl.float32, buffer=nl.sbuf)
        sq = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sq[0:P, 0:F], op=nl.square, data=sf[0:P, 0:F], reduce_op=nl.add,
                        reduce_res=part[0:P, 0:1], reduce_cmd=nisa.reduce_cmd.reset_reduce)
        prod = nl.ndarray((P, R, F), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod[0:P, 0:R, 0:F], data1=fnl[0:P, 0:R, 0:F],
                           data2=sf.ap(pattern=[[F, P], [0, R], [1, F]], offset=0), op=nl.multiply)
        nisa.tensor_reduce(dst=part[0:P, 1:1 + R], op=nl.add, data=prod[0:P, 0:R, 0:F], axis=2)
        tot = nl.ndarray((1, 1 + R), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=tot[0:1, 0:1 + R], stationary=one[0:P, 0:1], moving=part[0:P, 0:1 + R])
        tt = nl.ndarray((1, 1 + R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tt[0:1, 0:1 + R], src=tot[0:1, 0:1 + R])
        # 1 / rms, the mixes, the gates
        rinv = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rinv[0:1, 0:1], data=tt[0:1, 0:1], op0=nl.multiply, operand0=1.0 / W, op1=nl.add,
                           operand1=rms_eps)
        nisa.activation(dst=rinv[0:1, 0:1], op=nl.rsqrt, data=rinv[0:1, 0:1])
        mix = nl.ndarray((1, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mix[0:1, 0:R], data=tt[0:1, 1:1 + R], op0=nl.multiply, operand0=rinv[0:1, 0:1])
        nisa.tensor_tensor(dst=mix[0:1, 0:R], data1=mix[0:1, 0:R], data2=aff[0:1, 0, 0:R], op=nl.multiply)
        nisa.tensor_tensor(dst=mix[0:1, 0:R], data1=mix[0:1, 0:R], data2=aff[0:1, 1, 0:R], op=nl.add)
        gate = nl.ndarray((1, R), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=gate[0:1, 0:R], op=nl.sigmoid, data=mix[0:1, 0:R])
        pre = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pre[0:1, 0:n], data=gate[0:1, 0:n], op0=nl.add, operand0=hc_eps)
        pn = nl.ndarray((1, n), dtype=nl.float32, buffer=nl.sbuf)
        if R == 2 * n:
            nisa.tensor_scalar(dst=pn[0:1, 0:n], data=gate[0:1, n:2 * n], op0=nl.multiply, operand0=magnitude,
                               op1=nl.add, operand1=hc_eps)
        else:
            nisa.memset(dst=pn[0:1, 0:n], value=0.0)
        nisa.dma_copy(dst=post_out.ap(pattern=[[n, 1], [1, n]], offset=t * n), src=pn[0:1, 0:n])
        # x = sum_i pre_i * s_i: the placement scaled by pre (broadcast to every partition by a matmul of ones)
        preb = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.psum)
        onesrow = nl.ndarray((1, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=onesrow[0:1, 0:P], value=1.0)
        nisa.nc_matmul(dst=preb[0:P, 0:n], stationary=onesrow[0:1, 0:P], moving=pre[0:1, 0:n])
        pb = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pb[0:P, 0:n], src=preb[0:P, 0:n])
        wsel = nl.ndarray((P, n, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=wsel[0:P, 0:n, 0:G], data1=plc[0:P, 0:n, 0:G],
                           data2=pb.ap(pattern=[[n, P], [1, n], [0, G]], offset=0), op=nl.multiply)
        # rows of group i hold stream i: the place matrix's nonzero is at [32 i + r, i, r]; sum over i of wsel[:, i, :]
        ws = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ws[0:P, 0:G], op=nl.add, data=wsel.ap(pattern=[[n * G, P], [1, G], [G, n]], offset=0),
                           axis=2)
        xs = nl.ndarray((G, F), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=xs[0:G, 0:F], stationary=ws[0:P, 0:G], moving=sf[0:P, 0:F])
        xb = nl.ndarray((G, F), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xb[0:G, 0:F], src=xs[0:G, 0:F])
        nisa.dma_copy(dst=x_out.ap(pattern=[[F, G], [1, F]], offset=t * H), src=xb[0:G, 0:F])
    return new, x_out, post_out


def decode_constants(n: int, device=None):
    """(ones [128, 1], place [128, n, 128 / n]) fp32 for hc_decode."""
    import torch

    g = P // n
    p = torch.arange(P, device=device)[:, None, None]
    i = torch.arange(n, device=device)[None, :, None]
    r = torch.arange(g, device=device)[None, None, :]
    place = (p == i * g + r).to(torch.float32)
    return torch.ones(P, 1, device=device), place
