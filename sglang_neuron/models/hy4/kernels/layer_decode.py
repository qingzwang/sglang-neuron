# SPDX-License-Identifier: Apache-2.0
"""A whole Hy4 decode layer (a shared one: it reuses the last full layer's selection) in one NKI kernel.

    hc post + pre (the last layer's MLP output into the streams, this layer's attention input)
        -> attn_in (norm, q_a / kv_a / gate in one weight pass, latent norms, q_b, RoPE, the cache rows, the query's
           all-gather) -> attn_core (sparse attention, the statistics' all-gather, the reduce-scatter, the output
           projection, the all-reduce) -> hc post + pre (the attention output in, the MLP input out)
        -> moe_sublayer (norm, routing, shared and routed experts, the all-reduce)

The stages are the standalone kernels, called in one trace: one kernel start a layer instead of five, and the
compiler's scheduler sees the whole layer, so a later stage's weight DMAs (the MoE router's, the output projection's)
can issue during an earlier stage's collectives. Core barriers order the stages (each writes shared HBM the next reads
from both programs). The MLP output's post update is returned (as the next layer's pending input), so the layers of a
group chain through the fused hc post + pre."""

import nki
import nki.isa as nisa
import nki.language as nl

from sglang_neuron.models.hy4.kernels.attn_core import attn_core_decode, load_out_weights
from sglang_neuron.models.hy4.kernels.attn_in import attn_in_decode
from sglang_neuron.models.hy4.kernels.hc_decode import hc_decode, hc_decode_batched
from sglang_neuron.models.hy4.kernels.moe_sublayer import load_router_shared, moe_sublayer_decode


def layer_decode(streams, p_out, p_post, fn_a, aff_a, fn_m, aff_m, ones, place,
                 ln1, q_a, qa_ln, q_b, w_uk, kv_a, kva_ln, gate, cos, sin, cache, write_rows, slots, bias, onehot, sinks,
                 w_uv, o, ln2, router_pm, router_bias, base, gate_up, gate_up_scale, down, down_scale, sh_pm, sh_down,
                 ranks=(), eps=1e-5, hc_eps=1e-6, magnitude=1.0, scale=1.0, top_k=8, scaling=1.0, limit=10.0, t_pad=0,
                 qkv_sh=None, tag=""):
    """streams [T, n H] bf16; p_out [T, H] bf16 and p_post [T, n] fp32: the last layer's MLP output and its post
    weights (zeros: none); the rest as the stage kernels take them -> (streams' [T, n H] bf16, the MLP output [T, H]
    bf16 (all-reduced), its post weights [T, n] fp32, cache)."""
    T = streams.shape[0]
    hc = hc_decode_batched if T > 2 else hc_decode
    s1, x, post_a = hc(streams, p_out, p_post, fn_a, aff_a, ones, place, rms_eps=eps, hc_eps=hc_eps,
                       magnitude=magnitude, tag=tag + "_a")
    nisa.core_barrier(s1, (0, 1))
    nisa.core_barrier(x, (0, 1))
    nisa.core_barrier(post_a, (0, 1))
    # explicit prefetch: each later stage's weights issued right before an earlier stage's collective, so they stream
    # while it runs (the compiler keeps the kernel's program order: it does not hoist them itself)
    pre = {}

    def fetch_out():
        pre["out"] = load_out_weights(w_uv, o)

    def fetch_moe():
        pre["moe"] = load_router_shared(router_pm, sh_pm, sh_down, x.shape[1] // 128, router_bias.shape[1],
                                        sh_down.shape[0])

    query, compact, q_lat, cache, gate_x = attn_in_decode(x, ln1, q_a, qa_ln, q_b, w_uk, kv_a, kva_ln, cos, sin, cache,
                                                          write_rows, gate=gate, t_pad=t_pad, hook=fetch_out,
                                                          ranks=ranks, eps=eps, lat_eps=eps, qkv_sh=qkv_sh, tag=tag)
    nisa.core_barrier(query, (0, 1))
    nisa.core_barrier(gate_x, (0, 1))
    # the rows attn_in wrote (both programs, in place) before attn_core gathers keys from them: an indirect gather
    # is not ordered after the writes otherwise (the sparse attention then misses the newest rows)
    nisa.core_barrier(cache, (0, 1))
    TQ = query.shape[0]
    D = query.shape[2]
    att = attn_core_decode(query.reshape((TQ * query.shape[1] // 128, 128, D)), cache, slots, bias, onehot, sinks,
                           w_uv, gate_x, o, ranks=ranks, scale=scale, t_real=T, pre=pre["out"],
                           hooks={"ml": fetch_moe}, tag=tag)
    nisa.core_barrier(att, (0, 1))
    s2, x2, post_m = hc(s1, att, post_a, fn_m, aff_m, ones, place, rms_eps=eps, hc_eps=hc_eps, magnitude=magnitude,
                        tag=tag + "_m")
    nisa.core_barrier(s2, (0, 1))
    nisa.core_barrier(x2, (0, 1))
    nisa.core_barrier(post_m, (0, 1))
    mo = moe_sublayer_decode(x2, ln2, router_pm, router_bias, base, gate_up, gate_up_scale, down, down_scale, sh_pm,
                             sh_down, ranks=ranks, top_k=top_k, scaling=scaling, limit=limit, eps=eps, pre=pre["moe"],
                             tag=tag)
    nisa.core_barrier(mo, (0, 1))
    return s2, mo, post_m, cache
