# SPDX-License-Identifier: Apache-2.0
"""A CPU reference of Hy4's text model, for parity checks of the Neuron graphs.

It follows the checkpoint's reference implementation (transformers' modeling_hy_v4) step for step: the iHC mixers, MLA
expanded per head (not absorbed), the indexer's top-k as an attention mask, the learned sinks, the router and the
clamped SwiGLU experts, with the block-FP8 weights dequantized from the checkpoint. It runs the first ``--layers``
layers (the server's ``num_hidden_layers`` override builds the same truncated model) over one prompt and prints the
greedy continuation and the first step's top logprobs, or writes them as JSON.

    python tools/hy4/reference.py --model /models/Hy4-FP8-text --layers 6 --prompt "..." --new-tokens 8
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

BF16 = torch.bfloat16


class Checkpoint:
    def __init__(self, path):
        self.path = path
        with open(f"{path}/model.safetensors.index.json") as f:
            self.map = json.load(f)["weight_map"]
        self.files = {}

    def get(self, key, dtype=None):
        file = self.map[key]
        if file not in self.files:
            self.files[file] = safe_open(f"{self.path}/{file}", "pt")
        t = self.files[file].get_tensor(key)
        return t if dtype is None else t.to(dtype)

    def has(self, key):
        return key in self.map

    def linear(self, prefix, dtype=BF16):
        """A [out, in] weight, dequantized if it has a block scale."""
        w = self.get(f"{prefix}.weight")
        if self.has(f"{prefix}.weight_scale"):
            s = self.get(f"{prefix}.weight_scale").float()
            o, i = w.shape
            w = (w.float().view(o // 128, 128, i // 128, 128) * s[:, None, :, None]).view(o, i)
        return w.to(dtype)


def rms(x, w, eps):
    x32 = x.float()
    return w * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class Reference:
    def __init__(self, path, layers):
        self.ck = Checkpoint(path)
        with open(f"{path}/config.json") as f:
            self.cfg = json.load(f)
        self.n = layers
        self.cache = {}

    def w(self, key, dtype=BF16):
        if key not in self.cache:
            self.cache[key] = self.ck.get(key, dtype)
        return self.cache[key]

    def lin(self, prefix, dtype=BF16):
        key = ("lin", prefix, dtype)
        if key not in self.cache:
            self.cache[key] = self.ck.linear(prefix, dtype)
        return self.cache[key]

    def hc(self, prefix, streams):
        c = self.cfg
        n = c["hc_mult"]
        fn, base, scale = (self.w(f"{prefix}.{k}", torch.float32) for k in ("hc_fn", "hc_base", "hc_scale"))
        flat = streams.flatten(-2).float()
        mixes = (flat @ fn.t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + c["rms_norm_eps"])
        pre = torch.sigmoid(mixes[..., :n] * scale[0] + base[:n]) + c["hc_eps"]
        post = c["hc_magnitude"] * torch.sigmoid(mixes[..., n:] * scale[1] + base[n:]) + c["hc_eps"]
        return post, (pre[..., None] * streams.float()).sum(-2).to(streams.dtype)

    def attention(self, i, x, cos, sin, prev_topk):
        c = self.cfg
        p = f"model.layers.{i}.self_attn"
        T = x.shape[0]
        H, nope, rd, v = c["num_attention_heads"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        eps = c["rms_norm_eps"]
        halves = torch.cat([torch.arange(0, rd, 2), torch.arange(1, rd, 2)])
        gate = (x @ self.lin(f"{p}.linear_gate").t()).view(T, H, v)
        q_resid = rms(x @ self.lin(f"{p}.q_a_proj").t(), self.w(f"{p}.q_a_layernorm.weight"), eps)
        # the checkpoint's MLA rotates interleaved RoPE pairs (SGLang's rope_interleave default)
        q = (q_resid @ self.lin(f"{p}.q_b_proj").t()).view(T, H, nope + rd)
        q = torch.cat([q[..., :nope], q[..., nope:][..., halves]], dim=-1).transpose(0, 1)
        q_pass, q_rot = q.split([nope, rd], dim=-1)
        ckv = x @ self.lin(f"{p}.kv_a_proj_with_mqa").t()
        kv_pass, k_rot = ckv.split([c["kv_lora_rank"], rd], dim=-1)
        k_rot = k_rot[..., halves]
        k_pass = rms(kv_pass, self.w(f"{p}.kv_a_layernorm.weight"), eps)
        q_rot = rope(q_rot, cos, sin)
        k_rot = rope(k_rot, cos, sin)
        kv = (k_pass @ self.lin(f"{p}.kv_b_proj").t()).view(T, H, nope + v).transpose(0, 1)
        k_nope, values = kv.split([nope, v], dim=-1)
        keys = torch.cat([k_nope, k_rot[None].expand(H, -1, -1)], dim=-1)
        query = torch.cat([q_pass, q_rot], dim=-1)
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        if c["indexer_types"][i] == "full":
            topk = self.indexer(i, x, q_resid, cos, sin, causal)
        else:
            topk = prev_topk
        index_mask = torch.ones(T, T, dtype=torch.bool).scatter(-1, topk.long(), False)
        allowed = causal & ~index_mask
        scale = (nope + rd) ** -0.5
        logits = (query.float() @ keys.float().transpose(-1, -2)) * scale
        logits = logits.masked_fill(~allowed[None], float("-inf"))
        sinks = self.w(f"{p}.learnable_sink_param", torch.float32)[:, None, None].expand(H, T, 1)
        combined = torch.cat([logits, sinks], dim=-1)
        probs = torch.softmax(combined, dim=-1)[..., :-1]
        out = (probs @ values.float()).transpose(0, 1)  # [T, H, v]
        out = (out * torch.sigmoid(gate.float())).to(BF16).reshape(T, H * v)
        return out @ self.lin(f"{p}.o_proj").t(), topk

    def indexer(self, i, x, q_resid, cos, sin, causal):
        c = self.cfg
        p = f"model.layers.{i}.self_attn.indexer"
        T = x.shape[0]
        ih, idim, rd = c["index_n_heads"], c["index_head_dim"], c["qk_rope_head_dim"]
        q = (q_resid @ self.w(f"{p}.wq_b.weight").t()).view(T, ih, idim)
        q_pass, q_rot = q.split([idim - rd, rd], dim=-1)
        k = F.layer_norm((x @ self.w(f"{p}.wk.weight").t()).float(), (idim,), self.w(f"{p}.k_norm.weight",
                         torch.float32), self.w(f"{p}.k_norm.bias", torch.float32), c["rms_norm_eps"]).to(BF16)
        k_pass, k_rot = k.split([idim - rd, rd], dim=-1)
        q_rot = rope(q_rot, cos[:, None], sin[:, None])
        k_rot = rope(k_rot, cos, sin)
        q = torch.cat([q_pass, q_rot], dim=-1)
        k = torch.cat([k_pass, k_rot], dim=-1)
        scores = torch.einsum("shd,td->sht", q.float(), k.float()).relu()
        weights = (x.float() @ self.w(f"{p}.weights_proj.weight", torch.float32).t()) * ih ** -0.5 * idim ** -0.5
        index = torch.einsum("sht,sh->st", scores, weights).masked_fill(~causal, float("-inf"))
        return index.topk(min(c["index_topk"], T), dim=-1).indices

    def mlp(self, prefix, x):
        g = x @ self.lin(f"{prefix}.gate_proj").t()
        u = x @ self.lin(f"{prefix}.up_proj").t()
        return (F.silu(g) * u) @ self.lin(f"{prefix}.down_proj").t()

    def moe(self, i, x):
        c = self.cfg
        p = f"model.layers.{i}.mlp"
        logits = x.float() @ self.w(f"{p}.gate.weight", torch.float32).t()
        scores = logits.sigmoid()
        choice = scores + self.w(f"{p}.gate.e_score_correction_bias", torch.float32)
        idx = choice.topk(c["num_experts_per_tok"], dim=-1).indices
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling_factor"]
        out = torch.zeros_like(x, dtype=torch.float32)
        lim = c["swiglu_limit"]
        for e in idx.unique().tolist():
            tok, kpos = torch.where(idx == e)
            ep = f"{p}.experts.{e}"
            h = x[tok]
            g = (h @ self.ck.linear(f"{ep}.gate_proj").t()).clamp(max=lim)
            u = (h @ self.ck.linear(f"{ep}.up_proj").t()).clamp(min=-lim, max=lim)
            y = (F.silu(g) * u) @ self.ck.linear(f"{ep}.down_proj").t()
            out.index_add_(0, tok, (y * wts[tok, kpos, None]).float())
        out = out.to(BF16)
        return out + self.mlp(f"{p}.shared_experts", x)

    @torch.no_grad()
    def forward(self, ids):
        c = self.cfg
        T = len(ids)
        pos = torch.arange(T)
        rd = c["qk_rope_head_dim"]
        theta = c["rope_parameters"]["rope_theta"]
        inv = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float) / rd))
        f = pos[:, None].float() * inv
        emb = torch.cat([f, f], dim=-1)
        cos, sin = emb.cos().to(BF16), emb.sin().to(BF16)
        x = self.w("model.embed_tokens.weight")[torch.tensor(ids)]
        streams = x[:, None, :].expand(-1, c["hc_mult"], -1).contiguous()
        topk = None
        for i in range(self.n):
            t0 = time.time()
            post, h = self.hc(f"model.layers.{i}.hc_attn_layer.hc_pre", streams)
            h = rms(h, self.w(f"model.layers.{i}.input_layernorm.weight"), c["rms_norm_eps"])
            out, topk = self.attention(i, h, cos, sin, topk)
            streams = (post[..., None] * out.float()[:, None, :] + streams.float()).to(BF16)
            post, h = self.hc(f"model.layers.{i}.hc_mlp_layer.hc_pre", streams)
            h = rms(h, self.w(f"model.layers.{i}.post_attention_layernorm.weight"), c["rms_norm_eps"])
            out = self.mlp(f"model.layers.{i}.mlp", h) if c["mlp_layer_types"][i] == "dense" else self.moe(i, h)
            streams = (post[..., None] * out.float()[:, None, :] + streams.float()).to(BF16)
            print(f"layer {i} {time.time() - t0:.1f}s sum|x|={streams.float().abs().sum().item():.6g} "
                  f"row0={streams[0].float().abs().sum().item():.6g}", flush=True)
        fn = self.w("model.hc_head.hc_head_fn", torch.float32)
        flat = streams.flatten(-2).float()
        mixes = (flat @ fn.t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + c["rms_norm_eps"])
        pre = torch.sigmoid(mixes * self.w("model.hc_head.hc_head_scale", torch.float32)
                            + self.w("model.hc_head.hc_head_base", torch.float32)) + c["hc_eps"]
        h = (pre[..., None] * streams.float()).sum(-2).to(BF16)
        h = rms(h, self.w("model.norm.weight"), c["rms_norm_eps"])
        return h[-1:].float() @ self.w("lm_head.weight", torch.float32).t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-ids", default=None, help="a JSON file of token ids")
    ap.add_argument("--new-tokens", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    if args.prompt_ids:
        ids = json.load(open(args.prompt_ids))
    else:
        from transformers import AutoTokenizer

        ids = AutoTokenizer.from_pretrained(args.model).encode(args.prompt)
    ref = Reference(args.model, args.layers)
    steps = []
    for _ in range(args.new_tokens):
        logits = ref.forward(ids)[0]
        lp = torch.log_softmax(logits, -1)
        top = lp.topk(5)
        steps.append({"token": int(top.indices[0]), "top_ids": top.indices.tolist(), "top_logprobs": top.values.tolist()})
        print(steps[-1], flush=True)
        ids = ids + [steps[-1]["token"]]
    if args.out:
        json.dump(steps, open(args.out, "w"))


if __name__ == "__main__":
    main()
