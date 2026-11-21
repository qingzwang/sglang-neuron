# SPDX-License-Identifier: Apache-2.0
"""The shape of a Hy4 (``HYV4ForCausalLM``) checkpoint the graphs serve.

Hy4 is a 78-layer MoE transformer with gated DeepSeek sparse attention (MLA with a learned output gate and a learned
attention sink per head, and the DSA lightning indexer that picks 2,048 keys a token, its picks shared by the following
"shared" layers: IndexCache), identity hyper-connections (iHC: ``hc_mult`` residual streams mixed in and out of every
sublayer) and 256 routed experts with a SwiGLU clamp. The served checkpoint is ``tencent/Hy4-FP8``'s text model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

# the graphs compute in bfloat16
DTYPE = torch.bfloat16


@dataclass(frozen=True)
class Hy4Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    routed_scaling_factor: float
    rms_norm_eps: float
    rope_theta: float
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    # per layer: True where the layer runs its own indexer ("full"), False where it reuses the previous full layer's
    # picks ("shared")
    indexer_full: tuple
    # per layer: True for the dense MLP layers, False for the MoE layers
    mlp_dense: tuple
    hc_mult: int
    hc_magnitude: float
    hc_eps: float
    swiglu_limit: float
    tie_word_embeddings: bool = False

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def compact_dim(self) -> int:
        """A cached MLA row: the normalized latent, then the rotated RoPE key."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.q_head_dim ** -0.5

    @property
    def full_layers(self) -> tuple:
        """The layers that run an indexer, in order."""
        return tuple(i for i, full in enumerate(self.indexer_full) if full)

    def index_slot(self, layer: int) -> int:
        """The index-key cache slot of full layer ``layer``."""
        return self.full_layers.index(layer)


def hy4_config(hf_config) -> Hy4Config:
    """The served shape of a Hy4 checkpoint, refusing what the graphs do not compute."""
    n = int(hf_config.num_hidden_layers)
    params = getattr(hf_config, "rope_parameters", None) or {}
    unserved = {
        "a RoPE scaling": (params.get("rope_type") or "default") != "default",
        "a gating type other than elementwise": (getattr(hf_config, "gating_type", "elementwise") or "elementwise")
        != "elementwise",
        "an ungated MLA": not getattr(hf_config, "gated_mla", True),
        "attention without the learnable sink": not getattr(hf_config, "learnable_sink", True),
        "residuals without iHC": not getattr(hf_config, "enable_ihc", True),
        "group-limited routing": int(getattr(hf_config, "n_group", 1) or 1) != 1,
        "unnormalized top-k weights": not getattr(hf_config, "norm_topk_prob", True),
        "attention biases": bool(getattr(hf_config, "attention_bias", False)),
    }
    asked = [name for name, on in unserved.items() if on]
    if asked:
        raise ValueError(f"Hy4: the graphs do not serve {', '.join(asked)}")
    indexer_types = list(getattr(hf_config, "indexer_types", None) or ["full"] * n)[:n]
    mlp_types = list(getattr(hf_config, "mlp_layer_types", None)
                     or ["dense" if i < int(hf_config.first_k_dense_replace) else "sparse" for i in range(n)])[:n]
    if len(indexer_types) != n or len(mlp_types) != n:
        raise ValueError(f"Hy4: indexer_types and mlp_layer_types must name each of the {n} layers")
    if indexer_types[0] != "full":
        raise ValueError("Hy4: layer 0 must run its own indexer (a shared layer reuses an earlier full layer's picks)")
    bad = sorted({t for t in indexer_types if t not in ("full", "shared")} | {t for t in mlp_types
                                                                             if t not in ("dense", "sparse")})
    if bad:
        raise ValueError(f"Hy4: unknown layer kinds {bad}")
    return Hy4Config(
        vocab_size=int(hf_config.vocab_size),
        hidden_size=int(hf_config.hidden_size),
        intermediate_size=int(hf_config.intermediate_size),
        moe_intermediate_size=int(hf_config.moe_intermediate_size),
        num_hidden_layers=n,
        num_attention_heads=int(hf_config.num_attention_heads),
        q_lora_rank=int(hf_config.q_lora_rank),
        kv_lora_rank=int(hf_config.kv_lora_rank),
        qk_nope_head_dim=int(hf_config.qk_nope_head_dim),
        qk_rope_head_dim=int(hf_config.qk_rope_head_dim),
        v_head_dim=int(hf_config.v_head_dim),
        n_routed_experts=int(hf_config.n_routed_experts),
        num_experts_per_tok=int(hf_config.num_experts_per_tok),
        n_shared_experts=int(hf_config.n_shared_experts or 0),
        routed_scaling_factor=float(hf_config.routed_scaling_factor),
        rms_norm_eps=float(hf_config.rms_norm_eps),
        rope_theta=float(params.get("rope_theta", getattr(hf_config, "rope_theta", 10000.0))),
        index_n_heads=int(hf_config.index_n_heads),
        index_head_dim=int(hf_config.index_head_dim),
        index_topk=int(hf_config.index_topk),
        indexer_full=tuple(t == "full" for t in indexer_types),
        mlp_dense=tuple(t == "dense" for t in mlp_types),
        hc_mult=int(hf_config.hc_mult),
        hc_magnitude=float(hf_config.hc_magnitude),
        hc_eps=float(hf_config.hc_eps),
        swiglu_limit=float(getattr(hf_config, "swiglu_limit", 0.0) or 0.0),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
    )
