# -*- coding: utf-8 -*-
"""
Core FLOPs computation utilities for decoder-only LLMs.

This module exposes:
- LLMConfig: wrapper to read basic model dims from HuggingFace configs
- compute_llm_flops: total FLOPs for a forward pass (or training) with options
- humanize: pretty-print large numbers
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Iterable

from transformers import AutoConfig


# -----------------------------
#  Utility + Config structure
# -----------------------------
@dataclass
class LLMConfig:
    model_name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int
    tie: bool
    num_kv_heads: int

    @staticmethod
    def from_pretrained(model_name: str) -> "LLMConfig":
        cfg = AutoConfig.from_pretrained(model_name)
        d_model = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
        n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
        n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
        d_ff = getattr(cfg, "intermediate_size", getattr(cfg, "n_inner", 4 * (d_model or 0)))
        vocab_size = getattr(cfg, "vocab_size", None)
        tie = getattr(cfg, "tie_word_embeddings", False)
        num_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)

        if any(v is None for v in [d_model, n_layers, n_heads, d_ff, vocab_size]):
            raise ValueError(f"Cannot infer dimensions for {model_name}")

        return LLMConfig(
            model_name=model_name,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            tie=tie,
            num_kv_heads=num_kv_heads,
        )


def flops_linear(in_dim: int, out_dim: int, tokens: float) -> float:
    return 2 * tokens * in_dim * out_dim


def flops_attn_scores(b: int, s: float, h: int, dh: int) -> float:
    return 2 * b * h * s * s * dh


def flops_attn_weighted_sum(b: int, s: float, h: int, dh: int) -> float:
    return 2 * b * h * s * s * dh


def flops_layernorm(tokens: float, d: int) -> float:
    return 8 * tokens * d


def flops_activation(tokens: float, d: int) -> float:
    # GELU/SiLU/SwiGLU approx
    return 4 * tokens * d


def humanize(n: float) -> str:
    units = ["", "K", "M", "G", "T", "P", "E"]
    for u in units:
        if abs(n) < 1000:
            return f"{n:.3f}{u}"
        n /= 1000
    return f"{n:.3f}Z"


# ----------------------------------------
#  FLOPs estimation for decoder-only block
# ----------------------------------------
def decoder_block_flops(
    cfg: LLMConfig,
    batch: int,
    seq: float,
    kv_cache: bool = False,
    mode: str = "inference",
    gated_ffn: bool = True,
) -> float:
    H = cfg.d_model
    Nh = cfg.n_heads
    Dh = H // Nh
    tokens = batch * seq

    # --- Attention ---
    qkv = 3 * flops_linear(H, H, tokens)
    if kv_cache and mode == "inference":
        # causal triangle with KV cache reuse for past states
        tri = seq * (seq + 1) / 2.0
        scores = 2 * batch * Nh * Dh * tri
        av = 2 * batch * Nh * Dh * tri
    else:
        scores = flops_attn_scores(batch, seq, Nh, Dh)
        av = flops_attn_weighted_sum(batch, seq, Nh, Dh)
    attn_out = flops_linear(H, H, tokens)

    # --- MLP (SwiGLU default) ---
    if gated_ffn:
        up = 2 * flops_linear(H, cfg.d_ff, tokens)  # two projections
        act = flops_activation(tokens, cfg.d_ff)
        gate = tokens * cfg.d_ff
    else:
        up = flops_linear(H, cfg.d_ff, tokens)
        act = flops_activation(tokens, cfg.d_ff)
        gate = 0
    down = flops_linear(cfg.d_ff, H, tokens)

    # --- LayerNorms ---
    ln = 2 * flops_layernorm(tokens, H)

    fwd = qkv + scores + av + attn_out + up + act + gate + down + ln
    if mode == "training":
        fwd *= 3
    return fwd


def compute_llm_flops(
    model_name: str,
    batch: int = 1,
    seq: float = 2048,
    mode: str = "inference",
    kv_cache: bool = True,
    drop_layers: Iterable[int] | None = None,
    gated_ffn: bool = True,
    include_lm_head: bool = True,
    one_based: bool = True,
) -> float:
    """
    Compute approximate FLOPs for a single forward pass through a decoder-only LLM.

    Args:
        model_name: HF model id or local path
        batch: batch size
        seq: sequence length in tokens (float to allow averages)
        mode: "inference" or "training"
        kv_cache: use KV cache optimization for attention
        drop_layers: indices of transformer blocks dropped (structured pruning)
        gated_ffn: whether MLP uses gated variant (e.g., SwiGLU)
        include_lm_head: include final projection to vocab
        one_based: interpret drop_layers as 1-based indices if True
    """
    cfg = LLMConfig.from_pretrained(model_name)
    drop_layers = list(drop_layers or [])
    drop = set([(i - 1) if one_based else i for i in drop_layers])
    kept = cfg.n_layers - len(drop)

    per_block = decoder_block_flops(cfg, batch, seq, kv_cache, mode, gated_ffn)
    total_blocks = kept * per_block
    lm_head = 2 * batch * seq * cfg.d_model * cfg.vocab_size if include_lm_head else 0
    total = total_blocks + lm_head
    return float(total)
