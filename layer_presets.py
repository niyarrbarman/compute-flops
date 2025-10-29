"""
Layer preset configurations for each dataset, with optional per-model overrides.

Two supported shapes (both allowed for backward compatibility):
1) Dataset-scoped (legacy):
    LAYER_PRESETS[dataset] = { 'best': [...], 'bsba': [...] }

2) Model-scoped (preferred):
    MODEL_LAYER_PRESETS[model_key][dataset] = { 'best': [...], 'bsba': [...] }

Where model_key is a short identifier, e.g., 'llama-3.1-8b', 'qwen-2.5-7b', 'lucie-7b', 'mistral-7b'.
Indices are 1-based to match tables.
"""

from __future__ import annotations

from typing import Dict, List, Optional


DATASETS = [
    "ARC-Easy",
    "ARC-Challenge",
    "BoolQ",
    "MMLU",
    "CommonQA",
    "Winogrande",
    "BIG-Bench",
    "GSM8K-Hard",
    "MATH500",
]


LAYER_PRESETS: Dict[str, Dict[str, List[int]]] = {
    # Filled from the user's provided code snippets
    "ARC-Easy": {
        "best": [18, 19, 20, 28, 31],
        "bsba": [18, 19, 20, 21, 24, 26, 28, 31],
    },
    "ARC-Challenge": {
        "best": [18, 19, 22, 26],
        "bsba": [18, 19, 20, 22, 24, 26, 27],
    },
    "BoolQ": {
        "best": [18, 19, 20, 28, 31],
        "bsba": [18, 19, 20, 21, 24, 26, 28, 31],
    },

    # The remaining datasets don't have explicit layer sets in the provided context.
    # Leave empty lists by default; you can populate them later.
    "MMLU": {"best": [], "bsba": []},
    "CommonQA": {"best": [], "bsba": []},
    "Winogrande": {"best": [], "bsba": []},
    "BIG-Bench": {"best": [], "bsba": []},
    "GSM8K-Hard": {"best": [], "bsba": []},
    "MATH500": {"best": [], "bsba": []},
}

# ------------------------
# New: model-specific sets
# ------------------------

# If you added per-model presets, place them here under a short key.
# Example keys and alias substrings are defined below in MODEL_ALIASES.
MODEL_LAYER_PRESETS: Dict[str, Dict[str, Dict[str, List[int]]]] = {
    # Table 8 — LLaMA 3.1 8B (0-indexed)
    "llama-3.1-8b": {
        "ARC-Easy":   {"best": [18, 19, 20, 28, 31],
                       "bsba": [18, 19, 20, 21, 24, 26, 28, 31]},
        "ARC-Challenge": {"best": [18, 19, 22, 26],
                          "bsba": [18, 19, 20, 22, 24, 26, 27]},
        "BoolQ":      {"best": [20, 22, 27],
                       "bsba": [17, 20, 21, 26, 27, 31]},
        "MMLU":       {"best": [20],
                       "bsba": [18, 20, 21, 23, 24, 25, 26, 27, 30]},
        "CommonQA":   {"best": [18, 22, 27],
                       "bsba": [18, 21, 22, 25, 26, 27]},
        "Winogrande": {"best": [22, 23, 25, 31],
                       "bsba": [19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 31]},
        "BIG-Bench":  {"best": [13, 19, 21, 27, 28],
                       "bsba": [13, 17, 19, 20, 21, 22, 23, 27, 28, 30, 31]},
        "GSM8K-Hard": {"best": [2],
                       "bsba": [2, 20, 21, 24, 25, 26, 28]},
        "MATH500":    {"best": [2],
                       "bsba": [2, 3, 4]},
    },

    # Table 9 — Qwen 2.5 7B (0-indexed)
    "qwen-2.5-7b": {
        "ARC-Easy":   {"best": [18, 21, 27],
                       "bsba": [5, 18, 21, 23, 25, 26, 27]},
        "ARC-Challenge": {"best": [26, 27],
                          "bsba": [6, 21, 22, 25, 26, 27]},
        "BoolQ":      {"best": [17, 20, 26, 27],
                       "bsba": [11, 18, 20, 21, 25, 26, 27]},
        "MMLU":       {"best": [21, 22, 25, 26, 27],
                       "bsba": [17, 21, 22, 25, 26, 27]},
        "CommonQA":   {"best": [21, 27],
                       "bsba": [5, 20, 21, 22, 26, 27]},
        "Winogrande": {"best": [21, 25, 26],
                       "bsba": [5, 19, 21, 24, 25, 26]},
        "BIG-Bench":  {"best": [9, 18, 22, 24, 25, 26],
                       "bsba": [9, 18, 22, 24, 25, 26]},
        "GSM8K-Hard": {"best": [2, 3],
                       "bsba": [2, 3, 4, 5, 6]},
        "MATH500":    {"best": [2, 3],
                       "bsba": [2, 3, 4, 5]},
    },

    # Table 10 — Lucie 7B (0-indexed)
    "lucie-7b": {
        "ARC-Easy":   {"best": [14, 15, 22, 23, 26, 27],
                       "bsba": [12, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]},
        "ARC-Challenge": {"best": [15, 17, 19, 20, 22, 24, 25],
                          "bsba": [14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 27]},
        "BoolQ":      {"best": [7, 16, 24, 27, 28],
                       "bsba": [4, 7, 10, 11, 12, 13, 14, 15, 16, 18, 19, 22, 24, 25, 26, 27, 28, 30]},
        "MMLU":       {"best": [10, 11, 14, 15, 19, 20, 21, 27],
                       "bsba": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 30]},
        "CommonQA":   {"best": [10, 11, 26],
                       "bsba": [10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]},
        "Winogrande": {"best": [5, 6, 14, 16, 19, 20, 24, 25, 26],
                       "bsba": [5, 6, 12, 14, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28]},
        "BIG-Bench":  {"best": [5, 6, 14, 16, 19, 20, 24, 25, 26],
                       "bsba": [5, 6, 12, 14, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28]},
        "GSM8K-Hard": {"best": [11],
                       "bsba": [11, 20, 22]},
        "MATH500":    {"best": [2, 3],
                       "bsba": [2, 3, 4]},
    },

    # Table 11 — Mistral 7B (0-indexed)
    "mistral-7b": {
        "ARC-Easy":   {"best": [20, 21, 23, 25, 28],
                       "bsba": [20, 21, 22, 23, 24, 25, 28, 29, 31]},
        "ARC-Challenge": {"best": [21, 23, 24, 26, 27, 29],
                          "bsba": [20, 21, 23, 24, 25, 26, 27, 29]},
        "BoolQ":      {"best": [16, 21, 22, 23, 26, 31],
                       "bsba": [11, 16, 20, 22, 23, 24, 26, 27, 31]},
        "MMLU":       {"best": [23, 29],
                       "bsba": [21, 22, 23, 24, 25, 26, 29, 31]},
        "CommonQA":   {"best": [18, 21, 24, 27],
                       "bsba": [18, 20, 21, 23, 24, 27, 31]},
        "Winogrande": {"best": [17, 18, 19, 21, 22, 23, 25, 26, 30, 31],
                       "bsba": [3, 12, 17, 18, 19, 21, 22, 23, 25, 26, 28, 30, 31]},
        "BIG-Bench":  {"best": [2, 4, 14, 21, 22, 23, 25, 26, 27],
                       "bsba": [2, 4, 13, 14, 17, 21, 22, 23, 25, 26, 27]},
        "GSM8K-Hard": {"best": [5, 21],
                       "bsba": [5, 10, 21, 27]},
        "MATH500":    {"best": [2],
                       "bsba": [2, 3, 4, 5]},
    },

    # Placeholder — Qwen 2.5 0.5B (fill in your own layer indices later)
    "qwen-2.5-0.5b": {
        "ARC-Easy":   {"best": [2, 3, 4], 
                       "bsba": [2, 3, 4, 5, 6]},
        "ARC-Challenge": {"best": [2], 
                          "bsba": [2, 3, 4, 5]},
        "BoolQ":      {"best": [2, 3, 4, 5, 6], 
                       "bsba": [2, 3, 4, 5, 6, 7]},
        "MMLU":       {"best": [2, 3], 
                       "bsba": [2, 3, 4, 5, 6]},
        "CommonQA":   {"best": [2, 3], 
                       "bsba": [2, 3, 4]},
        "Winogrande": {"best": [2, 3, 4, 5, 6], 
                       "bsba": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
        "BIG-Bench":  {"best": [2, 3], 
                       "bsba": [2, 3]},
        "GSM8K-Hard": {"best": [2], 
                       "bsba": [2, 3]},
        "MATH500":    {"best": [2], 
                       "bsba": [2, 3]},
    },
}



# Map substrings in the HF model name to a short model preset key.
MODEL_ALIASES: Dict[str, str] = {
    # key: alias to search (lowercased) -> model_key
    "llama-3.1-8b": "llama-3.1-8b",
    "llama-3.1": "llama-3.1-8b",
    "meta-llama/llama-3.1-8b": "llama-3.1-8b",
    "meta-llama": "llama-3.1-8b",

    "qwen2.5-7b": "qwen-2.5-7b",
    "qwen 2.5 7b": "qwen-2.5-7b",
    "qwen": "qwen-2.5-7b",

    # Qwen 2.5 0.5B aliases
    "qwen2.5-0.5b": "qwen-2.5-0.5b",
    "qwen 2.5 0.5b": "qwen-2.5-0.5b",
    "0.5b": "qwen-2.5-0.5b",

    "lucie-7b": "lucie-7b",
    "lucie": "lucie-7b",

    "mistral-7b": "mistral-7b",
    "mistral": "mistral-7b",
}


def resolve_model_key(model_name: str) -> Optional[str]:
    name = (model_name or "").lower()
    for alias, key in MODEL_ALIASES.items():
        if alias in name:
            return key
    return None


def get_layer_config(dataset_name: str, model_key: Optional[str] = None) -> Dict[str, List[int]]:
    """
    Return {'best': [...], 'bsba': [...]} for a dataset (1-based indices).

    Resolution order:
      1) MODEL_LAYER_PRESETS[model_key][dataset]
      2) LAYER_PRESETS[dataset]
      3) default empty lists
    """
    if model_key and model_key in MODEL_LAYER_PRESETS:
        by_model = MODEL_LAYER_PRESETS[model_key]
        if dataset_name in by_model:
            return by_model[dataset_name]
    return LAYER_PRESETS.get(dataset_name, {"best": [], "bsba": []})
