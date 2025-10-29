# -*- coding: utf-8 -*-
"""
Run FLOPs calculations for multiple datasets for three variants:
  - Baseline (no layers dropped)
  - Best Model (dataset-specific layer set)
  - BSBA (dataset-specific larger layer set)

How to run:
    python3 main.py

Customize model, datasets, sequence lengths, and layer sets in the constants
below or by editing the small preset modules.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from flops_core import compute_llm_flops, humanize
from layer_presets import DATASETS, get_layer_config, resolve_model_key, MODEL_LAYER_PRESETS
from seq_presets import get_seq_for_dataset, DEFAULT_SEQ


# ------------------------------
# Hyperparameters / configuration
# ------------------------------

# Choose between a single model run or running all models that have presets.
RUN_ALL_MODELS = True

# If RUN_ALL_MODELS is False, we'll use MODEL_NAME below.
# HF model id or local path (used when RUN_ALL_MODELS is False)
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# When RUN_ALL_MODELS is True, we look up model IDs by preset key here.
# Edit these to your preferred checkpoints/IDs.
MODEL_ID_BY_KEY = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
    "lucie-7b": "OpenLLM-France/Lucie-7B-Instruct-v1.1",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "qwen-2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}

# Optional: override datasets list if you want a subset
DATASETS_TO_RUN = DATASETS

# FLOPs runtime knobs
MODE = "inference"          # "inference" or "training"
KV_CACHE = True
GATED_FFN = True
INCLUDE_LM_HEAD = True
ONE_BASED_LAYERS = True      # interpret preset layer numbers as 1-based
BS = 1                       # batch size

# Output behavior
SAVE_TO_FILE = True          # set False if you don't want a text file
REPORT_PATH = "flops_report.txt"


def calc_three_variants(
    model_name: str,
    seq: float,
    layers_best: List[int],
    layers_bsba: List[int],
    batch: int = BS,
    mode: str = MODE,
    kv_cache: bool = KV_CACHE,
    gated_ffn: bool = GATED_FFN,
    include_lm_head: bool = INCLUDE_LM_HEAD,
    one_based: bool = ONE_BASED_LAYERS,
) -> Dict[str, float]:
    """Return FLOPs for baseline, best, and bsba variants."""
    baseline = compute_llm_flops(
        model_name=model_name,
        batch=batch,
        seq=seq,
        mode=mode,
        kv_cache=kv_cache,
        drop_layers=[],
        gated_ffn=gated_ffn,
        include_lm_head=include_lm_head,
        one_based=one_based,
    )
    best = compute_llm_flops(
        model_name=model_name,
        batch=batch,
        seq=seq,
        mode=mode,
        kv_cache=kv_cache,
        drop_layers=layers_best,
        gated_ffn=gated_ffn,
        include_lm_head=include_lm_head,
        one_based=one_based,
    )
    bsba = compute_llm_flops(
        model_name=model_name,
        batch=batch,
        seq=seq,
        mode=mode,
        kv_cache=kv_cache,
        drop_layers=layers_bsba,
        gated_ffn=gated_ffn,
        include_lm_head=include_lm_head,
        one_based=one_based,
    )
    return {"baseline": baseline, "best": best, "bsba": bsba}


def format_dataset_report(dataset: str, seq: float, flops: Dict[str, float]) -> str:
    base = flops["baseline"]
    best = flops["best"]
    bsba = flops["bsba"]
    best_saving = 100.0 * (abs(base / best)-1) if base else 0.0
    bsba_saving = 100.0 * (abs(base / bsba)-1) if base else 0.0
    return (
        f"Dataset: {dataset}\n"
        f"  Seq (avg): {seq:.2f} tokens\n"
        f"  Baseline: {humanize(base)} FLOPs ({base:,.0f})\n"
        f"  Best:     {humanize(best)} FLOPs ({best:,.0f})  | saving: {best_saving:.2f}%\n"
        f"  BSBA:     {humanize(bsba)} FLOPs ({bsba:,.0f})  | saving: {bsba_saving:.2f}%\n"
    )


def _run_for_model(model_id: str, model_key: str | None, lines: List[str]) -> None:
    sub_header = (
        "\n" + "-" * 72 + "\n" +
        (f"Model: {model_id}\n" if model_id else "Model: <unknown>\n") +
        (f"Preset key: {model_key}\n" if model_key else "Preset key: <none> (dataset defaults)\n") +
        "-" * 72
    )
    print(sub_header)
    lines.append(sub_header)

    for ds in DATASETS_TO_RUN:
        seq = get_seq_for_dataset(ds, DEFAULT_SEQ)
        if seq == DEFAULT_SEQ:
            print(f"[info] Using default seq={DEFAULT_SEQ} for {ds}. Update seq_presets.py when you have measurements.")

        layer_cfg = get_layer_config(ds, model_key=model_key)
        try:
            results = calc_three_variants(
                model_name=model_id,
                seq=seq,
                layers_best=layer_cfg.get("best", []),
                layers_bsba=layer_cfg.get("bsba", []),
            )
        except Exception as e:
            err = f"[skip] Failed to load config for '{model_id}' while processing {ds}: {e}"
            print(err)
            lines.append(err)
            continue

        block = format_dataset_report(ds, seq, results)
        print(block)
        lines.append(block)


def run() -> str:
    lines: List[str] = []
    header = (
        "FLOPs Report\n"
        f"Mode={MODE}, KV-cache={KV_CACHE}, Gated-FFN={GATED_FFN}, Include LM Head={INCLUDE_LM_HEAD}\n"
        f"Batch={BS}, One-based layers={ONE_BASED_LAYERS}\n"
        f"Datasets: {', '.join(DATASETS_TO_RUN)}\n"
        "=" * 72
    )
    print(header)
    lines.append(header)

    if RUN_ALL_MODELS:
        # Iterate over all model preset keys we have layer sets for
        for model_key in MODEL_LAYER_PRESETS.keys():
            model_id = MODEL_ID_BY_KEY.get(model_key)
            if not model_id:
                note = f"[warn] No HF model ID configured for preset key '{model_key}'. Edit MODEL_ID_BY_KEY in main.py. Skipping."
                print(note)
                lines.append(note)
                continue
            _run_for_model(model_id, model_key, lines)
    else:
        # Single model path retains the previous behavior
        model_key = resolve_model_key(MODEL_NAME)
        _run_for_model(MODEL_NAME, model_key, lines)

    report = "\n".join(lines)
    if SAVE_TO_FILE:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved report to {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()
