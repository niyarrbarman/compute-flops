# FLOPs calculator for LLM pruning experiments

This mini-project computes approximate FLOPs for a decoder-only LLM across multiple datasets under three variants:

- Baseline (no layers dropped)
- Best Model (dataset-specific pruned layers)
- BSBA (dataset-specific broader pruning)

It’s designed for simple scripting — not a CLI. Edit a few constants and run it.

## Run

```
python3 main.py
```

The script prints a compact report and (optionally) saves it to `flops_report.txt`.

## Customize

- Change the model: edit `MODEL_NAME` in `main.py`.
- Change datasets: edit `DATASETS_TO_RUN` (or the presets in `layer_presets.py`).
- Update average prompt token lengths in `seq_presets.py` when you have measurements.
- Edit layer presets per dataset in `layer_presets.py` (1-based indices by default).

## Files

- `flops_core.py` – core FLOPs math and config loader
- `layer_presets.py` – per-dataset layer lists for Best and BSBA; supports per-model presets via `MODEL_LAYER_PRESETS` and automatic model-key resolution
- `seq_presets.py` – average prompt token counts per dataset
- `main.py` – orchestrates runs and prints/saves a report

## Notes

- FLOPs are approximate and follow common accounting (dense matmuls ×2, etc.).
- KV cache reduces attention cost in inference mode; disable via `KV_CACHE=False` in `main.py` to see the upper bound.
- If a dataset’s `best/bsba` lists are empty, its FLOPs equal the baseline.
