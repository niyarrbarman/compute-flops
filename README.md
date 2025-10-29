# FLOPs calculator for LLM pruning experiments

this mini-project computes approximate FLOPs for a decoder-only llm across multiple datasets under three variants:

- baseline (no layers dropped)
- best model (dataset-specific pruned layers)
- bsba (dataset-specific broader pruning)

## run

```
python3 main.py
```

## Notes

- flops are approximate and follow common accounting.
- kv cache reduces attention cost in inference mode; disable via `KV_CACHE=False` in `main.py` to see the upper bound.
- if a dataset’s `best/bsba` lists are empty, its flops equal the baseline.
