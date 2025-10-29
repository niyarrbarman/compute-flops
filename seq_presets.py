"""
Average prompt token lengths per dataset (estimated or measured).

We include known averages from the provided notebook for some datasets.
For others, we fall back to DEFAULT_SEQ.
Edit values as you collect better measurements.
"""

from __future__ import annotations

from typing import Dict


SEQ_PRESETS: Dict[str, float] = {
    "ARC-Challenge": 133.52,
    "ARC-Easy": 129.53,
    "BoolQ": 68.71,
    "MMLU": 120.47,
    "CommonQA": 78.18,
    "Winogrande": 88.61,
    "BIG-Bench": 58.00,
    "GSM8K-Hard": 143.85,
    "MATH500": 253.54,
}


DEFAULT_SEQ: float = 128.0


def get_seq_for_dataset(dataset_name: str, fallback: float = DEFAULT_SEQ) -> float:
    val = SEQ_PRESETS.get(dataset_name)
    return float(val) if isinstance(val, (int, float)) and val is not None else float(fallback)
