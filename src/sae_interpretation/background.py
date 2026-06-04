"""Background metadata distributions for SAE feature interpretation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm


def build_token_background(
    tokens_path: Path,
    samples_path: Path,
    splits: set[str],
    chunk_size: int = 500000,
    show_progress: bool = True,
) -> Dict:
    """Estimate background metadata fractions over OTU tokens in selected splits."""

    samples = pd.read_csv(samples_path)
    sample_cols = [col for col in ["sample_id", "project", "institution"] if col in samples.columns]
    sample_info = samples[sample_cols].drop_duplicates("sample_id")
    counters = {key: Counter() for key in ["primary_env", "env_first2", "technology", "project", "institution"]}
    total_tokens = 0
    unique_samples = set()

    reader = pd.read_csv(tokens_path, chunksize=chunk_size)
    iterator = tqdm(reader, desc="background", unit="chunk", disable=not show_progress)
    for chunk in iterator:
        mask = (chunk["token_type"] == "otu") & (chunk["split"].isin(splits))
        cols = [col for col in ["sample_id", "primary_env", "env_first2", "technology"] if col in chunk.columns]
        selected = chunk.loc[mask, cols].copy()
        if selected.empty:
            continue
        selected = selected.merge(sample_info, on="sample_id", how="left")
        total_tokens += int(len(selected))
        unique_samples.update(selected["sample_id"].dropna().astype(str).unique().tolist())
        for column in counters:
            if column in selected:
                counters[column].update(selected[column].fillna("<missing>").astype(str).tolist())
        iterator.set_postfix(tokens=total_tokens)

    distributions = {}
    for column, counter in counters.items():
        values = []
        for value, count in counter.most_common():
            values.append({"value": value, "count": int(count), "fraction": float(count / max(total_tokens, 1))})
        distributions[column] = values
    return {
        "splits": sorted(splits),
        "token_type": "otu",
        "total_tokens": int(total_tokens),
        "unique_samples": int(len(unique_samples)),
        "distributions": distributions,
    }


def distribution_fraction(background: Dict | None, column: str, value: str) -> float:
    if not background:
        return 0.0
    rows = background.get("distributions", {}).get(column, [])
    for row in rows:
        if str(row.get("value")) == str(value):
            return float(row.get("fraction", 0.0))
    return 0.0
