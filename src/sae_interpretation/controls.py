"""Negative-control selection for SAE feature interpretation."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from .taxonomy import parse_taxonomy


def _top_records(df: pd.DataFrame, n: int = 10) -> List[dict]:
    cols = [
        "sample_id",
        "otu_id",
        "activation",
        "normalized_activation",
        "abundance",
        "abundance_rank",
        "primary_env",
        "project",
        "taxonomy",
    ]
    keep = [col for col in cols if col in df.columns]
    if "activation" in df:
        df = df.sort_values("activation", ascending=False)
    return df[keep].head(n).to_dict(orient="records")


def _rank_bin(rank) -> str:
    try:
        value = int(rank)
    except Exception:
        return "unknown"
    if value <= 5:
        return "top_5"
    if value <= 20:
        return "top_20"
    if value <= 100:
        return "top_100"
    return "low_rank"


def build_negative_controls(
    events: pd.DataFrame,
    positive_otu_ids: Iterable[str],
    positive_sample_ids: Iterable[str],
    max_controls: int = 10,
) -> Dict[str, List[dict]]:
    """Build same-taxonomy, same-biome, same-rank, and same-study negative controls."""
    positive_otu_ids = set(positive_otu_ids)
    positive_sample_ids = set(positive_sample_ids)
    positives = events[events["otu_id"].isin(positive_otu_ids) | events["sample_id"].isin(positive_sample_ids)]
    negatives = events[~events.index.isin(positives.index)].copy()
    if positives.empty or negatives.empty:
        return {"same_taxonomy": [], "same_biome": [], "same_abundance_rank": [], "same_study": []}

    pos_taxa = set()
    for tax in positives.get("taxonomy", pd.Series([], dtype=str)).dropna():
        parsed = parse_taxonomy(tax)
        pos_taxa.update([parsed.get("genus"), parsed.get("family")])
    pos_taxa.discard("unknown")

    def same_taxonomy(row) -> bool:
        parsed = parse_taxonomy(row.get("taxonomy", ""))
        return parsed.get("genus") in pos_taxa or parsed.get("family") in pos_taxa

    pos_biomes = set(positives.get("primary_env", pd.Series([], dtype=str)).dropna().astype(str))
    pos_projects = set(positives.get("project", pd.Series([], dtype=str)).dropna().astype(str))
    pos_rank_bins = set(positives.get("abundance_rank", pd.Series([], dtype=int)).map(_rank_bin))

    controls = {
        "same_taxonomy": _top_records(negatives[negatives.apply(same_taxonomy, axis=1)], max_controls),
        "same_biome": _top_records(negatives[negatives.get("primary_env", "").astype(str).isin(pos_biomes)], max_controls),
        "same_abundance_rank": _top_records(
            negatives[negatives.get("abundance_rank", pd.Series([], dtype=int)).map(_rank_bin).isin(pos_rank_bins)],
            max_controls,
        ),
        "same_study": _top_records(negatives[negatives.get("project", "").astype(str).isin(pos_projects)], max_controls),
    }
    return controls

