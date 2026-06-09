"""Deterministic evidence-package construction."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .controls import build_negative_controls
from .normalization import add_normalized_activation_columns
from .background import distribution_fraction
from .taxonomy import RANKS, parse_taxonomy


def _safe_float(value) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _records(df: pd.DataFrame, n: int) -> List[dict]:
    cols = [
        "sample_id",
        "otu_id",
        "activation",
        "normalized_activation",
        "fmax",
        "abundance",
        "abundance_rank",
        "split",
        "primary_env",
        "env_first2",
        "technology",
        "project",
        "institution",
        "keywords",
        "taxonomy",
        "taxonomy_source",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
    ]
    keep = [col for col in cols if col in df.columns]
    records = []
    for row in df[keep].head(n).to_dict(orient="records"):
        records.append({key: (_safe_float(val) if isinstance(val, (np.floating, float)) else val) for key, val in row.items()})
    return records


def _value_enrichment(df: pd.DataFrame, column: str, n: int = 10) -> List[dict]:
    if column not in df:
        return []
    total = max(1, len(df))
    rows = []
    for value, count in df[column].fillna("<missing>").astype(str).value_counts().head(n).items():
        rows.append({"value": value, "count": int(count), "fraction": float(count / total)})
    return rows


def _taxonomic_enrichment(df: pd.DataFrame) -> Dict[str, List[dict]]:
    enriched = {}
    for rank in RANKS:
        if rank in df:
            enriched[rank] = _value_enrichment(df, rank)
    return enriched


def _add_taxonomy_columns(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if "taxonomy" not in out:
        return out
    parsed = out["taxonomy"].map(parse_taxonomy)
    for rank in RANKS:
        out[rank] = parsed.map(lambda item, r=rank: item.get(r, "unknown"))
    return out


def _correlation(x, y) -> float:
    if len(x) < 3:
        return 0.0
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    value = x.corr(y)
    return 0.0 if pd.isna(value) else float(value)


def _top_fraction(rows: List[dict]) -> tuple[str, float]:
    if not rows:
        return "", 0.0
    return str(rows[0].get("value", "")), float(rows[0].get("fraction", 0.0))


def _project_batch_artifact(top: pd.DataFrame, background: Dict | None = None) -> Dict[str, object]:
    top_project = _value_enrichment(top, "project", 5)
    top_project_value, max_project_fraction = _top_fraction(top_project)
    background_fraction = distribution_fraction(background, "project", top_project_value)
    enrichment_ratio = float(max_project_fraction / max(background_fraction, 1e-9)) if top_project_value else 0.0
    return {
        "max_project_fraction": float(max_project_fraction),
        "top_project": top_project_value,
        "background_fraction": float(background_fraction),
        "enrichment_ratio": enrichment_ratio,
        "caveat": bool(max_project_fraction >= 0.2),
        "suspected": bool(max_project_fraction >= 0.3),
        "strong": bool(max_project_fraction >= 0.5),
        "top_projects": top_project,
    }


def _technology_bias(top: pd.DataFrame, background: Dict | None = None) -> Dict[str, object]:
    top_technology = _value_enrichment(top, "technology", 5)
    top_value, max_fraction = _top_fraction(top_technology)
    background_fraction = distribution_fraction(background, "technology", top_value)
    enrichment_ratio = float(max_fraction / max(background_fraction, 1e-9)) if top_value else 0.0
    return {
        "top_technology": top_value,
        "max_technology_fraction": float(max_fraction),
        "background_fraction": float(background_fraction),
        "enrichment_ratio": enrichment_ratio,
        "caveat": bool(max_fraction >= 0.8),
        "suspected": bool(max_fraction >= 0.8 and enrichment_ratio >= 1.5),
        "top_technologies": top_technology,
    }


def _environment_enrichment(top: pd.DataFrame, background: Dict | None = None) -> Dict[str, object]:
    primary = _value_enrichment(top, "primary_env", 5)
    top_value, max_fraction = _top_fraction(primary)
    background_fraction = distribution_fraction(background, "primary_env", top_value)
    enrichment_ratio = float(max_fraction / max(background_fraction, 1e-9)) if top_value else 0.0
    return {
        "top_environment": top_value,
        "max_environment_fraction": float(max_fraction),
        "background_fraction": float(background_fraction),
        "enrichment_ratio": enrichment_ratio,
        "strong": bool(max_fraction >= 0.7 or enrichment_ratio >= 2.0),
        "top_environments": primary,
    }


def _sample_concentration(events: pd.DataFrame) -> Dict[str, object]:
    top_events = int(len(events))
    unique_samples = int(events["sample_id"].nunique()) if "sample_id" in events else 0
    unique_otus = int(events["otu_id"].nunique()) if "otu_id" in events else 0
    events_per_sample = float(top_events / max(unique_samples, 1))
    return {
        "top_events": top_events,
        "unique_samples": unique_samples,
        "unique_otus": unique_otus,
        "events_per_sample": events_per_sample,
        "caveat": bool(unique_samples <= 100 or events_per_sample >= 20),
        "suspected": bool(unique_samples <= 50 or events_per_sample >= 50),
    }


def _taxonomy_signal(top: pd.DataFrame) -> Dict[str, object]:
    taxonomy = _taxonomic_enrichment(top)
    candidates = []
    rank_order = ["phylum", "class", "order", "family", "genus", "species"]
    for depth, rank in enumerate(rank_order, start=1):
        rows = [
            row
            for row in taxonomy.get(rank, [])
            if str(row.get("value", "")).strip().lower() not in {"", "unknown", "nan", "none", "null"}
        ]
        if rows:
            candidates.append((rank, str(rows[0].get("value", "")), float(rows[0].get("fraction", 0.0)), depth))
    if candidates:
        strong_candidates = [item for item in candidates if item[2] >= 0.5]
        if strong_candidates:
            best_rank, best_value, best_fraction, _ = max(strong_candidates, key=lambda item: (item[3], item[2]))
        else:
            best_rank, best_value, best_fraction, _ = max(candidates, key=lambda item: (item[2], item[3]))
    else:
        rows = [
            row
            for row in taxonomy.get("kingdom", [])
            if str(row.get("value", "")).strip().lower() not in {"", "unknown", "nan", "none", "null"}
        ]
        if rows:
            best_rank = "kingdom"
            best_value = str(rows[0].get("value", ""))
            best_fraction = float(rows[0].get("fraction", 0.0))
        else:
            best_rank = ""
            best_value = ""
            best_fraction = 0.0
    return {
        "best_rank": best_rank,
        "best_value": best_value,
        "best_fraction": float(best_fraction),
        "strong": bool(best_fraction >= 0.8),
        "moderate": bool(best_fraction >= 0.5),
    }


def _legacy_study_batch_artifact(project_diag: Dict, technology_diag: Dict) -> Dict[str, object]:
    return {
        "max_project_fraction": float(project_diag.get("max_project_fraction", 0.0)),
        "max_technology_fraction": float(technology_diag.get("max_technology_fraction", 0.0)),
        "suspected": bool(project_diag.get("suspected", False) or technology_diag.get("suspected", False)),
        "top_projects": project_diag.get("top_projects", []),
        "top_technologies": technology_diag.get("top_technologies", []),
    }


def _co_occurring_taxa(events: pd.DataFrame, top: pd.DataFrame, n: int = 10) -> List[dict]:
    if "sample_id" not in events or "sample_id" not in top:
        return []
    top_samples = set(top["sample_id"].head(max(10, n)).astype(str))
    same_samples = events[events["sample_id"].astype(str).isin(top_samples)]
    column = "genus" if "genus" in same_samples else "taxonomy"
    return _value_enrichment(same_samples, column, n)


def build_evidence_package(
    feature_id: int,
    events: pd.DataFrame,
    max_examples: int = 50,
    lookup_stats: Dict | None = None,
    background: Dict | None = None,
) -> Dict:
    events = add_normalized_activation_columns(_add_taxonomy_columns(events))
    events = events.sort_values("normalized_activation", ascending=False).reset_index(drop=True)
    top = events.head(max_examples).copy()
    positive_otu_ids = set(top["otu_id"].head(max(5, max_examples // 4)))
    positive_sample_ids = set(top["sample_id"].head(max(5, max_examples // 4)))

    top_otus = (
        events.groupby("otu_id")
        .agg(
            max_activation=("activation", "max"),
            mean_activation=("activation", "mean"),
            event_count=("otu_id", "size"),
            taxonomy=("taxonomy", "first"),
        )
        .sort_values(["max_activation", "event_count"], ascending=False)
        .head(max_examples)
        .reset_index()
        .to_dict(orient="records")
    )
    top_samples = (
        events.groupby("sample_id")
        .agg(
            max_activation=("activation", "max"),
            mean_activation=("activation", "mean"),
            event_count=("sample_id", "size"),
            primary_env=("primary_env", "first"),
            project=("project", "first"),
        )
        .sort_values(["max_activation", "event_count"], ascending=False)
        .head(max_examples)
        .reset_index()
        .to_dict(orient="records")
    )

    if lookup_stats:
        global_stats = {
            "num_events": int(lookup_stats.get("count", len(events))),
            "num_positive_events": int(lookup_stats.get("positive_count", 0)),
            "positive_fraction": float(lookup_stats.get("positive_fraction", 0.0)),
            "activation_sum": float(lookup_stats.get("activation_sum", 0.0)),
            "activation_max": float(lookup_stats.get("activation_max", events["activation"].max() if len(events) else 0.0)),
            "activation_mean": float(lookup_stats.get("activation_mean", events["activation"].mean() if len(events) else 0.0)),
            "mean_positive_activation": float(lookup_stats.get("mean_positive_activation", 0.0)),
            "lookup_top_events": int(len(events)),
            "lookup_unique_otus": int(events["otu_id"].nunique()) if "otu_id" in events else 0,
            "lookup_unique_samples": int(events["sample_id"].nunique()) if "sample_id" in events else 0,
        }
    else:
        global_stats = {
            "num_events": int(len(events)),
            "num_unique_otus": int(events["otu_id"].nunique()) if "otu_id" in events else 0,
            "num_unique_samples": int(events["sample_id"].nunique()) if "sample_id" in events else 0,
            "activation_max": float(events["activation"].max()) if len(events) else 0.0,
            "activation_mean": float(events["activation"].mean()) if len(events) else 0.0,
            "activation_p95": float(events["activation"].quantile(0.95)) if len(events) else 0.0,
        }

    project_diag = _project_batch_artifact(top, background)
    technology_diag = _technology_bias(top, background)
    environment_diag = _environment_enrichment(top, background)
    sample_diag = _sample_concentration(events)
    taxonomy_diag = _taxonomy_signal(top)
    package = {
        "feature_id": int(feature_id),
        "activation_source": "dense_encoder_relu",
        "global_stats": global_stats,
        "top_examples": _records(top, max_examples),
        "heldout_positive_examples": _records(events.iloc[max_examples : max_examples + min(max_examples, 50)], max_examples),
        "top_otus": top_otus,
        "top_samples": top_samples,
        "enrichment": {
            "taxonomy": _taxonomic_enrichment(top),
            "co_occurring_taxa": _co_occurring_taxa(events, top),
            "metadata": {
                "primary_env": _value_enrichment(top, "primary_env"),
                "env_first2": _value_enrichment(top, "env_first2"),
                "technology": _value_enrichment(top, "technology"),
                "project": _value_enrichment(top, "project"),
                "institution": _value_enrichment(top, "institution"),
            },
            "abundance_correlation": {
                "activation_vs_abundance": _correlation(events["activation"], events.get("abundance", [])),
                "activation_vs_abundance_rank": _correlation(events["activation"], events.get("abundance_rank", [])),
            },
        },
        "negative_controls": build_negative_controls(events, positive_otu_ids, positive_sample_ids),
        "diagnostics": {
            "study_batch_artifact": _legacy_study_batch_artifact(project_diag, technology_diag),
            "project_batch_artifact": project_diag,
            "technology_bias": technology_diag,
            "sample_concentration": sample_diag,
            "environment_enrichment": environment_diag,
            "taxonomy_signal": taxonomy_diag,
            "biological_signal_strength": {
                "taxonomy_strong": bool(taxonomy_diag.get("strong", False)),
                "environment_strong": bool(environment_diag.get("strong", False)),
                "project_artifact_suspected": bool(project_diag.get("suspected", False)),
                "technology_artifact_suspected": bool(technology_diag.get("suspected", False)),
                "sample_concentration_suspected": bool(sample_diag.get("suspected", False)),
                "interpret_as_artifact_first": bool(
                    project_diag.get("strong", False)
                    or (
                        project_diag.get("suspected", False)
                        and sample_diag.get("suspected", False)
                        and not taxonomy_diag.get("strong", False)
                    )
                    or (
                        technology_diag.get("suspected", False)
                        and not taxonomy_diag.get("moderate", False)
                        and not environment_diag.get("strong", False)
                    )
                ),
            },
            "contextuality": {
                "same_otu_multiple_samples": bool((events.groupby("otu_id")["sample_id"].nunique() > 1).any()),
                "median_within_otu_activation_std": float(
                    events.groupby("otu_id")["activation"].std().dropna().median()
                )
                if "otu_id" in events
                else 0.0,
            },
        },
    }
    return package
