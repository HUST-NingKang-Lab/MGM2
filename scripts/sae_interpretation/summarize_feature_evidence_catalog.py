#!/usr/bin/env python3
"""Summarize deterministic SAE feature evidence packages into a catalog table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03_global.json"))
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _get(data: dict, path: str, default=None):
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _top_value(data: dict, path: str) -> tuple[str, float]:
    rows = _get(data, path, []) or []
    if not rows:
        return "", 0.0
    first = rows[0]
    return str(first.get("value", "")), float(first.get("fraction", 0.0))


def _row(path: Path) -> dict:
    evidence = json.loads(path.read_text())
    taxonomy = _get(evidence, "diagnostics.taxonomy_signal", {}) or {}
    environment = _get(evidence, "diagnostics.environment_enrichment", {}) or {}
    project = _get(evidence, "diagnostics.project_batch_artifact", {}) or {}
    technology = _get(evidence, "diagnostics.technology_bias", {}) or {}
    sample = _get(evidence, "diagnostics.sample_concentration", {}) or {}
    biology = _get(evidence, "diagnostics.biological_signal_strength", {}) or {}
    contextuality = _get(evidence, "diagnostics.contextuality", {}) or {}
    global_stats = evidence.get("global_stats", {})
    phylum, phylum_fraction = _top_value(evidence, "enrichment.taxonomy.phylum")
    genus, genus_fraction = _top_value(evidence, "enrichment.taxonomy.genus")
    primary_env, primary_env_fraction = _top_value(evidence, "enrichment.metadata.primary_env")
    return {
        "feature_id": int(evidence["feature_id"]),
        "activation_mode": str(global_stats.get("activation_mode", "")),
        "activation_source": str(global_stats.get("activation_source", "")),
        "activation_token_type": str(global_stats.get("activation_token_type", "")),
        "activation_sum": float(global_stats.get("activation_sum", 0.0)),
        "activation_mean": float(global_stats.get("activation_mean", 0.0)),
        "activation_max": float(global_stats.get("activation_max", 0.0)),
        "positive_fraction": float(global_stats.get("positive_fraction", 0.0)),
        "mean_positive_activation": float(global_stats.get("mean_positive_activation", 0.0)),
        "lookup_top_events": int(global_stats.get("lookup_top_events", 0)),
        "lookup_unique_otus": int(global_stats.get("lookup_unique_otus", 0)),
        "lookup_unique_samples": int(global_stats.get("lookup_unique_samples", 0)),
        "taxonomy_best_rank": taxonomy.get("best_rank", ""),
        "taxonomy_best_value": taxonomy.get("best_value", ""),
        "taxonomy_best_fraction": float(taxonomy.get("best_fraction", 0.0)),
        "taxonomy_strong": bool(taxonomy.get("strong", False)),
        "taxonomy_moderate": bool(taxonomy.get("moderate", False)),
        "top_phylum": phylum,
        "top_phylum_fraction": phylum_fraction,
        "top_genus": genus,
        "top_genus_fraction": genus_fraction,
        "top_primary_env": primary_env or environment.get("top_environment", ""),
        "top_primary_env_fraction": primary_env_fraction or float(environment.get("max_environment_fraction", 0.0)),
        "environment_enrichment_ratio": float(environment.get("enrichment_ratio", 0.0)),
        "environment_strong": bool(environment.get("strong", False)),
        "top_project": project.get("top_project", ""),
        "max_project_fraction": float(project.get("max_project_fraction", 0.0)),
        "project_enrichment_ratio": float(project.get("enrichment_ratio", 0.0)),
        "project_artifact_suspected": bool(project.get("suspected", False)),
        "project_artifact_strong": bool(project.get("strong", False)),
        "top_technology": technology.get("top_technology", ""),
        "max_technology_fraction": float(technology.get("max_technology_fraction", 0.0)),
        "technology_enrichment_ratio": float(technology.get("enrichment_ratio", 0.0)),
        "technology_bias_suspected": bool(technology.get("suspected", False)),
        "technology_caveat": bool(technology.get("caveat", False)),
        "unique_samples_top_events": int(sample.get("unique_samples", 0)),
        "unique_otus_top_events": int(sample.get("unique_otus", 0)),
        "events_per_sample_top_events": float(sample.get("events_per_sample", 0.0)),
        "sample_concentration_suspected": bool(sample.get("suspected", False)),
        "sample_concentration_caveat": bool(sample.get("caveat", False)),
        "interpret_as_artifact_first": bool(biology.get("interpret_as_artifact_first", False)),
        "same_otu_multiple_samples": bool(contextuality.get("same_otu_multiple_samples", False)),
        "median_within_otu_activation_std": float(contextuality.get("median_within_otu_activation_std", 0.0)),
        "activation_vs_abundance": float(_get(evidence, "enrichment.abundance_correlation.activation_vs_abundance", 0.0)),
        "activation_vs_abundance_rank": float(_get(evidence, "enrichment.abundance_correlation.activation_vs_abundance_rank", 0.0)),
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    root = Path(cfg(config, "output_dir"))
    evidence_dir = args.evidence_dir or root / "evidence"
    output_csv = args.output_csv or root / "reports" / "feature_evidence_catalog.csv"
    output_json = args.output_json or root / "reports" / "feature_evidence_catalog.json"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [_row(path) for path in sorted(evidence_dir.glob("feature_*.evidence.json"))]
    table = pd.DataFrame(rows).sort_values("feature_id") if rows else pd.DataFrame()
    table.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(json.dumps({"rows": len(rows), "csv": str(output_csv), "json": str(output_json)}, indent=2))


if __name__ == "__main__":
    main()
