#!/usr/bin/env python3
"""Refresh taxonomy columns in existing feature lookup event tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.config import cfg, load_config
from src.sae_interpretation.data_sources import load_combined_taxonomy_map


def _parse_feature_ids(value: str | None):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--feature-ids", type=str, default=None)
    parser.add_argument("--lookup-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    lookup_dir = args.lookup_dir or Path(cfg(config, "output_dir")) / "feature_lookup"
    feature_ids = _parse_feature_ids(args.feature_ids)
    paths = []
    if feature_ids is None:
        paths = sorted(lookup_dir.glob("feature_*.events.csv.gz"))
    else:
        paths = [lookup_dir / f"feature_{feature_id:06d}.events.csv.gz" for feature_id in feature_ids]
    wanted = set()
    tables = {}
    for path in paths:
        df = pd.read_csv(path)
        tables[path] = df
        wanted.update(df["otu_id"].dropna().astype(str).unique().tolist())
    taxonomy = load_combined_taxonomy_map(
        allinfo_path=Path(cfg(config, "taxonomy_allinfo")) if cfg(config, "taxonomy_allinfo", None) else None,
        fallback_path=Path(cfg(config, "taxonomy")) if cfg(config, "taxonomy", None) else None,
        wanted_otu_ids=wanted,
    )
    for path, df in tables.items():
        df["taxonomy"] = df["otu_id"].map(lambda otu: taxonomy.get(str(otu), {}).get("taxonomy", ""))
        df["taxonomy_source"] = df["otu_id"].map(lambda otu: taxonomy.get(str(otu), {}).get("taxonomy_source", ""))
        df["rep_species"] = df["otu_id"].map(lambda otu: taxonomy.get(str(otu), {}).get("rep_species", ""))
        df.to_csv(path, index=False)
        print(f"refreshed {path} rows={len(df)} taxonomy_nonempty={(df['taxonomy'].astype(str) != '').sum()}")


if __name__ == "__main__":
    main()
