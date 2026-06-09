#!/usr/bin/env python3
"""Build top activation lookup tables for selected SAE features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.sae_interpretation.activation_lookup import build_lookup_tables
from src.sae_interpretation.config import cfg, load_config
from src.sae_interpretation.data_sources import load_combined_taxonomy_map, load_sample_metadata
from src.sae_interpretation.sae_features import load_sae_from_checkpoint, select_feature_ids


def _read_feature_ids_file(path: Path) -> str:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("feature_ids", data.get("selected_feature_ids", []))
    return ",".join(str(int(feature_id)) for feature_id in data)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--feature-ids", type=str, default=None)
    parser.add_argument("--feature-ids-file", type=Path, default=None)
    parser.add_argument("--top-n-features", type=int, default=None)
    parser.add_argument("--all-features", action="store_true", help="Build lookup tables for every SAE feature.")
    parser.add_argument("--feature-scores", type=Path, default=None)
    parser.add_argument("--feature-selection-metric", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=None, help="Torch CPU threads for dense SAE activation lookup.")
    parser.add_argument("--max-events-per-feature", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=None, help="Number of SAE features to score at once.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Debug/pilot: stop after this many token chunks.")
    return parser.parse_args()


def main():
    args = parse_args()
    selection_modes = [bool(args.feature_ids), bool(args.feature_ids_file), bool(args.top_n_features), bool(args.all_features)]
    if sum(selection_modes) != 1:
        raise SystemExit("Provide exactly one of --feature-ids, --feature-ids-file, --top-n-features, or --all-features.")
    config = load_config(args.config)
    output_dir = args.output_dir or Path(cfg(config, "output_dir")) / "feature_lookup"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(cfg(config, "sae_checkpoint"))
    root = Path(cfg(config, "output_dir"))
    feature_scores = args.feature_scores or Path(cfg(config, "feature_scores", root / "feature_scores.csv"))
    feature_ids_arg = args.feature_ids
    if args.feature_ids_file:
        feature_ids_arg = _read_feature_ids_file(args.feature_ids_file)
    feature_ids = select_feature_ids(
        checkpoint,
        feature_ids_arg,
        args.top_n_features,
        feature_scores_path=feature_scores if args.top_n_features else None,
        feature_selection_metric=args.feature_selection_metric or cfg(config, "feature_selection_metric", "activation_sum"),
        all_features=args.all_features,
    )
    splits = set(cfg(config, "description_splits", ["train", "val"]))
    device = args.device or cfg(config, "device", "cpu")
    num_threads = args.num_threads if args.num_threads is not None else cfg(config, "num_threads", None)
    if num_threads is not None and int(num_threads) > 0:
        torch.set_num_threads(int(num_threads))
        print(f"Using torch CPU threads: {torch.get_num_threads()}")
    model = load_sae_from_checkpoint(checkpoint, device=device)
    samples = load_sample_metadata(Path(cfg(config, "samples")))
    lookup = build_lookup_tables(
        activations_path=Path(cfg(config, "activations")),
        tokens_path=Path(cfg(config, "tokens")),
        samples=samples,
        taxonomy_map={},
        model=model,
        feature_ids=feature_ids,
        splits=splits,
        max_events_per_feature=args.max_events_per_feature or int(cfg(config, "max_events_per_feature", 5000)),
        chunk_size=args.chunk_size or int(cfg(config, "chunk_size", 262144)),
        feature_batch_size=args.feature_batch_size or int(cfg(config, "feature_lookup_batch_size", 256)),
        device=device,
        activation_mode=cfg(config, "activation_mode", "dense"),
        activation_token_type=cfg(config, "activation_token_type", "auto"),
        max_chunks=args.max_chunks,
    )

    wanted_otus = {str(row["otu_id"]) for item in lookup.values() for row in item["events"] if row.get("otu_id")}
    taxonomy_map = load_combined_taxonomy_map(
        allinfo_path=Path(cfg(config, "taxonomy_allinfo")) if cfg(config, "taxonomy_allinfo", None) else None,
        fallback_path=Path(cfg(config, "taxonomy")) if cfg(config, "taxonomy", None) else None,
        wanted_otu_ids=wanted_otus,
    )
    summary = {
        "features": [],
        "selected_feature_ids": [int(feature_id) for feature_id in feature_ids],
        "splits": sorted(splits),
        "activation_source": cfg(config, "activation_source"),
        "activation_mode": cfg(config, "activation_mode", "dense"),
        "activation_token_type": cfg(config, "activation_token_type", "auto"),
        "feature_selection_metric": args.feature_selection_metric or cfg(config, "feature_selection_metric", "activation_sum"),
        "feature_scores": str(feature_scores) if args.top_n_features else None,
    }
    for feature_id, item in lookup.items():
        for row in item["events"]:
            tax = taxonomy_map.get(str(row.get("otu_id")), {})
            row["taxonomy"] = tax.get("taxonomy", "")
            row["taxonomy_source"] = tax.get("taxonomy_source", "")
            row["rep_species"] = tax.get("rep_species", "")
        events = pd.DataFrame(item["events"])
        if events.empty:
            events = pd.DataFrame(
                columns=[
                    "sample_id",
                    "split",
                    "token_type",
                    "otu_id",
                    "abundance",
                    "processed_abundance",
                    "abundance_rank",
                    "activation",
                    "project",
                    "institution",
                    "keywords",
                    "nonzero_otu_count",
                    "taxonomy",
                    "taxonomy_source",
                    "rep_species",
                ]
            )
        events_path = output_dir / f"feature_{feature_id:06d}.events.csv.gz"
        events.to_csv(events_path, index=False)
        feature_summary = {
            "feature_id": int(feature_id),
            "events_path": str(events_path),
            "activation_mode": cfg(config, "activation_mode", "dense"),
            "activation_token_type": cfg(config, "activation_token_type", "auto"),
            "activation_source": cfg(config, "activation_source"),
            "stats": item["stats"],
        }
        (output_dir / f"feature_{feature_id:06d}.lookup_summary.json").write_text(
            json.dumps(feature_summary, indent=2, sort_keys=True)
        )
        summary["features"].append(feature_summary)
    (output_dir / "lookup_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "selected_feature_ids.json").write_text(json.dumps(summary["selected_feature_ids"], indent=2))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
