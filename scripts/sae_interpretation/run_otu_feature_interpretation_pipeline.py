#!/usr/bin/env python3
"""Run lookup, evidence, and Codex task rendering for OTU SAE feature interpretation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sae_interpretation/otu_xlarge_k64_aux0p03.json"))
    parser.add_argument("--feature-ids", type=str, default=None)
    parser.add_argument("--top-n-features", type=int, default=None)
    parser.add_argument("--all-features", action="store_true", help="Build lookup/evidence/task packets for all SAE features.")
    parser.add_argument("--feature-scores", type=Path, default=None)
    parser.add_argument("--feature-selection-metric", type=str, default=None)
    parser.add_argument("--feature-score-batch-size", type=int, default=None)
    parser.add_argument("--feature-lookup-batch-size", type=int, default=None)
    parser.add_argument("--overwrite-feature-scores", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    return parser.parse_args()


def _run(cmd):
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    selection_modes = [bool(args.feature_ids), bool(args.top_n_features), bool(args.all_features)]
    if sum(selection_modes) != 1:
        raise SystemExit("Provide exactly one of --feature-ids, --top-n-features, or --all-features.")
    base = [sys.executable]
    if args.top_n_features:
        score_cmd = base + ["scripts/sae_interpretation/score_sae_features.py", "--config", str(args.config)]
        if args.feature_scores:
            score_cmd += ["--output-csv", str(args.feature_scores)]
        if args.device:
            score_cmd += ["--device", args.device]
        if args.num_threads is not None:
            score_cmd += ["--num-threads", str(args.num_threads)]
        if args.feature_score_batch_size is not None:
            score_cmd += ["--feature-batch-size", str(args.feature_score_batch_size)]
        if args.max_chunks is not None:
            score_cmd += ["--max-chunks", str(args.max_chunks), "--overwrite"]
        elif args.overwrite_feature_scores:
            score_cmd += ["--overwrite"]
        _run(score_cmd)

    background_cmd = base + ["scripts/sae_interpretation/build_background_metadata.py", "--config", str(args.config)]
    _run(background_cmd)

    lookup_cmd = base + ["scripts/sae_interpretation/build_feature_lookup.py", "--config", str(args.config)]
    if args.feature_ids:
        lookup_cmd += ["--feature-ids", args.feature_ids]
    if args.top_n_features:
        lookup_cmd += ["--top-n-features", str(args.top_n_features)]
    if args.all_features:
        lookup_cmd += ["--all-features"]
    if args.feature_scores:
        lookup_cmd += ["--feature-scores", str(args.feature_scores)]
    if args.feature_selection_metric:
        lookup_cmd += ["--feature-selection-metric", args.feature_selection_metric]
    if args.device:
        lookup_cmd += ["--device", args.device]
    if args.num_threads is not None:
        lookup_cmd += ["--num-threads", str(args.num_threads)]
    if args.feature_lookup_batch_size is not None:
        lookup_cmd += ["--feature-batch-size", str(args.feature_lookup_batch_size)]
    if args.max_chunks is not None:
        lookup_cmd += ["--max-chunks", str(args.max_chunks)]
    _run(lookup_cmd)

    feature_args = []
    if args.feature_ids:
        feature_args = ["--feature-ids", args.feature_ids]
    elif args.top_n_features:
        selected_path = Path("outputs_sae_interpretation/otu_xlarge_k64_aux0p03/feature_lookup/selected_feature_ids.json")
        if args.config:
            # Avoid importing project config here; lookup writes the selected ids next to its output.
            selected_path = Path(json.loads(Path(args.config).read_text())["output_dir"]) / "feature_lookup" / "selected_feature_ids.json"
        selected = json.loads(selected_path.read_text())
        feature_args = ["--feature-ids", ",".join(str(int(feature_id)) for feature_id in selected)]
    _run(base + ["scripts/sae_interpretation/build_evidence_packages.py", "--config", str(args.config)] + feature_args)
    _run(
        base
        + ["scripts/sae_interpretation/prepare_codex_interpretation_tasks.py", "--config", str(args.config)]
        + feature_args
    )


if __name__ == "__main__":
    main()
