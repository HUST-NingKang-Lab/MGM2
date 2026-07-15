#!/usr/bin/env python3
"""Merge sharded MGM2 xlarge OTU SAE feature-analysis raw stats and plot figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mgm2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mgm2-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-mgm2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts.sae.analyze_mgm2_xlarge_otu_sae_features import (
    PRIMARY_ENV_ORDER,
    _save,
    _plot_composition_pies,
    find_runs,
    group_summary,
    plot_density_by_env,
    plot_feature_embeddings,
    plot_feature_embeddings_by_env_specificity,
    plot_feature_frequency,
    plot_overview,
    plot_preferred_env_counts,
    save_run_outputs,
    summarize_features,
    write_analysis_report,
    write_feature_analysis_readme,
    write_top_feature_tables,
)


class _ModelRef:
    def __init__(self, dict_size: int):
        self.dict_size = int(dict_size)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, default=Path("outputs_sae_analysis/otu_sample500k_feature_analysis_shards"))
    parser.add_argument("--grid-dir", type=Path, default=Path("outputs_sae/mgm2_xlarge_otu_sample500k_batchtopk_grid_d4096"))
    parser.add_argument("--samples-csv", type=Path, default=Path("data/sae/pretrain_xlarge_sample500k/samples.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_sae_analysis/otu_sample500k_feature_analysis"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/sae/otu_sample500k_feature_analysis"))
    parser.add_argument("--top-k", type=int, nargs="*", default=[8, 16, 32, 64])
    parser.add_argument("--embedding-method", choices=["auto", "umap", "pca"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def plot_merged_composition(samples_csv: Path, all_groups: pd.DataFrame, figure_dir: Path):
    samples = pd.read_csv(samples_csv)
    samples["technology"] = samples["technology"].fillna("").replace("", "<missing>")
    samples["primary_env"] = samples["primary_env"].fillna("<missing>").replace("", "<missing>")
    first_top_k = int(all_groups["top_k"].min())
    selected_groups = all_groups[all_groups["top_k"] == first_top_k]
    token_env = (
        selected_groups[selected_groups["group_type"] == "primary_env"]
        .set_index("group")["token_count"]
        .reindex(PRIMARY_ENV_ORDER)
        .dropna()
    )
    token_tech = selected_groups[selected_groups["group_type"] == "technology"].set_index("group")["token_count"].sort_values(ascending=False)
    sample_env = samples["primary_env"].value_counts().reindex(PRIMARY_ENV_ORDER).dropna()
    sample_tech = samples["technology"].value_counts()

    plots = [
        (sample_env, "Sample composition by primary environment"),
        (token_env, "Analyzed OTU token composition by primary environment"),
        (sample_tech, "Sample composition by technology"),
        (token_tech, "Analyzed OTU token composition by technology"),
    ]
    _plot_composition_pies(plots, figure_dir)


def _load_npz(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _stats_from_npz(raw: dict) -> dict:
    return {
        "total_tokens": int(raw["total_tokens"][0]),
        "firing_count": raw["firing_count"].astype(np.int64),
        "activation_sum": raw["activation_sum"].astype(np.float64),
        "activation_max": raw["activation_max"].astype(np.float32),
        "primary_token_count": raw["primary_token_count"].astype(np.int64),
        "primary_firing_count": raw["primary_firing_count"].astype(np.int64),
        "primary_activation_sum": raw["primary_activation_sum"].astype(np.float64),
        "primary_labels": [str(value) for value in raw["primary_labels"].tolist()],
        "tech_token_count": raw["tech_token_count"].astype(np.int64),
        "tech_firing_count": raw["tech_firing_count"].astype(np.int64),
        "tech_activation_sum": raw["tech_activation_sum"].astype(np.float64),
        "tech_labels": [str(value) for value in raw["tech_labels"].tolist()],
    }


def _merge_two(left: dict, right: dict) -> dict:
    if left["primary_labels"] != right["primary_labels"]:
        raise ValueError(f"primary label mismatch: {left['primary_labels']} vs {right['primary_labels']}")
    if left["tech_labels"] != right["tech_labels"]:
        raise ValueError(f"technology label mismatch: {left['tech_labels']} vs {right['tech_labels']}")
    return {
        "total_tokens": left["total_tokens"] + right["total_tokens"],
        "firing_count": left["firing_count"] + right["firing_count"],
        "activation_sum": left["activation_sum"] + right["activation_sum"],
        "activation_max": np.maximum(left["activation_max"], right["activation_max"]),
        "primary_token_count": left["primary_token_count"] + right["primary_token_count"],
        "primary_firing_count": left["primary_firing_count"] + right["primary_firing_count"],
        "primary_activation_sum": left["primary_activation_sum"] + right["primary_activation_sum"],
        "primary_labels": left["primary_labels"],
        "tech_token_count": left["tech_token_count"] + right["tech_token_count"],
        "tech_firing_count": left["tech_firing_count"] + right["tech_firing_count"],
        "tech_activation_sum": left["tech_activation_sum"] + right["tech_activation_sum"],
        "tech_labels": left["tech_labels"],
    }


def load_shard_stats(shard_root: Path, runs: list[dict]) -> dict[str, dict]:
    shard_dirs = sorted(path for path in shard_root.glob("shard_*_of_*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found under {shard_root}")
    merged: dict[str, dict] = {}
    missing: list[str] = []
    for run in runs:
        run_name = run["run"]
        stats = None
        for shard_dir in shard_dirs:
            path = shard_dir / "raw_stats" / f"{run_name}.npz"
            if not path.exists():
                missing.append(str(path))
                continue
            loaded = _stats_from_npz(_load_npz(path))
            stats = loaded if stats is None else _merge_two(stats, loaded)
        if stats is not None:
            merged[run_name] = stats
    if missing:
        raise FileNotFoundError("Missing shard raw stats:\n" + "\n".join(missing[:20]) + ("\n..." if len(missing) > 20 else ""))
    return merged


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    runs = find_runs(args.grid_dir, args.top_k)
    stats_by_run = load_shard_stats(args.shard_root, runs)

    all_feature_frames = []
    all_group_frames = []
    run_summaries = []
    for run in runs:
        stats = stats_by_run[run["run"]]
        model_ref = _ModelRef(dict_size=len(stats["firing_count"]))
        feature_df = summarize_features(run, stats, model_ref)
        save_run_outputs(args.output_dir, run, feature_df, stats)
        groups = pd.concat(
            [group_summary(run, stats, "primary_env"), group_summary(run, stats, "technology")],
            ignore_index=True,
        )
        all_feature_frames.append(feature_df)
        all_group_frames.append(groups)
        run_summaries.append(
            {
                "run": run["run"],
                "top_k": run["top_k"],
                "tokens_analyzed": stats["total_tokens"],
                "features": model_ref.dict_size,
                "median_firing_frequency": float(feature_df["firing_frequency"].median()),
                "features_below_1e-4": int((feature_df["firing_frequency"] < 1e-4).sum()),
                "features_below_1e-5": int((feature_df["firing_frequency"] < 1e-5).sum()),
            }
        )

    all_features = pd.concat(all_feature_frames, ignore_index=True)
    all_groups = pd.concat(all_group_frames, ignore_index=True)
    run_summary = pd.DataFrame(run_summaries)
    all_features.to_csv(args.output_dir / "all_feature_summary.csv", index=False)
    all_groups.to_csv(args.output_dir / "all_group_summary.csv", index=False)
    run_summary.to_csv(args.output_dir / "run_summary.csv", index=False)

    plot_feature_frequency(all_features, args.figure_dir)
    plot_density_by_env(all_features, args.figure_dir)
    plot_preferred_env_counts(all_features, args.figure_dir)
    plot_overview(all_features, all_groups, args.figure_dir)
    plot_merged_composition(args.samples_csv, all_groups, args.figure_dir)
    embedding_methods = plot_feature_embeddings(
        runs,
        all_features,
        args.figure_dir,
        args.embedding_method,
        args.seed,
        coordinate_csv=args.output_dir / "feature_embedding_coordinates.csv",
    )
    env_embedding_methods = plot_feature_embeddings_by_env_specificity(
        runs,
        all_features,
        args.figure_dir,
        args.embedding_method,
        args.seed,
    )
    write_top_feature_tables(all_features, args.output_dir)
    write_analysis_report(all_features, all_groups, run_summary, args.output_dir, args.figure_dir)
    write_feature_analysis_readme(args.output_dir, args.figure_dir, run_summary)
    metadata = {
        "source": "merged_shards",
        "shard_root": str(args.shard_root),
        "grid_dir": str(args.grid_dir),
        "top_k": args.top_k,
        "embedding_methods": embedding_methods,
        "env_specificity_embedding_methods": env_embedding_methods,
        "outputs": {
            "output_dir": str(args.output_dir),
            "figure_dir": str(args.figure_dir),
        },
    }
    (args.output_dir / "analysis_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({"run_summary": run_summaries, "figure_dir": str(args.figure_dir), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
