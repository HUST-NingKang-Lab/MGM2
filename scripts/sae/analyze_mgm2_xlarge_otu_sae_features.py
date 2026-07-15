#!/usr/bin/env python3
"""Analyze MGM2 xlarge OTU BatchTopK SAE feature firing patterns."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mgm2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mgm2-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-mgm2")

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.models.BatchTopKSAE import BatchTopKSAE


TOKEN_TYPE_TO_ID = {"cls": 0, "otu": 1}
SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
PRIMARY_ENV_ORDER = ["animal", "aquatic", "soil", "plant", "<missing>"]
INACTIVE_FEATURE_LABEL = "inactive"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir",
        type=Path,
        default=Path("outputs_sae/mgm2_xlarge_otu_sample500k_batchtopk_grid_d4096"),
    )
    parser.add_argument(
        "--activations",
        type=Path,
        default=Path("data/sae/pretrain_xlarge_sample500k/mgm2_scale_xlarge_last_hidden_otu_activations.h5"),
    )
    parser.add_argument("--samples-csv", type=Path, default=Path("data/sae/pretrain_xlarge_sample500k/samples.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_sae_analysis/otu_sample500k_feature_analysis"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/sae/otu_sample500k_feature_analysis"))
    parser.add_argument("--top-k", type=int, nargs="*", default=[8, 16, 32, 64])
    parser.add_argument("--splits", nargs="*", default=["val"])
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--chunk-rows", type=int, default=262_144)
    parser.add_argument("--max-tokens", type=int, default=0, help="Debug/subsample cap after split filtering; 0 means all selected tokens.")
    parser.add_argument("--index-cache", type=Path, default=None, help="Optional .npy cache for selected OTU token row indices.")
    parser.add_argument("--shard-count", type=int, default=1, help="Split selected indices into this many shards before analysis.")
    parser.add_argument("--shard-index", type=int, default=0, help="Analyze only this shard index, 0-based.")
    parser.add_argument("--stats-dir", type=Path, default=None, help="Directory for raw per-run stats .npz files. Defaults to output_dir/raw_stats.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-method", choices=["auto", "umap", "pca"], default="auto")
    return parser.parse_args()


def _run_top_k(run_dir: Path) -> int | None:
    config_path = run_dir / "config_summary.json"
    if config_path.exists():
        with config_path.open() as handle:
            return int(json.load(handle)["top_k"])
    match = re.search(r"k(\d+)_", run_dir.name)
    return int(match.group(1)) if match else None


def find_runs(grid_dir: Path, top_k_values: Iterable[int]) -> list[dict]:
    wanted = set(int(value) for value in top_k_values)
    runs = []
    for run_dir in sorted(grid_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        top_k = _run_top_k(run_dir)
        if top_k not in wanted:
            continue
        checkpoint = run_dir / "best.pt"
        config = run_dir / "config_summary.json"
        metrics = run_dir / "metrics.csv"
        if not checkpoint.exists() or not config.exists() or not metrics.exists():
            continue
        runs.append({"top_k": top_k, "run": run_dir.name, "run_dir": run_dir, "checkpoint": checkpoint, "config": config})
    runs.sort(key=lambda item: item["top_k"])
    found = {item["top_k"] for item in runs}
    missing = wanted.difference(found)
    if missing:
        raise FileNotFoundError(f"Missing SAE runs for top_k values: {sorted(missing)} under {grid_dir}")
    return runs


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device) -> BatchTopKSAE:
    config = json.loads(config_path.read_text())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    model = BatchTopKSAE(
        input_dim=int(config.get("hidden_size", args.get("hidden_size", 512))),
        dict_size=int(config.get("dict_size", args.get("dict_size", 4096))),
        top_k=int(config.get("top_k", args.get("top_k"))),
        aux_k=int(config.get("aux_k", args.get("aux_k", 512))),
        aux_penalty=float(config.get("aux_penalty", args.get("aux_penalty", 0.0))),
        input_unit_norm=not bool(args.get("no_input_unit_norm", False)),
        dead_feature_threshold=int(config.get("dead_feature_threshold", args.get("dead_feature_threshold", 5))),
        inactive_counter_mode=str(config.get("inactive_counter_mode", args.get("inactive_counter_mode", "forwards"))),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_sample_metadata(samples_csv: Path) -> pd.DataFrame:
    samples = pd.read_csv(samples_csv)
    samples["technology"] = samples["technology"].fillna("").replace("", "<missing>")
    samples["primary_env"] = samples["primary_env"].fillna("<missing>").replace("", "<missing>")
    return samples[["primary_env", "technology", "split"]]


def selected_token_indices(handle: h5py.File, splits: list[str], max_tokens: int, seed: int) -> np.ndarray:
    split_ids = np.array([SPLIT_TO_ID[name] for name in splits], dtype=np.int8)
    split_ds = handle["split"]
    token_type_ds = handle["token_type"]
    selected_chunks = []
    for start in tqdm(range(0, split_ds.shape[0], 1_000_000), desc="index", leave=False):
        end = min(start + 1_000_000, split_ds.shape[0])
        mask = np.isin(split_ds[start:end], split_ids) & (token_type_ds[start:end] == TOKEN_TYPE_TO_ID["otu"])
        if mask.any():
            selected_chunks.append(np.flatnonzero(mask).astype(np.int64) + start)
    if not selected_chunks:
        raise ValueError(f"No OTU tokens found for splits={splits}")
    indices = np.concatenate(selected_chunks)
    if max_tokens and max_tokens < len(indices):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_tokens, replace=False))
    return indices


def load_or_build_indices(handle: h5py.File, splits: list[str], max_tokens: int, seed: int, cache_path: Path | None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        indices = np.load(cache_path)
        if max_tokens and max_tokens < len(indices):
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(indices, size=max_tokens, replace=False))
        return indices
    indices = selected_token_indices(handle, splits, max_tokens=0, seed=seed)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, indices)
    if max_tokens and max_tokens < len(indices):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_tokens, replace=False))
    return indices


def _group_codes(values: pd.Series, preferred_order: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    observed = list(pd.unique(values))
    if preferred_order:
        labels = [label for label in preferred_order if label in observed]
        labels.extend(sorted(label for label in observed if label not in labels))
    else:
        labels = sorted(observed)
    mapping = {label: idx for idx, label in enumerate(labels)}
    codes = values.map(mapping).to_numpy(dtype=np.int16)
    return codes, labels


def analyze_run(
    model: BatchTopKSAE,
    handle: h5py.File,
    indices: np.ndarray,
    sample_primary_codes: np.ndarray,
    primary_labels: list[str],
    sample_tech_codes: np.ndarray,
    tech_labels: list[str],
    batch_size: int,
    device: torch.device,
) -> dict:
    dict_size = model.dict_size
    n_primary = len(primary_labels)
    n_tech = len(tech_labels)
    firing_count = np.zeros(dict_size, dtype=np.int64)
    activation_sum = np.zeros(dict_size, dtype=np.float64)
    activation_max = np.zeros(dict_size, dtype=np.float32)
    primary_token_count = np.zeros(n_primary, dtype=np.int64)
    primary_firing_count = np.zeros((dict_size, n_primary), dtype=np.int64)
    primary_activation_sum = np.zeros((dict_size, n_primary), dtype=np.float64)
    tech_token_count = np.zeros(n_tech, dtype=np.int64)
    tech_firing_count = np.zeros((dict_size, n_tech), dtype=np.int64)
    tech_activation_sum = np.zeros((dict_size, n_tech), dtype=np.float64)

    activations = handle["activations"]
    sample_rows = handle["sample_row"]
    total_tokens = 0

    with torch.inference_mode():
        for offset in tqdm(range(0, len(indices), batch_size), desc=f"k={model.top_k}", leave=False):
            batch_idx = indices[offset : offset + batch_size]
            x_np = activations[batch_idx].astype(np.float32)
            rows = sample_rows[batch_idx]
            primary_codes = sample_primary_codes[rows]
            tech_codes = sample_tech_codes[rows]
            np.add.at(primary_token_count, primary_codes, 1)
            np.add.at(tech_token_count, tech_codes, 1)

            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            feature_acts = model(x, update_inactive=False).feature_acts.detach().cpu().numpy()
            fired_rows, fired_cols = np.nonzero(feature_acts > 0)
            fired_values = feature_acts[fired_rows, fired_cols].astype(np.float64)
            np.add.at(firing_count, fired_cols, 1)
            np.add.at(activation_sum, fired_cols, fired_values)
            np.maximum.at(activation_max, fired_cols, fired_values.astype(np.float32))

            fired_primary = primary_codes[fired_rows]
            fired_tech = tech_codes[fired_rows]
            np.add.at(primary_firing_count, (fired_cols, fired_primary), 1)
            np.add.at(primary_activation_sum, (fired_cols, fired_primary), fired_values)
            np.add.at(tech_firing_count, (fired_cols, fired_tech), 1)
            np.add.at(tech_activation_sum, (fired_cols, fired_tech), fired_values)
            total_tokens += len(batch_idx)

    return {
        "total_tokens": total_tokens,
        "firing_count": firing_count,
        "activation_sum": activation_sum,
        "activation_max": activation_max,
        "primary_token_count": primary_token_count,
        "primary_firing_count": primary_firing_count,
        "primary_activation_sum": primary_activation_sum,
        "primary_labels": primary_labels,
        "tech_token_count": tech_token_count,
        "tech_firing_count": tech_firing_count,
        "tech_activation_sum": tech_activation_sum,
        "tech_labels": tech_labels,
    }


def _empty_stats(dict_size: int, primary_labels: list[str], tech_labels: list[str]) -> dict:
    n_primary = len(primary_labels)
    n_tech = len(tech_labels)
    return {
        "total_tokens": 0,
        "firing_count": np.zeros(dict_size, dtype=np.int64),
        "activation_sum": np.zeros(dict_size, dtype=np.float64),
        "activation_max": np.zeros(dict_size, dtype=np.float32),
        "primary_token_count": np.zeros(n_primary, dtype=np.int64),
        "primary_firing_count": np.zeros((dict_size, n_primary), dtype=np.int64),
        "primary_activation_sum": np.zeros((dict_size, n_primary), dtype=np.float64),
        "primary_labels": primary_labels,
        "tech_token_count": np.zeros(n_tech, dtype=np.int64),
        "tech_firing_count": np.zeros((dict_size, n_tech), dtype=np.int64),
        "tech_activation_sum": np.zeros((dict_size, n_tech), dtype=np.float64),
        "tech_labels": tech_labels,
    }


def _accumulate_sparse_features(stats: dict, feature_acts: np.ndarray, primary_codes: np.ndarray, tech_codes: np.ndarray):
    fired_rows, fired_cols = np.nonzero(feature_acts > 0)
    if fired_cols.size == 0:
        return
    fired_values = feature_acts[fired_rows, fired_cols].astype(np.float64)
    np.add.at(stats["firing_count"], fired_cols, 1)
    np.add.at(stats["activation_sum"], fired_cols, fired_values)
    np.maximum.at(stats["activation_max"], fired_cols, fired_values.astype(np.float32))

    fired_primary = primary_codes[fired_rows]
    fired_tech = tech_codes[fired_rows]
    np.add.at(stats["primary_firing_count"], (fired_cols, fired_primary), 1)
    np.add.at(stats["primary_activation_sum"], (fired_cols, fired_primary), fired_values)
    np.add.at(stats["tech_firing_count"], (fired_cols, fired_tech), 1)
    np.add.at(stats["tech_activation_sum"], (fired_cols, fired_tech), fired_values)


def analyze_runs_shared_io(
    models_by_run: dict[str, BatchTopKSAE],
    handle: h5py.File,
    indices: np.ndarray,
    sample_primary_codes: np.ndarray,
    primary_labels: list[str],
    sample_tech_codes: np.ndarray,
    tech_labels: list[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict]:
    stats_by_run = {
        run_name: _empty_stats(model.dict_size, primary_labels, tech_labels)
        for run_name, model in models_by_run.items()
    }
    activations = handle["activations"]
    sample_rows = handle["sample_row"]
    total_batches = (len(indices) + batch_size - 1) // batch_size

    with torch.inference_mode():
        for offset in tqdm(range(0, len(indices), batch_size), total=total_batches, desc="feature stats", leave=False):
            batch_idx = indices[offset : offset + batch_size]
            x_np = activations[batch_idx].astype(np.float32)
            rows = sample_rows[batch_idx]
            primary_codes = sample_primary_codes[rows]
            tech_codes = sample_tech_codes[rows]
            for stats in stats_by_run.values():
                np.add.at(stats["primary_token_count"], primary_codes, 1)
                np.add.at(stats["tech_token_count"], tech_codes, 1)
                stats["total_tokens"] += len(batch_idx)

            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            for run_name, model in models_by_run.items():
                feature_acts = model(x, update_inactive=False).feature_acts.detach().cpu().numpy()
                _accumulate_sparse_features(stats_by_run[run_name], feature_acts, primary_codes, tech_codes)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    return stats_by_run


def summarize_features(run: dict, stats: dict, model: BatchTopKSAE) -> pd.DataFrame:
    total = float(stats["total_tokens"])
    firing_count = stats["firing_count"]
    activation_sum = stats["activation_sum"]
    firing_frequency = firing_count / total
    mean_nonzero = np.divide(activation_sum, firing_count, out=np.zeros_like(activation_sum), where=firing_count > 0)

    primary_density = stats["primary_firing_count"] / np.maximum(stats["primary_token_count"], 1)[None, :]
    primary_mean = np.divide(
        stats["primary_activation_sum"],
        stats["primary_firing_count"],
        out=np.zeros_like(stats["primary_activation_sum"], dtype=np.float64),
        where=stats["primary_firing_count"] > 0,
    )
    env_argmax = primary_density.argmax(axis=1)
    env_max = primary_density[np.arange(model.dict_size), env_argmax]
    env_second = np.sort(primary_density, axis=1)[:, -2] if primary_density.shape[1] > 1 else np.zeros(model.dict_size)
    preferred_env = np.asarray([stats["primary_labels"][idx] for idx in env_argmax], dtype=object)
    preferred_env[firing_count == 0] = INACTIVE_FEATURE_LABEL

    rows = {
        "feature_id": np.arange(model.dict_size),
        "run": run["run"],
        "top_k": run["top_k"],
        "firing_count": firing_count,
        "firing_frequency": firing_frequency,
        "mean_nonzero_activation": mean_nonzero,
        "activation_sum": activation_sum,
        "activation_max": stats["activation_max"],
        "preferred_primary_env": preferred_env,
        "primary_env_density_max": env_max,
        "primary_env_density_margin": env_max - env_second,
    }
    for idx, label in enumerate(stats["primary_labels"]):
        safe = label.replace("<", "").replace(">", "").replace(";", "_").replace(" ", "_")
        rows[f"density_env_{safe}"] = primary_density[:, idx]
        rows[f"mean_nonzero_env_{safe}"] = primary_mean[:, idx]
    return pd.DataFrame(rows)


def group_summary(run: dict, stats: dict, group_name: str) -> pd.DataFrame:
    if group_name == "primary_env":
        labels = stats["primary_labels"]
        token_count = stats["primary_token_count"]
        firing_count = stats["primary_firing_count"]
        activation_sum = stats["primary_activation_sum"]
    elif group_name == "technology":
        labels = stats["tech_labels"]
        token_count = stats["tech_token_count"]
        firing_count = stats["tech_firing_count"]
        activation_sum = stats["tech_activation_sum"]
    else:
        raise ValueError(group_name)
    total_firing = firing_count.sum(axis=0)
    total_activation = activation_sum.sum(axis=0)
    return pd.DataFrame(
        {
            "run": run["run"],
            "top_k": run["top_k"],
            "group_type": group_name,
            "group": labels,
            "token_count": token_count,
            "total_firing_count": total_firing,
            "token_firing_density": total_firing / np.maximum(token_count, 1),
            "mean_nonzero_activation": np.divide(total_activation, total_firing, out=np.zeros_like(total_activation), where=total_firing > 0),
        }
    )


def save_run_outputs(output_dir: Path, run: dict, feature_df: pd.DataFrame, stats: dict):
    run_dir = output_dir / run["run"]
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(run_dir / "feature_summary.csv", index=False)
    group_rows = [
        group_summary(run, stats, "primary_env"),
        group_summary(run, stats, "technology"),
    ]
    pd.concat(group_rows, ignore_index=True).to_csv(run_dir / "group_summary.csv", index=False)


def save_raw_stats(stats_dir: Path, run: dict, stats: dict):
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        stats_dir / f"{run['run']}.npz",
        total_tokens=np.asarray([stats["total_tokens"]], dtype=np.int64),
        firing_count=stats["firing_count"],
        activation_sum=stats["activation_sum"],
        activation_max=stats["activation_max"],
        primary_token_count=stats["primary_token_count"],
        primary_firing_count=stats["primary_firing_count"],
        primary_activation_sum=stats["primary_activation_sum"],
        primary_labels=np.asarray(stats["primary_labels"], dtype=str),
        tech_token_count=stats["tech_token_count"],
        tech_firing_count=stats["tech_firing_count"],
        tech_activation_sum=stats["tech_activation_sum"],
        tech_labels=np.asarray(stats["tech_labels"], dtype=str),
    )


def _save(fig: plt.Figure, figure_dir: Path, stem: str):
    figure_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(figure_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_composition_pies(plots: list[tuple[pd.Series, str]], figure_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (series, title) in zip(axes.flat, plots):
        series = series[series > 0]
        colors = sns.color_palette("Set2", n_colors=len(series))
        wedges, _, autotexts = ax.pie(
            series.to_numpy(dtype=float),
            labels=None,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3.0 else "",
            startangle=90,
            counterclock=False,
            colors=colors,
            wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
            textprops={"fontsize": 9},
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("equal")
        for text in autotexts:
            text.set_color("#222222")
            text.set_fontsize(8)
        total = float(series.sum())
        legend_labels = [f"{idx}: {int(value):,} ({value / total:.1%})" for idx, value in series.items()]
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            fontsize=8,
        )
    _save(fig, figure_dir, "training_and_analysis_composition")


def plot_training_composition(samples: pd.DataFrame, handle: h5py.File, indices: np.ndarray, figure_dir: Path):
    sample_rows = handle["sample_row"][indices]
    token_meta = samples.iloc[sample_rows]
    sample_env = samples["primary_env"].value_counts().reindex(PRIMARY_ENV_ORDER).dropna()
    token_env = token_meta["primary_env"].value_counts().reindex(PRIMARY_ENV_ORDER).dropna()
    sample_tech = samples["technology"].value_counts()
    token_tech = token_meta["technology"].value_counts()
    plots = [
        (sample_env, "Sample composition by primary environment"),
        (token_env, "Analyzed OTU token composition by primary environment"),
        (sample_tech, "Sample composition by technology"),
        (token_tech, "Analyzed OTU token composition by technology"),
    ]
    _plot_composition_pies(plots, figure_dir)


def plot_feature_frequency(all_features: pd.DataFrame, figure_dir: Path):
    df = all_features.copy()
    df["log10_firing_frequency"] = np.log10(df["firing_frequency"].clip(lower=1e-12))
    df["firing_frequency_plot"] = df["firing_frequency"].clip(lower=1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    sns.histplot(data=df, x="log10_firing_frequency", hue="top_k", bins=55, element="step", stat="density", common_norm=False, ax=axes[0])
    axes[0].set_title("Feature firing frequency distribution", fontweight="bold")
    axes[0].set_xlabel("log10 firing frequency")
    sns.boxplot(data=df, x="top_k", y="firing_frequency_plot", ax=axes[1], showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_title("Feature firing frequency by SAE sparsity", fontweight="bold")
    axes[1].set_xlabel("top k")
    axes[1].set_ylabel("firing frequency")
    for ax in axes:
        ax.grid(color="#DDDDDD", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _save(fig, figure_dir, "feature_firing_frequency_histograms")


def plot_density_by_env(all_features: pd.DataFrame, figure_dir: Path):
    density_cols = [col for col in all_features.columns if col.startswith("density_env_")]
    mean_cols = [col for col in all_features.columns if col.startswith("mean_nonzero_env_")]
    env_labels = [col.replace("density_env_", "") for col in density_cols]
    records = []
    for _, row in all_features.iterrows():
        for label, dcol, mcol in zip(env_labels, density_cols, mean_cols):
            records.append(
                {
                    "top_k": int(row["top_k"]),
                    "feature_id": int(row["feature_id"]),
                    "env": label,
                    "density": float(row[dcol]),
                    "mean_nonzero_activation": float(row[mcol]),
                }
            )
    long_df = pd.DataFrame(records)
    long_df["density_plot"] = long_df["density"].clip(lower=1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    sns.boxplot(data=long_df, x="env", y="density_plot", hue="top_k", showfliers=False, ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_title("Feature firing density by primary environment", fontweight="bold")
    axes[0].set_ylabel("feature firing density")
    sns.boxplot(data=long_df, x="env", y="mean_nonzero_activation", hue="top_k", showfliers=False, ax=axes[1])
    axes[1].set_title("Mean non-zero activation by primary environment", fontweight="bold")
    axes[1].set_ylabel("mean non-zero activation")
    for ax in axes:
        ax.set_xlabel("primary environment")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", color="#DDDDDD", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _save(fig, figure_dir, "feature_density_and_activation_by_primary_env")


def plot_preferred_env_counts(all_features: pd.DataFrame, figure_dir: Path):
    counts = (
        all_features.groupby(["top_k", "preferred_primary_env"])
        .size()
        .reset_index(name="feature_count")
    )
    env_labels = _ordered_env_labels(counts["preferred_primary_env"])
    pivot = (
        counts.pivot(index="top_k", columns="preferred_primary_env", values="feature_count")
        .reindex(columns=env_labels, fill_value=0)
        .fillna(0)
        .sort_index()
    )
    colors = sns.color_palette("Set2", n_colors=len(env_labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    bottom = np.zeros(len(pivot), dtype=np.float64)
    x = np.arange(len(pivot))
    for label, color in zip(env_labels, colors):
        values = pivot[label].to_numpy(dtype=np.float64)
        ax.bar(x, values, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.4)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) for value in pivot.index])
    ax.set_xlabel("top k")
    ax.set_ylabel("features")
    ax.set_title("Preferred primary environment of SAE features", fontweight="bold")
    ax.legend(frameon=False, title="preferred environment", ncol=min(3, len(env_labels)))
    ax.grid(axis="y", color="#DDDDDD", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, figure_dir, "preferred_primary_env_feature_counts")


def feature_embedding(weights: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, str]:
    if method in {"auto", "umap"}:
        try:
            from umap import UMAP

            reducer = UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed)
            return reducer.fit_transform(weights), "umap"
        except Exception as exc:
            if method == "umap":
                raise RuntimeError(f"UMAP failed: {exc}") from exc
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(weights), "pca"


def plot_feature_embeddings(
    runs: list[dict],
    all_features: pd.DataFrame,
    figure_dir: Path,
    method: str,
    seed: int,
    coordinate_csv: Path | None = None,
):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    method_used = {}
    coordinate_frames = []
    for ax, run in zip(axes, runs):
        checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
        weights = checkpoint["model_state_dict"]["W_dec"].float().numpy()
        coords, used = feature_embedding(weights, method, seed)
        method_used[run["run"]] = used
        df = all_features[all_features["top_k"] == run["top_k"]].sort_values("feature_id")
        if coordinate_csv is not None:
            coord_df = df[
                [
                    "run",
                    "top_k",
                    "feature_id",
                    "firing_count",
                    "firing_frequency",
                    "mean_nonzero_activation",
                    "preferred_primary_env",
                    "primary_env_density_max",
                    "primary_env_density_margin",
                ]
            ].copy()
            coord_df["embedding_method"] = used
            coord_df["embedding_x"] = coords[:, 0]
            coord_df["embedding_y"] = coords[:, 1]
            coord_df["decoder_weight_norm"] = np.linalg.norm(weights, axis=1)
            coordinate_frames.append(coord_df)
        color = np.log10(df["firing_frequency"].to_numpy().clip(min=1e-12))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color, s=5, cmap="viridis", alpha=0.85, linewidth=0)
        ax.set_title(f"k={run['top_k']} ({used.upper()})", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.75)
    cbar.set_label("log10 firing frequency")
    _save(fig, figure_dir, "feature_weight_embedding_by_firing_frequency")
    if coordinate_csv is not None:
        coordinate_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(coordinate_frames, ignore_index=True).to_csv(coordinate_csv, index=False)
    return method_used


def _ordered_env_labels(labels: Iterable[str]) -> list[str]:
    observed = list(dict.fromkeys(str(label) for label in labels))
    ordered = [label for label in PRIMARY_ENV_ORDER if label in observed]
    if INACTIVE_FEATURE_LABEL in observed:
        ordered.append(INACTIVE_FEATURE_LABEL)
    ordered.extend(sorted(label for label in observed if label not in ordered))
    return ordered


def plot_feature_embeddings_by_env_specificity(
    runs: list[dict],
    all_features: pd.DataFrame,
    figure_dir: Path,
    method: str,
    seed: int,
):
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 4.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    env_labels = _ordered_env_labels(all_features["preferred_primary_env"])
    palette = dict(zip(env_labels, sns.color_palette("Set2", n_colors=len(env_labels))))
    method_used = {}

    for ax, run in zip(axes, runs):
        checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
        weights = checkpoint["model_state_dict"]["W_dec"].float().numpy()
        coords, used = feature_embedding(weights, method, seed)
        method_used[run["run"]] = used

        df = all_features[all_features["top_k"] == run["top_k"]].sort_values("feature_id").copy()
        margins = df["primary_env_density_margin"].to_numpy(dtype=np.float64)
        if np.nanmax(margins) > 0:
            size = 5.0 + 42.0 * np.sqrt(np.clip(margins / np.nanmax(margins), 0.0, 1.0))
        else:
            size = np.full_like(margins, 6.0, dtype=np.float64)
        colors = df["preferred_primary_env"].map(palette).tolist()
        order = np.argsort(size)
        ax.scatter(coords[order, 0], coords[order, 1], c=np.asarray(colors, dtype=object)[order].tolist(), s=size[order], alpha=0.82, linewidth=0)
        ax.set_title(f"k={run['top_k']} ({used.upper()})", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[label], markersize=7, label=label)
        for label in env_labels
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), frameon=False, title="Preferred primary environment")
    _save(fig, figure_dir, "feature_weight_embedding_by_primary_env_specificity")
    return method_used


def plot_overview(all_features: pd.DataFrame, all_groups: pd.DataFrame, figure_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    summary = (
        all_features.groupby("top_k")
        .agg(
            median_firing_frequency=("firing_frequency", "median"),
            low_frequency_features=("firing_frequency", lambda x: int((x < 1e-4).sum())),
            median_mean_nonzero=("mean_nonzero_activation", "median"),
            median_env_margin=("primary_env_density_margin", "median"),
        )
        .reset_index()
    )
    summary["median_firing_frequency_plot"] = summary["median_firing_frequency"].clip(lower=1e-12)
    sns.lineplot(data=summary, x="top_k", y="median_firing_frequency_plot", marker="o", ax=axes[0, 0])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Median feature firing frequency", fontweight="bold")
    sns.barplot(data=summary, x="top_k", y="low_frequency_features", ax=axes[0, 1], color="#7AA6DC")
    axes[0, 1].set_title("Features with firing frequency < 1e-4", fontweight="bold")
    env_group = all_groups[all_groups["group_type"] == "primary_env"]
    sns.lineplot(data=env_group, x="top_k", y="token_firing_density", hue="group", marker="o", ax=axes[1, 0])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Token firing density by primary environment", fontweight="bold")
    sns.lineplot(data=summary, x="top_k", y="median_env_margin", marker="o", ax=axes[1, 1], color="#D27C2C")
    axes[1, 1].set_title("Median primary-env density margin", fontweight="bold")
    for ax in axes.flat:
        ax.grid(color="#DDDDDD", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    _save(fig, figure_dir, "sae_feature_analysis_overview")


def write_top_feature_tables(all_features: pd.DataFrame, output_dir: Path, top_n: int = 50):
    active = all_features[
        (all_features["firing_count"] > 0)
        & (all_features["preferred_primary_env"] != INACTIVE_FEATURE_LABEL)
    ].copy()
    env_top = (
        active.sort_values(
            ["top_k", "preferred_primary_env", "primary_env_density_margin", "primary_env_density_max", "firing_frequency"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["top_k", "preferred_primary_env"], group_keys=False)
        .head(top_n)
    )
    env_top.to_csv(output_dir / "top_environment_specific_features.csv", index=False)

    firing_top = (
        active.sort_values(["top_k", "firing_frequency", "mean_nonzero_activation"], ascending=[True, False, False])
        .groupby("top_k", group_keys=False)
        .head(top_n)
    )
    firing_top.to_csv(output_dir / "top_firing_features.csv", index=False)
    return env_top, firing_top


def write_analysis_report(
    all_features: pd.DataFrame,
    all_groups: pd.DataFrame,
    run_summary: pd.DataFrame,
    output_dir: Path,
    figure_dir: Path,
):
    lines = [
        "# MGM2 xlarge OTU SAE feature analysis",
        "",
        "## Scope",
        "",
        "- SAE top-k values: " + ", ".join(str(int(value)) for value in sorted(run_summary["top_k"].unique())),
        "- Features per SAE: "
        + ", ".join(
            f"k={int(top_k)}: {int(count)}"
            for top_k, count in all_features.groupby("top_k")["feature_id"].nunique().sort_index().items()
        ),
        "- Tokens analyzed per SAE: "
        + ", ".join(
            f"k={int(row.top_k)}: {int(row.tokens_analyzed):,}"
            for row in run_summary.sort_values("top_k").itertuples()
        ),
        "",
        "## Sparsity summary",
        "",
        "| top_k | active features | inactive features | median firing frequency | features < 1e-4 | median env margin |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for top_k, df in all_features.groupby("top_k", sort=True):
        active = int((df["firing_count"] > 0).sum())
        inactive = int((df["firing_count"] == 0).sum())
        rare = int((df["firing_frequency"] < 1e-4).sum())
        lines.append(
            f"| {int(top_k)} | {active:,} | {inactive:,} | "
            f"{float(df['firing_frequency'].median()):.6g} | {rare:,} | "
            f"{float(df['primary_env_density_margin'].median()):.6g} |"
        )

    lines.extend(
        [
            "",
            "## Preferred primary environment",
            "",
            "| top_k | environment | features | active features | median firing frequency | median env margin |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    preferred = (
        all_features.groupby(["top_k", "preferred_primary_env"], sort=True)
        .agg(
            features=("feature_id", "count"),
            active_features=("firing_count", lambda x: int((x > 0).sum())),
            median_firing_frequency=("firing_frequency", "median"),
            median_env_margin=("primary_env_density_margin", "median"),
        )
        .reset_index()
    )
    for row in preferred.itertuples(index=False):
        lines.append(
            f"| {int(row.top_k)} | {row.preferred_primary_env} | {int(row.features):,} | "
            f"{int(row.active_features):,} | {float(row.median_firing_frequency):.6g} | "
            f"{float(row.median_env_margin):.6g} |"
        )

    lines.extend(
        [
            "",
            "## Primary environment token firing density",
            "",
            "| top_k | primary environment | tokens | token firing density | mean non-zero activation |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    env_groups = all_groups[all_groups["group_type"] == "primary_env"].sort_values(["top_k", "group"])
    for row in env_groups.itertuples(index=False):
        lines.append(
            f"| {int(row.top_k)} | {row.group} | {int(row.token_count):,} | "
            f"{float(row.token_firing_density):.6g} | {float(row.mean_nonzero_activation):.6g} |"
        )

    top_env_path = output_dir / "top_environment_specific_features.csv"
    top_firing_path = output_dir / "top_firing_features.csv"
    if top_env_path.exists() and top_firing_path.exists():
        lines.extend(
            [
                "",
                "## Feature tables",
                "",
                f"- `{top_env_path}`: top environment-specific active features per top-k and preferred primary environment.",
                f"- `{top_firing_path}`: highest firing active features per top-k.",
                f"- `{output_dir / 'feature_embedding_coordinates.csv'}`: per-feature decoder-weight embedding coordinates and summary metrics.",
            ]
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )
    stems = [
        "training_and_analysis_composition",
        "feature_firing_frequency_histograms",
        "feature_density_and_activation_by_primary_env",
        "preferred_primary_env_feature_counts",
        "feature_weight_embedding_by_firing_frequency",
        "feature_weight_embedding_by_primary_env_specificity",
        "sae_feature_analysis_overview",
    ]
    for stem in stems:
        lines.append(f"- `{figure_dir / (stem + '.png')}`")

    report_path = output_dir / "sae_feature_analysis_report.md"
    report_path.write_text("\n".join(lines) + "\n")


def write_feature_analysis_readme(output_dir: Path, figure_dir: Path, run_summary: pd.DataFrame):
    top_k_values = ", ".join(str(int(value)) for value in sorted(run_summary["top_k"].unique()))
    token_text = ", ".join(
        f"k={int(row.top_k)}: {int(row.tokens_analyzed):,}"
        for row in run_summary.sort_values("top_k").itertuples()
    )
    lines = [
        "# MGM2 xlarge OTU SAE feature analysis README",
        "",
        "本目录汇总了 MGM2 xlarge OTU activation 上 BatchTopK SAE 的 feature-level 分析结果。",
        f"分析覆盖 top-k = {top_k_values}，每个 SAE 含 4096 个 dictionary features。",
        f"本次统计的 OTU token 数为：{token_text}。",
        "",
        "## 推荐阅读顺序",
        "",
        "1. 先看 `sae_feature_analysis_report.md`，获得 active/inactive feature、firing frequency 和环境偏好的数值摘要。",
        "2. 再看 `figures/sae/otu_sample500k_feature_analysis/` 下的图，确认不同 k 的稀疏性、覆盖度和环境结构。",
        "3. 最后用 `top_environment_specific_features.csv` 和 `feature_embedding_coordinates.csv` 选择需要深入解释的 features。",
        "",
        "## 图说明",
        "",
        "### `training_and_analysis_composition.png`",
        "",
        "四个饼图分别展示样本和被分析 OTU token 的组成：",
        "",
        "- 样本按 primary environment 的组成。",
        "- 分析 token 按 primary environment 的组成。",
        "- 样本按测序技术的组成。",
        "- 分析 token 按测序技术的组成。",
        "",
        "这张图用于检查 SAE 训练/分析数据是否被某一环境或测序技术严重主导。样本组成和 token 组成可能不同，因为每个样本的有效 OTU 数不同。",
        "",
        "### `feature_firing_frequency_histograms.png`",
        "",
        "展示 k8、k16、k32、k64 的 feature firing frequency 分布。左侧看整体分布，右侧比较每个 k 的 firing frequency boxplot。",
        "",
        "这张图回答 SAE 是否足够稀疏、是否存在大量 dead/rare features。k8 中大量 feature 接近不激活；k32/k64 的覆盖明显更充分，但 k64 更密。",
        "",
        "### `feature_density_and_activation_by_primary_env.png`",
        "",
        "按 primary environment 比较每个 feature 的 firing density 和 mean non-zero activation。",
        "",
        "这张图用于判断 SAE features 是否在不同生态环境中表现出系统性差异。firing density 反映某环境中 feature 被触发的频率，mean non-zero activation 反映触发后的平均强度。",
        "",
        "### `preferred_primary_env_feature_counts.png`",
        "",
        "统计每个 k 下 feature 的 preferred primary environment。完全不 firing 的 feature 标记为 `inactive`。",
        "",
        "这张图用于比较不同 k 的 feature 覆盖度和环境偏好分布。inactive 太多说明字典容量没有被充分使用；某个环境占比过高则提示需要检查数据组成或表示偏差。",
        "",
        "### `feature_weight_embedding_by_firing_frequency.png`",
        "",
        "对 SAE decoder weights 做 UMAP/PCA 二维投影，并按 log10 firing frequency 着色。",
        "",
        "这张图用于观察 dictionary feature 空间中是否存在高频/低频 feature 的结构性聚集。如果高频 features 集中在局部区域，说明字典中某些方向承担了更通用的重构功能。",
        "",
        "### `feature_weight_embedding_by_primary_env_specificity.png`",
        "",
        "同样基于 SAE decoder weights 的二维投影，但颜色表示 preferred primary environment，点大小表示环境特异性 margin。",
        "",
        "这张图对应 Evo2-style feature embedding 的环境语义版本。若相近 features 呈现相似环境偏好，说明 SAE 字典方向可能编码了稳定的生态语义结构。",
        "",
        "### `sae_feature_analysis_overview.png`",
        "",
        "汇总不同 k 的关键指标，包括 median firing frequency、低频 feature 数、按环境的 token firing density，以及 primary-env density margin。",
        "",
        "这张图用于做 k 的整体选择。当前结果中 k32 通常是较平衡的选择：inactive feature 极少，同时没有 k64 那么密。",
        "",
        "## 表说明",
        "",
        "- `run_summary.csv`: 每个 k 的 token 数、feature 数、median firing frequency 和 rare feature 数。",
        "- `all_feature_summary.csv`: 每个 feature 的 firing count、firing frequency、activation 强度、preferred environment 和环境密度。",
        "- `all_group_summary.csv`: 按 primary environment 和 technology 汇总的 token firing density 与 mean non-zero activation。",
        "- `feature_embedding_coordinates.csv`: 每个 feature 的二维 embedding 坐标，可用于自定义散点图和标注 top features。",
        "- `top_environment_specific_features.csv`: 每个 k 和环境中最具环境特异性的 active features。",
        "- `top_firing_features.csv`: 每个 k 中 firing frequency 最高的 active features。",
        "- `verified_feature_analysis_summary.csv`: verifier 通过后的简要汇总。",
        "",
        "## 当前使用建议",
        "",
        "主分析建议优先使用 k32。k8 太稀疏且 inactive feature 过多；k16 可作为更稀疏的对照；k64 覆盖充分但更密，适合作为高覆盖对照。",
        "",
        "测试集没有参与当前主图生成。若要做最终 feature interpretation，建议固定 k32 后再在 test split 上复算 firing 和环境偏好，用于验证 top features 的稳定性。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    runs = find_runs(args.grid_dir, args.top_k)
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")
    stats_dir = args.stats_dir or (args.output_dir / "raw_stats")
    samples = load_sample_metadata(args.samples_csv)
    primary_codes, primary_labels = _group_codes(samples["primary_env"], PRIMARY_ENV_ORDER)
    tech_codes, tech_labels = _group_codes(samples["technology"])
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    with h5py.File(args.activations, "r") as handle:
        indices = load_or_build_indices(handle, args.splits, args.max_tokens, args.seed, args.index_cache)
        total_selected_tokens = int(len(indices))
        if args.shard_count > 1:
            indices = np.array_split(indices, args.shard_count)[args.shard_index]
        plot_training_composition(samples, handle, indices, args.figure_dir)
        models_by_run = {run["run"]: load_model(run["checkpoint"], run["config"], device) for run in runs}
        stats_by_run = analyze_runs_shared_io(
            models_by_run,
            handle,
            indices,
            primary_codes,
            primary_labels,
            tech_codes,
            tech_labels,
            args.batch_size,
            device,
        )
        all_feature_frames = []
        all_group_frames = []
        run_summaries = []
        for run in runs:
            model = models_by_run[run["run"]]
            stats = stats_by_run[run["run"]]
            feature_df = summarize_features(run, stats, model)
            save_raw_stats(stats_dir, run, stats)
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
                    "features": model.dict_size,
                    "median_firing_frequency": float(feature_df["firing_frequency"].median()),
                    "features_below_1e-4": int((feature_df["firing_frequency"] < 1e-4).sum()),
                    "features_below_1e-5": int((feature_df["firing_frequency"] < 1e-5).sum()),
                }
            )
        del models_by_run
        if device.type == "cuda":
            torch.cuda.empty_cache()

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
        "grid_dir": str(args.grid_dir),
        "activations": str(args.activations),
        "samples_csv": str(args.samples_csv),
        "splits": args.splits,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "total_selected_tokens_before_sharding": total_selected_tokens,
        "tokens_analyzed_after_sharding": int(len(indices)),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "index_cache": str(args.index_cache) if args.index_cache else None,
        "stats_dir": str(stats_dir),
        "device": str(device),
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
