#!/usr/bin/env python3
"""Evaluate OTU SAE grid checkpoints on the held-out test split and plot grid-style figures."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mgm2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/mgm2-cache")

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


SPLIT_TO_ID = {"train": 0, "val": 1, "test": 2}
TOKEN_TYPE_TO_ID = {"cls": 0, "otu": 1}

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
    parser.add_argument("--output-dir", type=Path, default=Path("figures/sae/otu_sample500k_grid_no_k128_test"))
    parser.add_argument("--split", choices=sorted(SPLIT_TO_ID), default="test")
    parser.add_argument("--token-type", choices=sorted(TOKEN_TYPE_TO_ID), default="otu")
    parser.add_argument("--index-cache", type=Path, default=Path("outputs_sae_analysis/otu_sample500k_feature_analysis_test/selected_indices_test.npy"))
    parser.add_argument("--checkpoint-kinds", nargs="+", choices=["best", "last"], default=["best", "last"])
    parser.add_argument("--summary-checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--exclude-top-k", type=int, nargs="*", default=[128])
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--eval-batches", type=int, default=50, help="Random batches to evaluate. Ignored when --full-scan is set.")
    parser.add_argument("--full-scan", action="store_true", help="Evaluate every selected test token exactly once.")
    parser.add_argument("--max-tokens", type=int, default=0, help="Debug cap on selected tokens before evaluation; 0 means no cap.")
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _run_top_k(run_dir: Path) -> int | None:
    config_path = run_dir / "config_summary.json"
    if config_path.exists():
        return int(json.loads(config_path.read_text())["top_k"])
    match = re.search(r"k(\d+)_", run_dir.name)
    return int(match.group(1)) if match else None


def find_runs(grid_dir: Path, exclude_top_k: set[int]) -> list[dict]:
    runs = []
    for run_dir in sorted(grid_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config_summary.json"
        metrics_path = run_dir / "metrics.csv"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = json.loads(config_path.read_text())
        top_k = _run_top_k(run_dir)
        if top_k is None or top_k in exclude_top_k:
            continue
        runs.append(
            {
                "run": run_dir.name,
                "run_dir": run_dir,
                "top_k": int(top_k),
                "aux_penalty": float(config["aux_penalty"]),
                "dict_size": int(config["dict_size"]),
                "config": config,
                "metrics": pd.read_csv(metrics_path),
            }
        )
    if not runs:
        raise FileNotFoundError(f"No SAE grid runs found under {grid_dir}")
    return sorted(runs, key=lambda item: (item["top_k"], item["aux_penalty"]))


def selected_indices(handle: h5py.File, split: str, token_type: str, cache_path: Path | None, max_tokens: int, seed: int) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        indices = np.load(cache_path)
    else:
        split_id = SPLIT_TO_ID[split]
        token_id = TOKEN_TYPE_TO_ID[token_type]
        split_ds = handle["split"]
        token_ds = handle["token_type"]
        chunks = []
        for start in tqdm(range(0, split_ds.shape[0], 1_000_000), desc="index", leave=False):
            end = min(start + 1_000_000, split_ds.shape[0])
            mask = (split_ds[start:end] == split_id) & (token_ds[start:end] == token_id)
            if mask.any():
                chunks.append(np.flatnonzero(mask).astype(np.int64) + start)
        if not chunks:
            raise ValueError(f"No tokens found for split={split}, token_type={token_type}")
        indices = np.concatenate(chunks)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, indices)
    if max_tokens and max_tokens < len(indices):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_tokens, replace=False))
    return indices


def load_model(run: dict, checkpoint_kind: str, device: torch.device) -> tuple[BatchTopKSAE, dict]:
    checkpoint_path = run["run_dir"] / f"{checkpoint_kind}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    config = run["config"]
    model = BatchTopKSAE(
        input_dim=int(config.get("hidden_size", args.get("hidden_size", 512))),
        dict_size=int(config.get("dict_size", args.get("dict_size", 4096))),
        top_k=int(config.get("top_k", args.get("top_k", run["top_k"]))),
        aux_k=int(config.get("aux_k", args.get("aux_k", 512))),
        aux_penalty=float(config.get("aux_penalty", args.get("aux_penalty", 0.0))),
        input_unit_norm=not bool(args.get("no_input_unit_norm", False)),
        dead_feature_threshold=int(config.get("dead_feature_threshold", args.get("dead_feature_threshold", 5))),
        inactive_counter_mode=str(config.get("inactive_counter_mode", args.get("inactive_counter_mode", "forwards"))),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_model(
    model: BatchTopKSAE,
    activations: h5py.Dataset,
    indices: np.ndarray,
    batch_size: int,
    eval_batches: int,
    full_scan: bool,
    seed: int,
    device: torch.device,
) -> dict:
    rng = np.random.default_rng(seed)
    losses = []
    mses = []
    evs = []
    l0s = []
    total_tokens = 0

    if full_scan:
        batches = [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]
    else:
        replace = len(indices) < batch_size
        batches = []
        for _ in range(eval_batches):
            batch_idx = rng.choice(indices, size=batch_size, replace=replace).astype(np.int64)
            batches.append(np.sort(batch_idx))

    with torch.inference_mode():
        for batch_idx in tqdm(batches, desc=f"k={model.top_k}", leave=False):
            x_np = activations[batch_idx].astype(np.float32)
            x = torch.from_numpy(x_np).to(device)
            out = model(x, update_inactive=False)
            losses.append(float(out.loss.item()))
            mses.append(float(torch.mean((out.sae_out - x.float()).pow(2)).item()))
            evs.append(float(model.explained_variance(x.float(), out.sae_out).item()))
            l0s.append(float((out.feature_acts > 0).sum(dim=-1).float().mean().item()))
            total_tokens += int(len(batch_idx))

    return {
        "test_loss": float(np.mean(losses)),
        "test_mse": float(np.mean(mses)),
        "test_explained_variance": float(np.mean(evs)),
        "test_l0": float(np.mean(l0s)),
        "dead_features": int(model.num_dead_features().item()),
        "tokens_evaluated": int(total_tokens),
        "eval_mode": "full_scan" if full_scan else "sampled_batches",
    }


def _run_label(row: pd.Series) -> str:
    return f"k={int(row['top_k'])}, aux={row['aux_penalty']:.5g}"


def _save(fig: plt.Figure, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_best_summary(summary: pd.DataFrame, output_dir: Path):
    df = summary.copy()
    df["label"] = df.apply(_run_label, axis=1)
    df = df.sort_values("test_loss")
    colors = sns.color_palette("viridis", n_colors=len(df))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    specs = [
        ("test_loss", "Test loss", False),
        ("test_explained_variance", "Test explained variance", True),
        ("test_mse", "Test MSE", False),
        ("dead_features", "Dead features", False),
    ]
    for ax, (col, title, higher_better) in zip(axes.flat, specs):
        ordered = df.sort_values(col, ascending=not higher_better)
        ax.barh(ordered["label"], ordered[col], color=colors, alpha=0.88)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x", color="#DDDDDD", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, value in enumerate(ordered[col]):
            text = f"{int(value)}" if "dead" in col else f"{value:.4f}"
            ax.text(value, i, f" {text}", va="center", fontsize=8)
    fig.suptitle("MGM2 xlarge OTU BatchTopK SAE grid: held-out test metrics", fontsize=15, fontweight="bold")
    _save(fig, output_dir, "otu_sae_grid_best_summary")


def plot_heatmaps(summary: pd.DataFrame, output_dir: Path):
    metrics = [
        ("test_loss", "Test loss", "mako_r", ".4f"),
        ("test_explained_variance", "Test explained variance", "viridis", ".4f"),
        ("dead_features", "Dead features", "rocket_r", ".0f"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, (col, title, cmap, fmt) in zip(axes, metrics):
        table = summary.pivot(index="top_k", columns="aux_penalty", values=col).sort_index(ascending=False)
        sns.heatmap(table, annot=True, fmt=fmt, cmap=cmap, linewidths=0.5, linecolor="white", ax=ax, cbar=True)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("aux penalty")
        ax.set_ylabel("top k")
    fig.suptitle("Held-out test metric heatmaps", fontsize=15, fontweight="bold")
    _save(fig, output_dir, "otu_sae_grid_heatmaps")


def plot_tradeoff(summary: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    scatter = ax.scatter(
        summary["dead_features"],
        summary["test_explained_variance"],
        s=summary["top_k"] * 8,
        c=summary["aux_penalty"],
        cmap="coolwarm",
        edgecolor="#222222",
        linewidth=0.8,
        alpha=0.9,
    )
    for _, row in summary.iterrows():
        ax.text(row["dead_features"] + 12, row["test_explained_variance"], _run_label(row), fontsize=8, va="center")
    ax.set_xlabel("Dead features")
    ax.set_ylabel("Test explained variance")
    ax.set_title("Held-out reconstruction vs feature usage trade-off", fontsize=13, fontweight="bold")
    ax.grid(color="#DDDDDD", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("aux penalty")
    _save(fig, output_dir, "otu_sae_grid_tradeoff")


def plot_curves(all_rows: pd.DataFrame, output_dir: Path):
    df = all_rows.sort_values(["checkpoint_kind", "top_k"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    specs = [
        ("test_loss", "Test loss", "lower is better"),
        ("test_explained_variance", "Test explained variance", "higher is better"),
        ("dead_features", "Dead features", "lower is better"),
    ]
    for ax, (col, title, subtitle) in zip(axes, specs):
        sns.lineplot(data=df, x="top_k", y=col, hue="checkpoint_kind", marker="o", ax=ax)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
        ax.set_xlabel("top k")
        ax.grid(color="#DDDDDD", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Held-out test metrics by top-k checkpoint", fontsize=15, fontweight="bold")
    _save(fig, output_dir, "otu_sae_grid_curves")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    runs = find_runs(args.grid_dir, set(args.exclude_top_k))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with h5py.File(args.activations, "r") as handle:
        indices = selected_indices(handle, args.split, args.token_type, args.index_cache, args.max_tokens, args.seed)
        activations = handle["activations"]
        for run in runs:
            for checkpoint_kind in args.checkpoint_kinds:
                model, checkpoint = load_model(run, checkpoint_kind, device)
                metrics = evaluate_model(
                    model,
                    activations,
                    indices,
                    args.batch_size,
                    args.eval_batches,
                    args.full_scan,
                    args.seed + run["top_k"] * 100 + (0 if checkpoint_kind == "best" else 1),
                    device,
                )
                val_metrics = checkpoint.get("metrics", {})
                rows.append(
                    {
                        "run": run["run"],
                        "top_k": run["top_k"],
                        "aux_penalty": run["aux_penalty"],
                        "dict_size": run["dict_size"],
                        "checkpoint_kind": checkpoint_kind,
                        "checkpoint_step": int(checkpoint.get("step", -1)),
                        "checkpoint_val_loss": float(val_metrics.get("val_loss", np.nan)),
                        "checkpoint_val_explained_variance": float(val_metrics.get("val_explained_variance", np.nan)),
                        **metrics,
                    }
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    all_rows = pd.DataFrame(rows)
    summary = all_rows[all_rows["checkpoint_kind"] == args.summary_checkpoint].copy()
    if summary.empty:
        raise ValueError(f"No rows for summary checkpoint kind: {args.summary_checkpoint}")
    all_rows.to_csv(args.output_dir / "otu_sae_grid_test_all_checkpoints.csv", index=False)
    summary.to_csv(args.output_dir / "otu_sae_grid_summary.csv", index=False)
    metadata = {
        "grid_dir": str(args.grid_dir),
        "activations": str(args.activations),
        "split": args.split,
        "token_type": args.token_type,
        "index_cache": str(args.index_cache),
        "checkpoint_kinds": args.checkpoint_kinds,
        "summary_checkpoint": args.summary_checkpoint,
        "excluded_top_k": args.exclude_top_k,
        "batch_size": args.batch_size,
        "eval_batches": args.eval_batches,
        "full_scan": args.full_scan,
        "max_tokens": args.max_tokens,
        "device": str(device),
        "tokens_available": int(len(np.load(args.index_cache, mmap_mode="r"))) if args.index_cache.exists() else None,
    }
    (args.output_dir / "otu_sae_grid_test_config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    plot_best_summary(summary, args.output_dir)
    plot_heatmaps(summary, args.output_dir)
    plot_tradeoff(summary, args.output_dir)
    plot_curves(all_rows, args.output_dir)
    print(f"Saved held-out test grid summary and figures to {args.output_dir}")
    print(summary.sort_values("top_k").to_string(index=False))


if __name__ == "__main__":
    main()
