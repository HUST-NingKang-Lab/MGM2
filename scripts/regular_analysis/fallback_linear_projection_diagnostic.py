#!/usr/bin/env python3
"""Linear projection diagnostic plot for taxonomy fallback genus centroids.

This script mirrors ``fallback_umap_diagnostic.py`` for loading and reliability
metrics, but replaces UMAP with PCA or TruncatedSVD.  Because the projection is
linear, projected OTU means and fallback offsets are more interpretable than in
nonlinear UMAP space.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fallback_umap_diagnostic import STATUS_COLORS, build_top_genus_diagnostics, load_data

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_linear_projection(data: Dict[str, object], metrics: pd.DataFrame, args):
    rng = np.random.default_rng(args.seed)
    genus_to_otus = data["rank_to_path_to_otus"]["genus"]
    top_genera = sorted(genus_to_otus.items(), key=lambda item: -len(item[1]))[: args.n_genera]

    selected_otu_indices = []
    labels = []
    for genus_idx, (_, otu_indices) in enumerate(top_genera):
        n = min(len(otu_indices), max(1, args.n_otus // args.n_genera))
        sampled = rng.choice(otu_indices, size=n, replace=False)
        selected_otu_indices.extend(sampled.tolist())
        labels.extend([genus_idx] * n)
    selected_otu_indices = np.asarray(selected_otu_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    fallback_rows = metrics[metrics["has_fallback"].astype(bool)].copy()
    fallback_vectors = np.vstack([data["fb_emb"][f"genus|{row.path}"] for row in fallback_rows.itertuples()])
    combined = np.vstack([data["matched_emb"][selected_otu_indices], fallback_vectors])

    if args.method == "pca":
        projector = PCA(n_components=2, random_state=args.seed)
    else:
        projector = TruncatedSVD(n_components=2, random_state=args.seed)

    logger.info(
        "Running %s on %d OTUs + %d fallback centroids",
        args.method.upper(),
        len(selected_otu_indices),
        len(fallback_rows),
    )
    coords = projector.fit_transform(combined)
    otu_coords = coords[: len(selected_otu_indices)]
    fallback_coords = coords[len(selected_otu_indices) :]

    coordinate_rows = []
    for idx, otu_idx in enumerate(selected_otu_indices):
        coordinate_rows.append(
            {
                "point_type": "otu",
                "plot_id": int(labels[idx]) + 1,
                "label": metrics.iloc[int(labels[idx])]["genus"],
                "otu_index": int(otu_idx),
                "x": float(otu_coords[idx, 0]),
                "y": float(otu_coords[idx, 1]),
            }
        )
    for row_idx, row in enumerate(fallback_rows.itertuples()):
        coordinate_rows.append(
            {
                "point_type": "fallback",
                "plot_id": int(row.plot_id),
                "label": row.genus,
                "otu_index": "",
                "x": float(fallback_coords[row_idx, 0]),
                "y": float(fallback_coords[row_idx, 1]),
            }
        )

    return otu_coords, labels, fallback_coords, fallback_rows, pd.DataFrame(coordinate_rows), projector


def plot_linear_diagnostic(data: Dict[str, object], metrics: pd.DataFrame, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    otu_coords, labels, fallback_coords, fallback_rows, coordinates, projector = make_linear_projection(data, metrics, args)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.25], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    cmap = plt.get_cmap("tab20", args.n_genera)
    for genus_idx in range(args.n_genera):
        mask = labels == genus_idx
        if not mask.any():
            continue
        ax.scatter(
            otu_coords[mask, 0],
            otu_coords[mask, 1],
            s=7,
            c=[cmap(genus_idx)],
            alpha=0.34,
            edgecolors="none",
            rasterized=True,
        )

    for coord_idx, row in enumerate(fallback_rows.itertuples()):
        genus_idx = int(row.plot_id) - 1
        status_color = STATUS_COLORS.get(row.status, "#333333")
        otu_group = otu_coords[labels == genus_idx]
        if len(otu_group):
            mean_xy = otu_group.mean(axis=0)
            ax.plot(
                [fallback_coords[coord_idx, 0], mean_xy[0]],
                [fallback_coords[coord_idx, 1], mean_xy[1]],
                linestyle="--",
                linewidth=1.25,
                color=cmap(genus_idx),
                alpha=0.56,
                zorder=3,
            )
        ax.scatter(
            fallback_coords[coord_idx, 0],
            fallback_coords[coord_idx, 1],
            s=270,
            c=[cmap(genus_idx)],
            edgecolors=status_color,
            linewidth=3.1,
            marker="D",
            zorder=10,
        )
        ax.text(
            fallback_coords[coord_idx, 0],
            fallback_coords[coord_idx, 1],
            str(int(row.plot_id)),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=11,
        )

    evr = getattr(projector, "explained_variance_ratio_", None)
    if evr is not None and len(evr) >= 2:
        ax.set_xlabel(f"{args.method.upper()}-1 ({evr[0] * 100:.1f}% variance)", fontsize=12)
        ax.set_ylabel(f"{args.method.upper()}-2 ({evr[1] * 100:.1f}% variance)", fontsize=12)
    else:
        ax.set_xlabel(f"{args.method.upper()}-1", fontsize=12)
        ax.set_ylabel(f"{args.method.upper()}-2", fontsize=12)
    ax.set_title(
        "Top genus OTUs + taxonomy fallback centroids\n"
        "linear projection; dashed line: fallback -> projected sampled OTU mean",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    metrics_sorted = metrics.sort_values("own_cos_to_genus_mean", na_position="first").copy()
    y = np.arange(len(metrics_sorted))
    bar_colors = [STATUS_COLORS.get(status, "#777777") for status in metrics_sorted["status"]]
    ax_bar.barh(y, metrics_sorted["own_cos_to_genus_mean"].fillna(0.0), color=bar_colors, alpha=0.9)
    labels_right = [
        f"{int(row.plot_id):02d} {row.genus} ({row.status}, r={int(row.own_rank_among_top_genera) if pd.notna(row.own_rank_among_top_genera) else 'NA'})"
        for row in metrics_sorted.itertuples()
    ]
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels_right, fontsize=8)
    ax_bar.axvline(args.ambiguous_cos, color="#555", linestyle=":", linewidth=1)
    ax_bar.axvline(args.reliable_cos, color="#222", linestyle="--", linewidth=1)
    ax_bar.set_xlabel("cos(fallback, own genus mean)", fontsize=10)
    ax_bar.set_title("High-D Cosine Diagnostic", fontsize=11, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="white",
            markerfacecolor="#CCCCCC",
            markeredgecolor=color,
            markeredgewidth=3,
            markersize=10,
            label=status,
        )
        for status, color in STATUS_COLORS.items()
        if status != "missing"
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)

    fig.suptitle(
        f"Fallback {args.method.upper()} Diagnostic (top {args.n_genera} genera, "
        f"{'standardized' if args.standardize else 'raw'} embedding space)",
        fontsize=15,
        fontweight="bold",
    )
    for fmt in ["png", "pdf"]:
        out = args.output_dir / f"fallback_linear_projection.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", out)
    plt.close(fig)

    metrics_path = args.output_dir / "fallback_linear_projection_metrics.csv"
    coords_path = args.output_dir / "fallback_linear_projection_coordinates.csv"
    metrics.to_csv(metrics_path, index=False)
    coordinates.to_csv(coords_path, index=False)
    logger.info("Saved %s", metrics_path)
    logger.info("Saved %s", coords_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a PCA/SVD overlay with high-dimensional diagnostics for taxonomy fallback genus centroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ntv3", type=Path, default=Path("ntv3_embeddings/otus_99_embeddings.h5"))
    parser.add_argument("--fallback", type=Path, default=Path("ntv3_embeddings/otus_99_taxonomy_fallback.h5"))
    parser.add_argument("--otu_info", type=Path, default=Path("/data6/zhanghaohong/databases/microbeatals/otus.info.tsv"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/fallback/fallback_eval_linear_projection"))
    parser.add_argument("--method", choices=["pca", "svd"], default="pca")
    parser.add_argument("--n_genera", type=int, default=20)
    parser.add_argument("--n_otus", type=int, default=10000)
    parser.add_argument("--retrieval_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reliable_cos", type=float, default=0.2)
    parser.add_argument("--ambiguous_cos", type=float, default=0.1)
    parser.add_argument("--ambiguous_rank", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(args)
    metric_rows, _ = build_top_genus_diagnostics(data, args.n_genera, args.retrieval_k, args)
    metrics = pd.DataFrame(metric_rows)
    plot_linear_diagnostic(data, metrics, args)
    summary = metrics["status"].value_counts(dropna=False).to_dict()
    print(f"Done. Status counts: {summary}")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
