#!/usr/bin/env python3
"""UMAP plot for top genus fallback prototypes forced to be in-cluster.

The default taxonomy fallback is a taxon-level mean.  That is useful as a
prototype, but nonlinear UMAP can place mean vectors outside the visible OTU
cloud.  This diagnostic builds an alternative top-genus fallback strategy:

  medoid: use the OTU vector closest to the genus high-dimensional mean.

Because the medoid is an actual member OTU, its UMAP position should lie in the
corresponding genus cluster.  The plot is intended to answer a visualization
question, not to replace the full taxonomy fallback file by default.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fallback_umap_diagnostic import STATUS_COLORS, classify_status, load_data

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def top_genus_paths(data: Dict[str, object], n_genera: int) -> List[Tuple[str, List[int]]]:
    genus_to_otus = data["rank_to_path_to_otus"]["genus"]
    return sorted(genus_to_otus.items(), key=lambda item: -len(item[1]))[:n_genera]


def build_incluster_prototypes(data: Dict[str, object], top_genera, args) -> Tuple[pd.DataFrame, np.ndarray]:
    top_paths = [path for path, _ in top_genera]
    top_names = [path.split(";")[-1] for path in top_paths]
    top_means = np.vstack([data["matched_emb"][otu_indices].mean(axis=0) for _, otu_indices in top_genera])

    nn = NearestNeighbors(n_neighbors=min(args.retrieval_k, len(data["matched_emb"])), metric="cosine", n_jobs=-1)
    nn.fit(data["matched_emb"])

    rows = []
    prototypes = []
    for genus_idx, (path, otu_indices) in enumerate(top_genera):
        genus_name = top_names[genus_idx]
        genus_vectors = data["matched_emb"][otu_indices]
        genus_mean = top_means[genus_idx]
        own_cos_to_members = cosine_similarity(genus_mean.reshape(1, -1), genus_vectors)[0]
        medoid_local_idx = int(np.argmax(own_cos_to_members))
        medoid_global_idx = int(otu_indices[medoid_local_idx])
        prototype = data["matched_emb"][medoid_global_idx]

        prototypes.append(prototype)

        own_cos_to_mean = float(cosine_similarity(prototype.reshape(1, -1), genus_mean.reshape(1, -1))[0, 0])
        cos_to_top_means = cosine_similarity(prototype.reshape(1, -1), top_means)[0]
        order = np.argsort(-cos_to_top_means)
        own_rank = int(np.where(order == genus_idx)[0][0]) + 1
        nearest_idx = int(order[0])
        own_otu_cos = cosine_similarity(prototype.reshape(1, -1), genus_vectors)[0]
        neighbor_idx = nn.kneighbors(prototype.reshape(1, -1), return_distance=False)[0]
        otu_set = set(otu_indices)
        retrieval_precision = sum(int(idx in otu_set) for idx in neighbor_idx) / max(1, len(neighbor_idx))

        status = classify_status(
            own_cos=own_cos_to_mean,
            own_rank=own_rank,
            reliable_cos=args.reliable_cos,
            ambiguous_cos=args.ambiguous_cos,
            ambiguous_rank=args.ambiguous_rank,
        )

        original = data["fb_emb"].get(f"genus|{path}")
        original_cos = np.nan
        original_rank = np.nan
        if original is not None:
            original_cos_to_top = cosine_similarity(original.reshape(1, -1), top_means)[0]
            original_order = np.argsort(-original_cos_to_top)
            original_rank = int(np.where(original_order == genus_idx)[0][0]) + 1
            original_cos = float(original_cos_to_top[genus_idx])

        rows.append(
            {
                "plot_id": genus_idx + 1,
                "genus": genus_name,
                "path": path,
                "n_otus": len(otu_indices),
                "strategy": "medoid",
                "prototype_otu_index": medoid_global_idx,
                "has_fallback": True,
                "status": status,
                "own_cos_to_genus_mean": own_cos_to_mean,
                "own_rank_among_top_genera": own_rank,
                "nearest_top_genus": top_names[nearest_idx],
                "nearest_top_genus_cos": float(cos_to_top_means[nearest_idx]),
                "own_otu_cos_mean": float(np.mean(own_otu_cos)),
                "own_otu_cos_median": float(np.median(own_otu_cos)),
                f"retrieval_precision_at_{args.retrieval_k}": retrieval_precision,
                "original_fallback_own_cos_to_genus_mean": original_cos,
                "original_fallback_own_rank_among_top_genera": original_rank,
            }
        )

    return pd.DataFrame(rows), np.vstack(prototypes).astype(np.float32)


def make_umap(data: Dict[str, object], top_genera, metrics: pd.DataFrame, prototypes: np.ndarray, args):
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("umap-learn is required for this plot") from exc

    rng = np.random.default_rng(args.seed)
    selected_otu_indices = []
    labels = []
    for genus_idx, (_, otu_indices) in enumerate(top_genera):
        n = min(len(otu_indices), max(1, args.n_otus // args.n_genera))
        sampled = rng.choice(otu_indices, size=n, replace=False)
        selected_otu_indices.extend(sampled.tolist())
        labels.extend([genus_idx] * n)

    selected_otu_indices = np.asarray(selected_otu_indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    combined = np.vstack([data["matched_emb"][selected_otu_indices], prototypes])

    logger.info("Running UMAP on %d OTUs + %d medoid fallback prototypes", len(selected_otu_indices), len(prototypes))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    coords = reducer.fit_transform(combined)
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
    for row_idx, row in enumerate(metrics.itertuples()):
        coordinate_rows.append(
            {
                "point_type": "fallback_medoid",
                "plot_id": int(row.plot_id),
                "label": row.genus,
                "otu_index": int(row.prototype_otu_index),
                "x": float(fallback_coords[row_idx, 0]),
                "y": float(fallback_coords[row_idx, 1]),
            }
        )
    return otu_coords, labels, fallback_coords, pd.DataFrame(coordinate_rows)


def plot(data: Dict[str, object], metrics: pd.DataFrame, top_genera, prototypes: np.ndarray, args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    otu_coords, labels, fallback_coords, coordinates = make_umap(data, top_genera, metrics, prototypes, args)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.3], wspace=0.08)
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

    for coord_idx, row in enumerate(metrics.itertuples()):
        genus_idx = int(row.plot_id) - 1
        status_color = STATUS_COLORS.get(row.status, "#333333")
        ax.scatter(
            fallback_coords[coord_idx, 0],
            fallback_coords[coord_idx, 1],
            s=300,
            c=[cmap(genus_idx)],
            edgecolors=status_color,
            linewidth=3.2,
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

    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title(
        "Top 20 genus OTUs + in-cluster medoid fallback prototypes\n"
        "fallback is the OTU closest to each genus high-dimensional mean",
        fontsize=13,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ordered = metrics.sort_values("own_cos_to_genus_mean", na_position="first").copy()
    y = np.arange(len(ordered))
    bar_colors = [STATUS_COLORS.get(status, "#777777") for status in ordered["status"]]
    ax_bar.barh(y, ordered["own_cos_to_genus_mean"].fillna(0.0), color=bar_colors, alpha=0.9)
    labels_right = [
        f"{int(row.plot_id):02d} {row.genus} ({row.status}, r={int(row.own_rank_among_top_genera)})"
        for row in ordered.itertuples()
    ]
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels_right, fontsize=8)
    ax_bar.axvline(args.ambiguous_cos, color="#555", linestyle=":", linewidth=1)
    ax_bar.axvline(args.reliable_cos, color="#222", linestyle="--", linewidth=1)
    ax_bar.set_xlabel("cos(medoid fallback, own genus mean)", fontsize=10)
    ax_bar.set_title("In-Cluster Fallback Diagnostic", fontsize=11, fontweight="bold")
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
        f"Fallback UMAP In-Cluster Strategy (top {args.n_genera} genera, {'standardized' if args.standardize else 'raw'} embedding space)",
        fontsize=15,
        fontweight="bold",
    )

    for fmt in ["png", "pdf"]:
        out = args.output_dir / f"fallback_umap_incluster_medoid.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", out)
    plt.close(fig)

    metrics_path = args.output_dir / "fallback_umap_incluster_medoid_metrics.csv"
    coords_path = args.output_dir / "fallback_umap_incluster_medoid_coordinates.csv"
    metrics.to_csv(metrics_path, index=False)
    coordinates.to_csv(coords_path, index=False)
    logger.info("Saved %s", metrics_path)
    logger.info("Saved %s", coords_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw top-genus UMAP with medoid fallback prototypes that sit inside genus clusters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ntv3", type=Path, default=Path("ntv3_embeddings/otus_99_embeddings.h5"))
    parser.add_argument("--fallback", type=Path, default=Path("ntv3_embeddings/otus_99_taxonomy_fallback.h5"))
    parser.add_argument("--otu_info", type=Path, default=Path("/data6/zhanghaohong/databases/microbeatals/otus.info.tsv"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/fallback/fallback_eval_umap_incluster"))
    parser.add_argument("--n_genera", type=int, default=20)
    parser.add_argument("--n_otus", type=int, default=10000)
    parser.add_argument("--retrieval_k", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reliable_cos", type=float, default=0.2)
    parser.add_argument("--ambiguous_cos", type=float, default=0.1)
    parser.add_argument("--ambiguous_rank", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_data(args)
    top_genera = top_genus_paths(data, args.n_genera)
    metrics, prototypes = build_incluster_prototypes(data, top_genera, args)
    plot(data, metrics, top_genera, prototypes, args)
    summary = metrics["status"].value_counts(dropna=False).to_dict()
    print(f"Done. Status counts: {summary}")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
