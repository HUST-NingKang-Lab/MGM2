#!/usr/bin/env python3
"""UMAP diagnostic plot for taxonomy fallback genus centroids.

This is a focused variant of eval_taxonomy_fallback_v2.py. It keeps the
top-genera UMAP overlay, but adds high-dimensional cosine diagnostics so the
plot is not interpreted from 2D UMAP geometry alone.

Outputs:
  fallback_umap_diagnostic.{png,pdf}
  fallback_umap_diagnostic_metrics.csv
  fallback_umap_diagnostic_coordinates.csv
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
STATUS_COLORS = {
    "reliable": "#2E7D32",
    "ambiguous": "#EF8A17",
    "bad": "#C62828",
    "missing": "#777777",
}


def decode_strings(values: Iterable[object]) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            text = str(value).strip()
            while (text.startswith("b'") and text.endswith("'")) or (text.startswith('b"') and text.endswith('"')):
                text = text[2:-1]
            decoded.append(text.strip().strip("'").strip('"'))
    return decoded


def load_fallback(path: Path):
    logger.info("Loading fallback embeddings from %s", path)
    with h5py.File(path, "r") as handle:
        taxon_emb = handle["taxon_embeddings"][:].astype(np.float32)
        taxon_keys = decode_strings(handle["taxon_keys"][:])
        taxon_ranks = decode_strings(handle["taxon_ranks"][:])
        taxon_paths = decode_strings(handle["taxon_paths"][:])
        taxon_counts = handle["taxon_counts"][:].astype(int)
    return taxon_emb, taxon_keys, taxon_ranks, taxon_paths, taxon_counts


def load_ntv3(path: Path):
    logger.info("Loading NTv3 embeddings from %s", path)
    with h5py.File(path, "r") as handle:
        embeddings = handle["embeddings"][:].astype(np.float32)
        otu_ids = decode_strings(handle["otu_ids"][:])
    return embeddings, otu_ids


def load_otu_taxonomy(path: Path) -> Dict[str, str]:
    logger.info("Loading OTU taxonomy from %s", path)
    otu_to_tax: Dict[str, str] = {}
    for chunk in pd.read_csv(path, sep="\t", usecols=["OTU", "Tax"], chunksize=200000):
        chunk = chunk.dropna(subset=["OTU", "Tax"])
        for otu_value, tax_value in zip(chunk["OTU"], chunk["Tax"]):
            tax = str(tax_value).strip()
            if not tax:
                continue
            for token in str(otu_value).split(";"):
                otu_id = token.strip()
                if otu_id.startswith("99_"):
                    otu_to_tax[otu_id] = tax
                    break
    return otu_to_tax


def build_rank_index(otu_ids: List[str], otu_to_tax: Dict[str, str]):
    matched_indices = []
    matched_ids = []
    rank_to_path_to_otus = {rank: defaultdict(list) for rank in RANKS}
    for ntv3_idx, otu_id in enumerate(otu_ids):
        tax = otu_to_tax.get(otu_id)
        if not tax:
            continue
        local_idx = len(matched_indices)
        matched_indices.append(ntv3_idx)
        matched_ids.append(otu_id)
        parts = [part.strip() for part in tax.replace("; ", ";").split(";") if part.strip()]
        for level, rank in enumerate(RANKS):
            if level >= len(parts):
                break
            path = ";".join(parts[: level + 1])
            rank_to_path_to_otus[rank][path].append(local_idx)
    return np.asarray(matched_indices, dtype=np.int64), matched_ids, rank_to_path_to_otus


def classify_status(own_cos: float, own_rank: int, reliable_cos: float, ambiguous_cos: float, ambiguous_rank: int) -> str:
    if own_rank == 1 and own_cos >= reliable_cos:
        return "reliable"
    if own_rank <= ambiguous_rank or own_cos >= ambiguous_cos:
        return "ambiguous"
    return "bad"


def build_top_genus_diagnostics(data: Dict[str, object], n_genera: int, retrieval_k: int, args) -> Tuple[List[dict], np.ndarray]:
    genus_to_otus = data["rank_to_path_to_otus"]["genus"]
    top_genera = sorted(genus_to_otus.items(), key=lambda item: -len(item[1]))[:n_genera]
    top_paths = [path for path, _ in top_genera]
    top_names = [path.split(";")[-1] for path in top_paths]
    top_means = np.vstack([data["matched_emb"][genus_to_otus[path]].mean(axis=0) for path in top_paths])

    nn = NearestNeighbors(n_neighbors=min(retrieval_k, len(data["matched_emb"])), metric="cosine", n_jobs=-1)
    nn.fit(data["matched_emb"])

    rows = []
    fb_vectors = []
    for genus_idx, (path, otu_indices) in enumerate(top_genera):
        key = f"genus|{path}"
        genus_name = top_names[genus_idx]
        fb = data["fb_emb"].get(key)
        if fb is None:
            rows.append(
                {
                    "plot_id": genus_idx + 1,
                    "genus": genus_name,
                    "path": path,
                    "n_otus": len(otu_indices),
                    "has_fallback": False,
                    "status": "missing",
                }
            )
            continue

        fb_vectors.append(fb)
        own_cos_to_mean = float(cosine_similarity(fb.reshape(1, -1), top_means[genus_idx : genus_idx + 1])[0, 0])
        cos_to_top_means = cosine_similarity(fb.reshape(1, -1), top_means)[0]
        order = np.argsort(-cos_to_top_means)
        own_rank = int(np.where(order == genus_idx)[0][0]) + 1
        nearest_idx = int(order[0])

        own_otu_cos = cosine_similarity(fb.reshape(1, -1), data["matched_emb"][otu_indices])[0]
        neighbor_idx = nn.kneighbors(fb.reshape(1, -1), return_distance=False)[0]
        retrieval_precision = sum(int(idx in set(otu_indices)) for idx in neighbor_idx) / max(1, len(neighbor_idx))

        status = classify_status(
            own_cos=own_cos_to_mean,
            own_rank=own_rank,
            reliable_cos=args.reliable_cos,
            ambiguous_cos=args.ambiguous_cos,
            ambiguous_rank=args.ambiguous_rank,
        )
        rows.append(
            {
                "plot_id": genus_idx + 1,
                "genus": genus_name,
                "path": path,
                "n_otus": len(otu_indices),
                "has_fallback": True,
                "status": status,
                "own_cos_to_genus_mean": own_cos_to_mean,
                "own_rank_among_top_genera": own_rank,
                "nearest_top_genus": top_names[nearest_idx],
                "nearest_top_genus_cos": float(cos_to_top_means[nearest_idx]),
                "own_otu_cos_mean": float(np.mean(own_otu_cos)),
                "own_otu_cos_median": float(np.median(own_otu_cos)),
                f"retrieval_precision_at_{retrieval_k}": retrieval_precision,
            }
        )
    return rows, np.vstack(fb_vectors) if fb_vectors else np.zeros((0, data["matched_emb"].shape[1]), dtype=np.float32)


def load_data(args):
    taxon_emb, taxon_keys, taxon_ranks, taxon_paths, taxon_counts = load_fallback(args.fallback)
    ntv3_emb, ntv3_ids = load_ntv3(args.ntv3)
    otu_to_tax = load_otu_taxonomy(args.otu_info)
    matched_indices, matched_ids, rank_to_path_to_otus = build_rank_index(ntv3_ids, otu_to_tax)

    raw_matched = ntv3_emb[matched_indices]
    if args.standardize:
        logger.info("Standardizing matched NTv3 embeddings and fallback embeddings")
        scaler = StandardScaler()
        matched_emb = scaler.fit_transform(raw_matched)
        taxon_emb = scaler.transform(taxon_emb)
    else:
        matched_emb = raw_matched

    return {
        "matched_emb": matched_emb,
        "otu_ids": matched_ids,
        "rank_to_path_to_otus": rank_to_path_to_otus,
        "fb_emb": {key: emb for key, emb in zip(taxon_keys, taxon_emb)},
        "taxon_keys": taxon_keys,
        "taxon_ranks": taxon_ranks,
        "taxon_paths": taxon_paths,
        "taxon_counts": taxon_counts,
    }


def make_umap(data: Dict[str, object], metrics: pd.DataFrame, args):
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("umap-learn is required for this plot") from exc

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

    logger.info("Running UMAP on %d OTUs + %d fallback centroids", len(selected_otu_indices), len(fallback_rows))
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
                "x": float(fallback_coords[row_idx, 0]),
                "y": float(fallback_coords[row_idx, 1]),
            }
        )
    return otu_coords, labels, fallback_coords, fallback_rows, pd.DataFrame(coordinate_rows)


def plot_diagnostic(data, metrics: pd.DataFrame, args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    otu_coords, labels, fallback_coords, fallback_rows, coordinates = make_umap(data, metrics, args)

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
                alpha=0.48,
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

    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.set_title(
        "Top genus OTUs + taxonomy fallback centroids\n"
        "diamond edge: high-dimensional reliability, dashed line: 2D fallback -> OTU mean",
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
        plt.Line2D([0], [0], marker="D", color="white", markerfacecolor="#CCCCCC", markeredgecolor=color, markeredgewidth=3, markersize=10, label=status)
        for status, color in STATUS_COLORS.items()
        if status != "missing"
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9)

    fig.suptitle(
        f"Fallback UMAP Diagnostic (top {args.n_genera} genera, {'standardized' if args.standardize else 'raw'} embedding space)",
        fontsize=15,
        fontweight="bold",
    )
    for fmt in ["png", "pdf"]:
        out = args.output_dir / f"fallback_umap_diagnostic.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", out)
    plt.close(fig)

    metrics_path = args.output_dir / "fallback_umap_diagnostic_metrics.csv"
    coords_path = args.output_dir / "fallback_umap_diagnostic_coordinates.csv"
    metrics.to_csv(metrics_path, index=False)
    coordinates.to_csv(coords_path, index=False)
    logger.info("Saved %s", metrics_path)
    logger.info("Saved %s", coords_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a UMAP overlay with high-dimensional diagnostics for taxonomy fallback genus centroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ntv3", type=Path, default=Path("ntv3_embeddings/otus_99_embeddings.h5"))
    parser.add_argument("--fallback", type=Path, default=Path("ntv3_embeddings/otus_99_taxonomy_fallback.h5"))
    parser.add_argument("--otu_info", type=Path, default=Path("/data6/zhanghaohong/databases/microbeatals/otus.info.tsv"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/fallback/fallback_eval_umap_diagnostic"))
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
    metric_rows, _ = build_top_genus_diagnostics(data, args.n_genera, args.retrieval_k, args)
    metrics = pd.DataFrame(metric_rows)
    plot_diagnostic(data, metrics, args)
    summary = metrics["status"].value_counts(dropna=False).to_dict()
    print(f"Done. Status counts: {summary}")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
