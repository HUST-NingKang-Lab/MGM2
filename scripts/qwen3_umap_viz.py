#!/usr/bin/env python3
"""
Qwen3 Embedding UMAP Visualization with Region Word Clouds.

Performs UMAP dimensionality reduction on Qwen3 keyword embeddings,
clusters the results, and annotates each cluster region with a word cloud
of the most characteristic keywords.

Usage:
    python scripts/qwen3_umap_viz.py \
        --emb_path qwen3_embeddings/microbeatlas_keywords_emb.h5 \
        --metadata_path /data6/zhanghaohong/databases/microbeatals/samples.env.info.tsv \
        --n_samples 50000 \
        --n_clusters 15 \
        --output_dir figures/umap_qwen3
"""

import argparse
import sys
import os
import json
import logging
import warnings
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_sample_id(raw: str) -> str:
    """Recursively strip nested b'...' / b\"...\" wrapping from HDF5 sample IDs."""
    t = raw.strip()
    while (t.startswith("b'") and t.endswith("'")) or (t.startswith('b"') and t.endswith('"')):
        t = t[2:-1]
    t = t.strip()
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        t = t[1:-1]
    return t


def load_embeddings_and_metadata(
    emb_path: str,
    metadata_path: str,
    n_samples: int,
    seed: int = 42,
):
    """Load embeddings matched with keywords — only samples with keywords are considered, guaranteeing 100% match."""
    logger.info("Loading metadata from %s", metadata_path)
    df = pd.read_csv(metadata_path, sep="\t", usecols=["MAP_SID", "Keywords"])
    df["MAP_SID"] = df["MAP_SID"].astype(str).str.strip()
    df = df.dropna(subset=["Keywords"])
    df = df[df["Keywords"].astype(str).str.strip() != ""]
    valid_meta_ids = set(df["MAP_SID"].tolist())
    sid_to_kw = dict(zip(df["MAP_SID"], df["Keywords"].astype(str)))
    logger.info("Metadata samples with non-empty keywords: %d", len(valid_meta_ids))

    logger.info("Loading embeddings from %s", emb_path)
    with h5py.File(emb_path, "r") as f:
        all_embeddings = f["embeddings"][:]  # (N, 4096) float16
        all_sample_ids_raw = [str(s) for s in f["sample_ids"][:]]

    N = len(all_sample_ids_raw)
    logger.info("Total embeddings: %d (dim=%d)", N, all_embeddings.shape[1])

    # Find intersection: sample IDs present in both H5 and metadata (with keywords)
    id_to_emb_idx = {}
    for i, raw in enumerate(all_sample_ids_raw):
        sid = clean_sample_id(raw)
        if sid in valid_meta_ids:
            id_to_emb_idx[sid] = i

    intersect_ids = sorted(id_to_emb_idx.keys())
    logger.info("Intersection (has embedding + keyword): %d samples", len(intersect_ids))

    if n_samples >= len(intersect_ids):
        selected_ids = intersect_ids
        logger.info("Using all %d intersecting samples", len(selected_ids))
    else:
        rng = np.random.default_rng(seed)
        selected_ids = sorted(rng.choice(intersect_ids, size=n_samples, replace=False).tolist())
        logger.info("Sampled %d / %d intersecting samples", n_samples, len(intersect_ids))

    # Gather embeddings and keywords
    keep_embeddings = np.stack([
        all_embeddings[id_to_emb_idx[sid]].astype(np.float32) for sid in selected_ids
    ], axis=0)
    keep_keywords = [sid_to_kw[sid] for sid in selected_ids]

    logger.info("Final dataset: %d samples (100%% keyword match)", len(keep_embeddings))
    return keep_embeddings, selected_ids, keep_keywords


def compute_umap(embeddings: np.ndarray, pca_dim: int = 100, random_state: int = 42):
    """Reduce embeddings: PCA -> UMAP -> 2D."""
    logger.info("Running PCA (%d -> %d)", embeddings.shape[1], pca_dim)
    pca = PCA(n_components=pca_dim, random_state=random_state)
    coords_pca = pca.fit_transform(embeddings)
    logger.info("PCA explained variance: %.3f", pca.explained_variance_ratio_.sum())

    logger.info("Running UMAP -> 2D (this may take a few minutes)...")
    try:
        import umap
    except ImportError:
        logger.error("umap-learn not installed. Run: pip install umap-learn")
        raise

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        low_memory=False,
        verbose=True,
    )
    coords_2d = reducer.fit_transform(StandardScaler().fit_transform(coords_pca))
    logger.info("UMAP done: shape=%s", coords_2d.shape)
    return coords_2d


def cluster_coords(coords_2d: np.ndarray, n_clusters: int, random_state: int = 42):
    """Cluster 2D UMAP coordinates with KMeans."""
    logger.info("Clustering with KMeans (k=%d)...", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_transform(coords_2d).argmin(axis=1)  # cluster per point via min distance
    # KMeans labels_ already gives cluster assignment
    cluster_ids = kmeans.labels_
    centroids = kmeans.cluster_centers_
    return cluster_ids, centroids


def compute_tfidf_keywords(keywords_list, cluster_ids, n_clusters, top_k=10):
    """Compute TF-IDF top keywords per cluster."""
    logger.info("Computing TF-IDF keywords per cluster...")

    # Build per-cluster documents
    cluster_docs = {c: [] for c in range(n_clusters)}
    cluster_all_words = {c: [] for c in range(n_clusters)}

    for kw_str, cid in zip(keywords_list, cluster_ids):
        # Split keywords
        tokens = [t.strip().lower() for t in kw_str.replace(";", ",").split(",") if t.strip()]
        cluster_docs[cid].append(" ".join(tokens))
        cluster_all_words[cid].extend(tokens)

    # TF-IDF per cluster
    # Use all cluster docs as corpus
    corpus = [" ".join(cluster_all_words[c]) for c in range(n_clusters)]

    # Filter out clusters with too few samples
    valid_clusters = [c for c in range(n_clusters) if len(cluster_docs[c]) > 0]
    if len(valid_clusters) < n_clusters:
        logger.warning("Some clusters have no samples: %s",
                       [c for c in range(n_clusters) if c not in valid_clusters])

    # Use simple frequency-based scoring (more robust than TF-IDF for single-word tokens)
    global_counter = Counter()
    for c in range(n_clusters):
        global_counter.update(cluster_all_words[c])

    cluster_top_keywords = {}
    for c in range(n_clusters):
        if not cluster_all_words[c]:
            cluster_top_keywords[c] = []
            continue
        local_counter = Counter(cluster_all_words[c])
        n_c = sum(local_counter.values())

        # TF-IDF-like: (local_freq / local_total) * log(N / doc_freq)
        scores = {}
        for word, local_count in local_counter.items():
            tf = local_count / n_c
            doc_freq = sum(1 for cc in range(n_clusters) if word in cluster_all_words[cc])
            idf = np.log((n_clusters + 1) / (doc_freq + 1)) + 1
            scores[word] = tf * idf

        sorted_words = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        cluster_top_keywords[c] = sorted_words

    return cluster_top_keywords


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_wordcloud_annotation(ax, x, y, keywords, cluster_size, color, max_font=14):
    """Draw a word cloud annotation at (x, y) on the axes."""
    if not keywords:
        return

    # Scale font sizes by TF-IDF weight
    max_weight = max(w for _, w in keywords)
    min_weight = min(w for _, w in keywords)
    weight_range = max_weight - min_weight if max_weight > min_weight else 1

    lines = []
    for word, weight in keywords[:8]:  # top 8 keywords
        font_size = max_font * (0.4 + 0.6 * (weight - min_weight) / weight_range)
        lines.append((word, font_size, weight))

    # Build text with varying font sizes via multiple text calls
    y_offset = 0
    text_objs = []
    for word, fs, _ in lines:
        t = ax.annotate(
            word,
            xy=(0, 0), xycoords="data",
            fontsize=fs, color=color, alpha=0.9,
            weight="bold", ha="center", va="center",
        )
        t.set_visible(False)  # We'll position manually
        text_objs.append((word, fs))

    # Simpler approach: just draw multi-line text
    # Position at (x, y)
    header = f"Cluster ({cluster_size:,} samples)"
    ax.annotate(
        header,
        xy=(x, y + 1.0),
        fontsize=8, color=color, alpha=0.7,
        ha="center", va="bottom", style="italic",
    )

    # Draw keyword text block
    kw_text = "\n".join(f"{word}" for word, _ in keywords[:8])
    ax.annotate(
        kw_text,
        xy=(x, y),
        fontsize=9, color=color, alpha=0.85,
        ha="center", va="center",
        weight="bold",
    )


def plot_umap_with_wordclouds(
    coords_2d: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_keywords: dict,
    centroids: np.ndarray,
    output_dir: str,
    title: str = "Qwen3 Keyword Embeddings — UMAP",
):
    """Create the main figure: UMAP scatter + cluster word cloud annotations."""
    n_clusters = centroids.shape[0]
    # Generate a colormap with enough distinct colors
    cmap = plt.get_cmap("tab20", n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]

    # ------ Full figure with word clouds ------
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_facecolor("#f8f8f8")

    # Plot each cluster as a scatter (subsample large clusters for speed)
    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            continue
        # Subsample for plotting if too many points
        plot_n = min(len(idx), 3000)
        if len(idx) > plot_n:
            plot_idx = np.random.default_rng(42).choice(idx, size=plot_n, replace=False)
        else:
            plot_idx = idx
        ax.scatter(
            coords_2d[plot_idx, 0], coords_2d[plot_idx, 1],
            s=10, c=[colors[cid]], alpha=0.35, edgecolors="none", rasterized=True,
        )

    # Annotate cluster centers with word clouds
    for cid in range(n_clusters):
        keywords = cluster_keywords.get(cid, [])
        if not keywords:
            continue
        cluster_size = int(np.sum(cluster_ids == cid))
        draw_wordcloud_annotation(
            ax,
            centroids[cid, 0], centroids[cid, 1],
            keywords, cluster_size,
            color=colors[cid],
            max_font=14,
        )

    # Draw cluster boundary boxes
    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) < 5:
            continue
        cx, cy = centroids[cid, 0], centroids[cid, 1]
        std_x = np.std(coords_2d[idx, 0]) * 1.5
        std_y = np.std(coords_2d[idx, 1]) * 1.5
        rect = FancyBboxPatch(
            (cx - std_x, cy - std_y), 2 * std_x, 2 * std_y,
            boxstyle="round,pad=0.1", facecolor="none",
            edgecolor=colors[cid], linewidth=1.5, alpha=0.5, linestyle="--",
        )
        ax.add_patch(rect)

    ax.set_title(title, fontsize=18, weight="bold", pad=15)
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.tick_params(labelsize=9)

    # Add legend below the plot
    # legend_handles = []
    # for cid in range(n_clusters):
    #     kw_list = cluster_keywords.get(cid, [])
    #     label = ", ".join(w for w, _ in kw_list[:5]) if kw_list else f"Cluster {cid}"
    #     from matplotlib.lines import Line2D
    #     legend_handles.append(
    #         Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[cid],
    #                markersize=10, label=f"C{cid}: {label}")
    #     )
    # ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(1.01, 0),
    #           fontsize=7, title="Clusters", title_fontsize=9)

    plt.tight_layout()
    fig_path_png = os.path.join(output_dir, "umap_qwen3_wordclouds.png")
    fig_path_pdf = os.path.join(output_dir, "umap_qwen3_wordclouds.pdf")
    fig.savefig(fig_path_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_path_pdf, bbox_inches="tight", facecolor="white")
    logger.info("Saved full figure to %s and %s", fig_path_png, fig_path_pdf)
    plt.close(fig)

    # ------ Clean scatter (no annotations) ------
    fig2, ax2 = plt.subplots(figsize=(15, 15))
    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            continue
        plot_n = min(len(idx), 3000)
        plot_idx = idx if len(idx) <= plot_n else np.random.default_rng(42).choice(idx, size=plot_n, replace=False)
        ax2.scatter(
            coords_2d[plot_idx, 0], coords_2d[plot_idx, 1],
            s=10, c=[colors[cid]], alpha=0.4, edgecolors="none", rasterized=True,
        )
    ax2.set_title(f"{title}\n({len(coords_2d):,} samples, {n_clusters} clusters)", fontsize=14, weight="bold")
    ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")
    fig2.tight_layout()
    scatter_path_png = os.path.join(output_dir, "umap_qwen3_scatter.png")
    scatter_path_pdf = os.path.join(output_dir, "umap_qwen3_scatter.pdf")
    fig2.savefig(scatter_path_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig2.savefig(scatter_path_pdf, bbox_inches="tight", facecolor="white")
    logger.info("Saved scatter-only figure to %s and %s", scatter_path_png, scatter_path_pdf)
    plt.close(fig2)

    # ------ Per-cluster word cloud subplots ------
    n_cols = 4
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for cid in range(n_clusters):
        ax = axes[cid]
        keywords = cluster_keywords.get(cid, [])
        if not keywords:
            ax.set_visible(False)
            continue
        cluster_size = int(np.sum(cluster_ids == cid))
        # Draw word cloud as horizontal bar chart of TF-IDF scores
        words, weights = zip(*keywords[:12])
        y_pos = range(len(words) - 1, -1, -1)
        bars = ax.barh(y_pos, weights, color=colors[cid], alpha=0.7, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_title(f"Cluster {cid} (n={cluster_size:,})", fontsize=10, weight="bold", color=colors[cid])
        ax.set_xlabel("TF-IDF", fontsize=8)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for cid in range(n_clusters, len(axes)):
        axes[cid].set_visible(False)

    fig3.suptitle("Per-Cluster Top Keywords (TF-IDF weighted)", fontsize=16, weight="bold", y=1.01)
    fig3.tight_layout()
    bar_path_png = os.path.join(output_dir, "umap_qwen3_cluster_keywords.png")
    bar_path_pdf = os.path.join(output_dir, "umap_qwen3_cluster_keywords.pdf")
    fig3.savefig(bar_path_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig3.savefig(bar_path_pdf, bbox_inches="tight", facecolor="white")
    logger.info("Saved cluster keyword bars to %s and %s", bar_path_png, bar_path_pdf)
    plt.close(fig3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UMAP visualization of Qwen3 embeddings with word cloud annotations")
    parser.add_argument("--emb_path", default='qwen3_embeddings/microbeatlas_keywords_emb.h5', help="Path to Qwen3 embeddings HDF5")
    parser.add_argument("--metadata_path", default='/data9/zhanghaohong/databases/microbeatals/samples.env.info.tsv', help="Path to samples.env.info.tsv")
    parser.add_argument("--n_samples", type=int, default=50000, help="Number of samples for UMAP (default: 50000)")
    parser.add_argument("--n_clusters", type=int, default=20, help="Number of clusters (default: 15)")
    parser.add_argument("--pca_dim", type=int, default=100, help="PCA intermediate dimension (default: 100)")
    parser.add_argument("--top_k", type=int, default=10, help="Top keywords per cluster (default: 10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="figures/umap_qwen3", help="Output directory")
    parser.add_argument("--skip_umap", action="store_true", help="Skip UMAP, load cached coords")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    embeddings, sample_ids, keywords = load_embeddings_and_metadata(
        args.emb_path, args.metadata_path, args.n_samples, args.seed
    )
    logger.info("Final dataset: %d samples", len(embeddings))

    # 2. UMAP (or load cached)
    coords_cache = os.path.join(args.output_dir, "umap_coords.npy")
    ids_cache = os.path.join(args.output_dir, "sample_ids.json")
    kw_cache = os.path.join(args.output_dir, "keywords.json")

    if args.skip_umap and os.path.exists(coords_cache):
        coords_2d = np.load(coords_cache)
        with open(ids_cache) as f:
            sample_ids = json.load(f)
        with open(kw_cache) as f:
            keywords = json.load(f)
        logger.info("Loaded cached UMAP coords: %s", coords_2d.shape)
    else:
        coords_2d = compute_umap(embeddings, pca_dim=args.pca_dim, random_state=args.seed)
        np.save(coords_cache, coords_2d)
        with open(ids_cache, "w") as f:
            json.dump(sample_ids, f)
        with open(kw_cache, "w") as f:
            json.dump(keywords, f)
        logger.info("Cached UMAP coords to %s", coords_cache)

    # 3. Cluster
    cluster_ids, centroids = cluster_coords(coords_2d, args.n_clusters, random_state=args.seed)

    # 4. TF-IDF keywords per cluster
    cluster_keywords = compute_tfidf_keywords(keywords, cluster_ids, args.n_clusters, top_k=args.top_k)

    # Print cluster summaries
    print("\n" + "=" * 70)
    print("CLUSTER SUMMARIES")
    print("=" * 70)
    for cid in sorted(cluster_keywords.keys()):
        kw_list = cluster_keywords[cid]
        n = int(np.sum(cluster_ids == cid))
        kw_str = " | ".join(f"{w} ({s:.3f})" for w, s in kw_list[:8])
        print(f"  Cluster {cid:2d} (n={n:6d}): {kw_str}")

    # 5. Plot
    plot_umap_with_wordclouds(
        coords_2d, cluster_ids, cluster_keywords, centroids,
        args.output_dir,
        title=f"Qwen3 Keyword Embeddings — UMAP ({len(coords_2d):,} samples, {args.n_clusters} clusters)",
    )

    print(f"\nDone! Output saved to {args.output_dir}/")
    print(f"  - umap_qwen3_wordclouds.png/pdf  (UMAP with cluster word clouds)")
    print(f"  - umap_qwen3_scatter.png/pdf     (clean scatter plot)")
    print(f"  - umap_qwen3_cluster_keywords.png/pdf (per-cluster keyword bars)")


if __name__ == "__main__":
    main()
