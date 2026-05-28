#!/usr/bin/env python3
"""
Evaluate taxonomy fallback reliability — improved metrics.

Metrics:
  1. Rank Monotonicity    — for each OTU, finer-rank fallback should be closer.
  2. LOO Reconstruction   — leave-one-out taxon mean vs left-out OTU (per rank).
  3. Fallback Retrieval   — use fallback as query, retrieve OTUs, check same-taxon rate.
  4. UMAP overlay         — original OTU embeddings + fallback centroids (genus level).

Usage:
  python scripts/regular_analysis/eval_taxonomy_fallback_v2.py
"""

import argparse, logging, os, sys
from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
RANK_COLORS = {
    "kingdom": "#6D2A1F", "phylum": "#4E7B48", "class": "#7CB159",
    "order": "#3D549D", "family": "#D4782B", "genus": "#AC4431",
    "species": "#7B1FA2",
}


def load_data(args):
    logger.info("Loading fallback ...")
    with h5py.File(args.fallback, "r") as f:
        taxon_emb = f["taxon_embeddings"][:].astype(np.float32)
        taxon_keys = [x.decode("utf-8") for x in f["taxon_keys"][:]]
        taxon_ranks = [x.decode("utf-8") for x in f["taxon_ranks"][:]]
        taxon_paths = [x.decode("utf-8") for x in f["taxon_paths"][:]]
        taxon_counts = f["taxon_counts"][:].astype(int)
        global_emb_raw = f["global_embedding"][:].astype(np.float32)

    logger.info("Loading original NTv3 ...")
    with h5py.File(args.ntv3, "r") as f:
        ntv3_emb = f["embeddings"][:].astype(np.float32)
        ntv3_ids = []
        for raw in f["otu_ids"][:]:
            t = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            t = t.strip().strip("'").strip('"')
            ntv3_ids.append(t)

    # Load OTU taxonomy
    logger.info("Loading OTU taxonomy ...")
    df = pd.read_csv(args.otu_info, sep="\t", usecols=["OTU", "Tax"])
    otu_to_tax = {}
    for _, row in df.iterrows():
        otu_str = str(row["OTU"])
        tax_str = str(row["Tax"]) if pd.notna(row["Tax"]) else ""
        for chunk in otu_str.split(";"):
            chunk = chunk.strip()
            if chunk.startswith("99_") and tax_str:
                otu_to_tax[chunk] = tax_str
                break

    # Match NTv3 with taxonomy
    idx_matched = []
    otu_ids_matched = []
    for i, oid in enumerate(ntv3_ids):
        if oid in otu_to_tax:
            idx_matched.append(i)
            otu_ids_matched.append(oid)

    # Standardize all embeddings together
    scaler = StandardScaler()
    emb_raw = ntv3_emb[idx_matched]
    matched_emb = scaler.fit_transform(emb_raw)
    taxon_emb_std = scaler.transform(taxon_emb)
    global_emb = scaler.transform(global_emb_raw.reshape(1, -1))[0]

    # Build fallback lookup: rank|path → embedding
    fb_emb = {}
    for k, e in zip(taxon_keys, taxon_emb_std):
        fb_emb[k] = e

    # Build per-rank: path → list of OTU indices
    rank_to_path_to_otus = {r: defaultdict(list) for r in RANKS}
    for idx, oid in enumerate(otu_ids_matched):
        tax_str = otu_to_tax[oid]
        parts = tax_str.replace("; ", ";").split(";")
        for level, rank in enumerate(RANKS):
            if level < len(parts):
                path = ";".join(parts[:level + 1])
                rank_to_path_to_otus[rank][path].append(idx)

    return {
        "matched_emb": matched_emb,
        "otu_ids": otu_ids_matched,
        "otu_to_tax": otu_to_tax,
        "fb_emb": fb_emb,
        "rank_to_path_to_otus": rank_to_path_to_otus,
        "taxon_emb_std": taxon_emb_std,
        "taxon_keys": taxon_keys,
        "taxon_ranks": taxon_ranks,
        "taxon_paths": taxon_paths,
        "taxon_counts": taxon_counts,
        "global_emb": global_emb,
    }


# ===================================================================
# Metric 1: Rank Monotonicity
#   For each OTU, compute cos-sim to its fallback at each rank.
#   Finer ranks should be more similar. Plot fraction showing monotonicity.
# ===================================================================
def metric_rank_monotonicity(data, output_dir, n_sample=5000, seed=42):
    logger.info("=== Rank Monotonicity ===")
    rng = np.random.default_rng(seed)
    idx_sample = rng.choice(len(data["otu_ids"]), size=min(n_sample, len(data["otu_ids"])), replace=False)

    all_sims = {r: [] for r in RANKS}
    count_monotonic = 0
    results = []

    for idx in tqdm(idx_sample, desc="Monotonicity"):
        oid = data["otu_ids"][idx]
        orig = data["matched_emb"][idx:idx + 1]
        tax_str = data["otu_to_tax"][oid]
        parts = tax_str.replace("; ", ";").split(";")

        sims = []
        for level, rank in enumerate(RANKS):
            if level >= len(parts):
                continue
            path = ";".join(parts[:level + 1])
            key = f"{rank}|{path}"
            fb = data["fb_emb"].get(key)
            if fb is None:
                continue
            sim = float(cosine_similarity(orig, fb.reshape(1, -1))[0, 0])
            sims.append(sim)
            all_sims[rank].append(sim)

        # Check monotonic: finer rank → higher similarity
        if len(sims) >= 3 and all(sims[i] <= sims[i + 1] for i in range(len(sims) - 1)):
            count_monotonic += 1

    mono_rate = count_monotonic / len(idx_sample)
    logger.info("Monotonic fraction: %.4f (%d/%d)", mono_rate, count_monotonic, len(idx_sample))

    # Plot: boxplot of similarity by rank
    plot_ranks = [r for r in RANKS if all_sims[r]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: boxplot
    data_bp = [all_sims[r] for r in plot_ranks]
    axes[0].boxplot(data_bp, patch_artist=True, showfliers=False)
    axes[0].set_xticklabels(plot_ranks, rotation=30, ha="right")
    axes[0].set_ylabel("Cosine similarity to fallback", fontsize=11)
    axes[0].set_title(f"Fallback Similarity by Rank\n({len(idx_sample):,} OTUs, {mono_rate:.1%} monotonic)", fontsize=11, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: per-OTU line plot (subsample for clarity)
    n_lines = min(100, len(idx_sample))
    line_idx = rng.choice(len(idx_sample), size=n_lines, replace=False)
    for i in line_idx[:30]:
        oid = data["otu_ids"][idx_sample[i]]
        tax_str = data["otu_to_tax"][oid]
        parts = tax_str.replace("; ", ";").split(";")
        orig = data["matched_emb"][idx_sample[i]:idx_sample[i] + 1]
        sims_line = []
        labels = []
        for level, rank in enumerate(RANKS):
            if level >= len(parts):
                continue
            path = ";".join(parts[:level + 1])
            key = f"{rank}|{path}"
            fb = data["fb_emb"].get(key)
            if fb is None:
                continue
            sim = float(cosine_similarity(orig, fb.reshape(1, -1))[0, 0])
            sims_line.append(sim)
            labels.append(rank[:2])
        axes[1].plot(range(len(sims_line)), sims_line, "-o", alpha=0.3, linewidth=0.8, markersize=3, color="#3D549D")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Cosine similarity", fontsize=11)
    axes[1].set_title("Per-OTU Monotonicity (30 examples)", fontsize=11, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Rank Monotonicity: Finer Rank → Higher Fallback Similarity",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_monotonicity.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved fallback_monotonicity.{png,pdf}")

    # Table
    rows = []
    for r in plot_ranks:
        rows.append({"rank": r, "mean_sim": np.mean(all_sims[r]), "std_sim": np.std(all_sims[r]), "n": len(all_sims[r])})
    rows.append({"rank": "global", "mean_sim": float(cosine_similarity(data["matched_emb"][idx_sample], data["global_emb"].reshape(1, -1)).mean()),
                  "std_sim": float(cosine_similarity(data["matched_emb"][idx_sample], data["global_emb"].reshape(1, -1)).std()),
                  "n": len(idx_sample)})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "fallback_monotonicity.csv"), index=False)


# ===================================================================
# Metric 2: Leave-One-Out Reconstruction
#   For taxa with >=3 OTUs, leave one out, compute mean of rest,
#   measure cosine similarity to left-out OTU. Plot per rank.
# ===================================================================
def metric_loo_reconstruction(data, output_dir, seed=42):
    logger.info("=== LOO Reconstruction ===")
    rng = np.random.default_rng(seed)

    ranks = ["phylum", "class", "order", "family", "genus"]
    loo_sims = {r: [] for r in ranks}

    for rank in ranks:
        good_paths = [p for p, idxs in data["rank_to_path_to_otus"][rank].items() if len(idxs) >= 5]
        if len(good_paths) < 3:
            continue
        # Sample up to 200 taxa
        paths_sample = rng.choice(good_paths, size=min(200, len(good_paths)), replace=False)
        for path in paths_sample:
            idxs = data["rank_to_path_to_otus"][rank][path]
            if len(idxs) < 3:
                continue
            loo = rng.choice(idxs)
            rest = [i for i in idxs if i != loo]
            if not rest:
                continue
            mean_emb = data["matched_emb"][rest].mean(axis=0)
            sim = float(cosine_similarity(data["matched_emb"][loo:loo + 1], mean_emb.reshape(1, -1))[0, 0])
            loo_sims[rank].append(sim)

    plot_ranks = [r for r in ranks if loo_sims[r]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: boxplot
    data_bp = [loo_sims[r] for r in plot_ranks]
    bp = axes[0].boxplot(data_bp, patch_artist=True, showfliers=False)
    for patch, r in zip(bp["boxes"], plot_ranks):
        patch.set_facecolor(RANK_COLORS[r])
    axes[0].set_xticklabels(plot_ranks, rotation=30, ha="right")
    axes[0].set_ylabel("LOO Cosine similarity", fontsize=11)
    axes[0].set_title("Leave-One-Out Reconstruction\n(per taxon mean vs left-out OTU)", fontsize=11, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: compare LOO mean vs fallback mean
    loo_means = [np.mean(loo_sims[r]) for r in plot_ranks]
    x = range(len(plot_ranks))
    axes[1].plot(x, loo_means, "o-", color="#3D549D", linewidth=2, markersize=8, label="LOO (gold standard)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_ranks, rotation=30, ha="right")
    axes[1].set_ylabel("Mean cosine similarity", fontsize=11)
    axes[1].set_title("LOO Reconstruction by Rank", fontsize=11, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("LOO Validation of Taxon-Mean Fallback", fontsize=13, fontweight="bold")
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_loo.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved fallback_loo.{png,pdf}")

    rows = []
    for r in plot_ranks:
        rows.append({"rank": r, "loo_mean_sim": np.mean(loo_sims[r]), "loo_std": np.std(loo_sims[r]), "n_pairs": len(loo_sims[r])})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "fallback_loo.csv"), index=False)


# ===================================================================
# Metric 3: Fallback Retrieval
#   Use fallback embedding as query, retrieve top-k OTU embeddings,
#   measure precision (same taxon?). Random baseline = 1/N_taxa.
# ===================================================================
def metric_fallback_retrieval(data, output_dir, k=10, seed=42):
    logger.info("=== Fallback Retrieval ===")
    rng = np.random.default_rng(seed)

    ranks = ["phylum", "class", "order", "family", "genus"]
    precision_curves = {}

    # Build k-NN index over all OTU embeddings
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(data["matched_emb"])

    for rank in ranks:
        # Get fallback keys for this rank
        fb_keys_rank = [(k, e) for k, e in zip(data["taxon_keys"], data["taxon_emb_std"]) if data["fb_emb"].get(k) is not None]
        # Filter to fallback embeddings that correspond to this rank
        rank_keys = [tk for tk, rk in zip(data["taxon_keys"], data["taxon_ranks"]) if rk == rank]
        rank_embs = {k: data["fb_emb"][k] for k in rank_keys if k in data["fb_emb"]}

        if len(rank_embs) < 5:
            continue

        # For each fallback, find k-NN OTUs, measure precision
        precisions = []
        for key, fb in tqdm(rank_embs.items(), desc=f"Retrieval {rank}", leave=False):
            # key format: "genus|Bacteria;Firmicutes;Bacillus"
            rank_name, taxon_path = key.split("|", 1)
            taxon_set = set(data["rank_to_path_to_otus"][rank].get(taxon_path, []))

            # Find nearest neighbors
            dists, indices = nn.kneighbors(fb.reshape(1, -1), n_neighbors=min(k + 1, len(data["matched_emb"])))
            hits = sum(1 for ni in indices[0] if ni in taxon_set)
            precisions.append(hits / k)

        precision_curves[rank] = np.mean(precisions) if precisions else 0
        logger.info("  %s: precision@%d = %.4f (n=%d fallbacks)", rank, k, precision_curves[rank], len(rank_embs))

    # Plot
    valid = [r for r in ranks if r in precision_curves]
    vals = [precision_curves[r] for r in valid]
    colors = [RANK_COLORS[r] for r in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(valid, vals, color=colors, edgecolor="#333", linewidth=1, width=0.55)
    for i, (r, v) in enumerate(zip(valid, vals)):
        # Estimate random baseline
        n_groups = len(set(p.split("|", 1)[1] for p in data["fb_emb"] if p.startswith(f"{r}|")))
        baseline = 1.0 / max(1, n_groups)
        ax.text(i, v + 0.01, f"{v:.3f}\n(random≈{baseline:.3f})",
                ha="center", fontsize=8, fontweight="bold", linespacing=1.3)
    ax.set_ylabel(f"Precision@{k}", fontsize=11)
    ax.set_title(f"Fallback Retrieval Precision@{k}\n(use fallback as query, retrieve OTUs, check same-taxon)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_retrieval.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved fallback_retrieval.{png,pdf}")

    rows = []
    for r in valid:
        n_grp = len(set(p.split("|", 1)[1] for p in data["fb_emb"] if p.startswith(f"{r}|")))
        rows.append({"rank": r, f"precision@{k}": precision_curves[r], "random_baseline": 1.0 / max(1, n_grp),
                      "n_fallbacks": n_grp, "enrichment": precision_curves[r] * n_grp})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "fallback_retrieval.csv"), index=False)


# ===================================================================
# Metric 4: UMAP overlay — original OTUs + fallback centroids (genus)
# ===================================================================
def metric_umap_overlay(data, output_dir, n_otus=10000, n_genera=20, seed=42):
    logger.info("=== UMAP Overlay ===")
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP overlay")
        return

    rng = np.random.default_rng(seed)

    # Pick top genera with most OTUs
    genus_to_otus = data["rank_to_path_to_otus"]["genus"]
    top_genera = sorted(genus_to_otus.items(), key=lambda x: -len(x[1]))[:n_genera]
    top_genus_names = [p.split(";")[-1] for p, _ in top_genera]
    top_genus_keys = {p: f"genus|{p}" for p, _ in top_genera}

    # Collect OTUs from top genera
    all_otus = []
    labels = []
    for gi, (path, idxs) in enumerate(top_genera):
        n = min(len(idxs), n_otus // n_genera)
        sample = rng.choice(idxs, size=n, replace=False)
        all_otus.extend(sample)
        labels.extend([gi] * n)
    all_otus = np.array(all_otus)
    labels = np.array(labels)

    # Collect fallback centroids for these genera
    fb_centroids = []
    fb_gi = []
    for gi, (path, _) in enumerate(top_genera):
        key = f"genus|{path}"
        if key in data["fb_emb"]:
            fb_centroids.append(data["fb_emb"][key])
            fb_gi.append(gi)

    # UMAP
    logger.info("Running UMAP on %d OTUs + %d centroids ...", len(all_otus), len(fb_centroids))
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed)
    combined = np.vstack([data["matched_emb"][all_otus], np.array(fb_centroids)])
    coords = reducer.fit_transform(combined)
    coords_otus = coords[:len(all_otus)]
    coords_fb = coords[len(all_otus):]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap("tab20", n_genera)
    for gi in range(n_genera):
        mask = labels == gi
        if mask.any():
            ax.scatter(coords_otus[mask, 0], coords_otus[mask, 1],
                       s=5, c=[cmap(gi)], alpha=0.3, edgecolors="none", rasterized=True)
    # Draw fallback centroids
    for i, gi in enumerate(fb_gi):
        ax.scatter(coords_fb[i, 0], coords_fb[i, 1],
                   s=200, c=[cmap(gi)], edgecolors="#333", linewidth=2, marker="D",
                   zorder=10, label=top_genus_names[gi])
        otu_coords = coords_otus[labels == gi]
        if len(otu_coords) > 0:
            ax.plot([coords_fb[i, 0], otu_coords[:, 0].mean()],
                    [coords_fb[i, 1], otu_coords[:, 1].mean()],
                    color=cmap(gi), alpha=0.5, linewidth=1.5, linestyle="--")

    ax.legend(fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(1, 0))
    ax.set_title(f"UMAP: Top {n_genera} Genera + Fallback Centroids (◆)\n(dashed = centroid → OTU mean, closer = better fallback)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_umap.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved fallback_umap.{png,pdf}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntv3", default="ntv3_embeddings/otus_99_embeddings.h5")
    parser.add_argument("--fallback", default="ntv3_embeddings/otus_99_taxonomy_fallback.h5")
    parser.add_argument("--otu_info", default="/data6/zhanghaohong/databases/microbeatals/otus.info.tsv")
    parser.add_argument("--output_dir", default="figures/fallback/fallback_eval_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args)

    metric_rank_monotonicity(data, args.output_dir)
    metric_loo_reconstruction(data, args.output_dir)
    metric_fallback_retrieval(data, args.output_dir)
    metric_umap_overlay(data, args.output_dir)

    print(f"\nDone. Figures + tables saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
