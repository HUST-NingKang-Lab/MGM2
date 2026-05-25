#!/usr/bin/env python3
"""
Evaluate taxonomy fallback embedding reliability.

Figures produced:
  1. fallback_coverage.png/pdf      — per-rank prototype & OTU count distribution
  2. fallback_within_between.png/pdf — within-taxon vs between-taxon cosine sim
  3. fallback_reconstruction.png/pdf — original vs fallback cosine sim (info loss)
  4. fallback_silhouette.png/pdf     — silhouette score by taxonomic rank

Usage:
  python scripts/regular_analysis/eval_taxonomy_fallback.py \
      --ntv3 ntv3_embeddings/otus_99_embeddings.h5 \
      --fallback ntv3_embeddings/otus_99_taxonomy_fallback.h5 \
      --otu_info /data6/zhanghaohong/databases/microbeatals/otus.info.tsv \
      --output_dir figures/fallback/fallback_eval
"""

import argparse, json, logging, os, sys
from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
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
    logger.info("Loading fallback embeddings ...")
    with h5py.File(args.fallback, "r") as f:
        taxon_emb = f["taxon_embeddings"][:].astype(np.float32)
        taxon_keys = [str(x) for x in f["taxon_keys"][:]]
        taxon_ranks = [x.decode("utf-8") for x in f["taxon_ranks"][:]]
        taxon_paths = [str(x) for x in f["taxon_paths"][:]]
        taxon_counts = f["taxon_counts"][:].astype(int)

    logger.info("Loading original NTv3 embeddings ...")
    with h5py.File(args.ntv3, "r") as f:
        ntv3_emb = f["embeddings"][:].astype(np.float32)
        ntv3_ids = [str(x) for x in f["otu_ids"][:]]
        # Clean: b'99_XXXXX' -> 99_XXXXX
        ntv3_ids = []
        for raw in f["otu_ids"][:]:
            t = str(raw).strip()
            while (t.startswith("b'") and t.endswith("'")) or (t.startswith('b"') and t.endswith('"')):
                t = t[2:-1]
            t = t.strip().strip("'").strip('"')
            ntv3_ids.append(t)

    # Build rank→path→taxon_embedding index for quick lookup
    path_to_emb = {}
    for key, path, emb in zip(taxon_keys, taxon_paths, taxon_emb):
        path_to_emb[key] = emb  # key format: "phylum|Bacteria;Firmicutes"

    # Load OTU taxonomy
    logger.info("Loading OTU taxonomy ...")
    df = pd.read_csv(args.otu_info, sep="\t", usecols=["OTU", "Tax"])
    otu_to_tax = {}
    for _, row in df.iterrows():
        otu_str = str(row["OTU"])
        tax_str = str(row["Tax"]) if pd.notna(row["Tax"]) else ""
        for chunk in otu_str.split(";"):
            chunk = chunk.strip()
            if chunk.startswith("99_"):
                otu_to_tax[chunk] = tax_str
                break

    # Match NTv3 OTUs with taxonomy
    idx_in_ntv3 = []
    otu_ids_matched = []
    for i, oid in enumerate(ntv3_ids):
        if oid in otu_to_tax:
            idx_in_ntv3.append(i)
            otu_ids_matched.append(oid)

    logger.info("Matched %d / %d OTUs with taxonomy", len(idx_in_ntv3), len(ntv3_ids))
    # Standardize all embeddings with same scaler for meaningful cosine distances
    logger.info("Standardizing NTv3 + fallback embeddings together ...")
    scaler = StandardScaler()
    matched_emb = scaler.fit_transform(ntv3_emb[idx_in_ntv3])
    taxon_emb_std = scaler.transform(taxon_emb)
    global_emb = scaler.transform(
        np.array(h5py.File(args.fallback, "r")["global_embedding"][:]).reshape(1, -1)
    )[0]

    # Rebuild path_to_emb with standardized fallback embeddings
    path_to_emb_std = {}
    for key, emb in zip(taxon_keys, taxon_emb_std):
        path_to_emb_std[key] = emb

    data = {
        "taxon_emb": taxon_emb_std,
        "taxon_keys": taxon_keys,
        "taxon_ranks": taxon_ranks,
        "taxon_paths": taxon_paths,
        "taxon_counts": taxon_counts,
        "path_to_emb": path_to_emb_std,
        "otu_to_tax": otu_to_tax,
        "matched_emb": matched_emb,
        "otu_ids_matched": otu_ids_matched,
        "global_emb": global_emb,
    }
    return data


# ===================================================================
# Figure 1: Coverage — per-rank prototype & OTU count distribution
# ===================================================================
def fig_coverage(data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: number of prototypes per rank
    rank_counts = pd.Series(data["taxon_ranks"]).value_counts().reindex(RANKS).fillna(0).astype(int)
    colors = [RANK_COLORS[r] for r in RANKS]
    bars = axes[0].bar(range(len(RANKS)), rank_counts.values, color=colors, edgecolor="#333", linewidth=1)
    for i, v in enumerate(rank_counts.values):
        axes[0].text(i, v + max(rank_counts.values) * 0.02, str(v), ha="center", fontsize=9, fontweight="bold")
    axes[0].set_xticks(range(len(RANKS)))
    axes[0].set_xticklabels(RANKS, rotation=30, ha="right", fontsize=10)
    axes[0].set_ylabel("Number of prototypes", fontsize=11)
    axes[0].set_title("Taxon Prototypes per Rank", fontsize=12, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right: backing OTU count distribution per rank (boxplot, log scale)
    rank_data = []
    rank_labels = []
    for r in RANKS:
        idx = np.array(data["taxon_ranks"]) == r
        if idx.any():
            rank_data.append(data["taxon_counts"][idx])
            rank_labels.append(r)
    bp = axes[1].boxplot(rank_data, patch_artist=True, showfliers=True,
                         flierprops=dict(marker=".", alpha=0.3, markersize=3))
    axes[1].set_xticklabels(rank_labels)
    for patch, r in zip(bp["boxes"], rank_labels):
        patch.set_facecolor(RANK_COLORS[r])
    axes[1].set_yscale("log")
    axes[1].set_ylabel("OTU count per taxon (log)", fontsize=11)
    axes[1].set_title("Backing OTU Count Distribution", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=30, labelsize=10)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Fallback Embedding Coverage", fontsize=14, fontweight="bold")
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_coverage.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved fallback_coverage.{png,pdf}")

    # Table
    summary = []
    for r, d in zip(rank_labels, rank_data):
        summary.append({
            "rank": r, "n_prototypes": len(d),
            "min_otus": int(np.min(d)), "median_otus": int(np.median(d)),
            "max_otus": int(np.max(d)), "mean_otus": np.mean(d),
        })
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "fallback_coverage.csv"), index=False)
    logger.info("Saved fallback_coverage.csv")


# ===================================================================
# Figure 2: Within-taxon vs Between-taxon Cosine Similarity
# ===================================================================
def fig_within_between(data, output_dir, n_pairs=5000, seed=42):
    rng = np.random.default_rng(seed)
    ranks = ["phylum", "class", "order", "family", "genus", "species"]

    # Index: rank → path → list of OTU indices
    # Species-level: lower min OTUs per taxon to 3 for within-pair sampling
    path_to_otu_idx = defaultdict(list)
    for idx, oid in enumerate(data["otu_ids_matched"]):
        tax_str = data["otu_to_tax"][oid]
        parts = tax_str.replace("; ", ";").split(";")
        for level, rank in enumerate(ranks):
            if level < len(parts):
                path = ";".join(parts[:level + 1])
                path_to_otu_idx[(rank, path)].append(idx)

    fig, axes = plt.subplots(1, len(ranks), figsize=(21, 5))
    results = []

    min_otus_for_rank = {"species": 2}  # species taxa often have only 2-3 OTUs

    for ax, rank in zip(axes, ranks):
        within_sims = []
        between_sims = []
        paths = sorted(set(k[1] for k in path_to_otu_idx if k[0] == rank))
        path_to_idx = {p: path_to_otu_idx[(rank, p)] for p in paths}
        min_n = min_otus_for_rank.get(rank, 3)
        valid_paths = [p for p in paths if len(path_to_idx[p]) >= min_n]

        for _ in range(n_pairs):
            # Within: two random OTUs from same taxon
            if valid_paths:
                p = rng.choice(valid_paths)
                a, b = rng.choice(path_to_idx[p], size=2, replace=False)
                within_sims.append(float(cosine_similarity(
                    data["matched_emb"][a:a + 1], data["matched_emb"][b:b + 1])[0, 0]))

            # Between: two OTUs from different taxa
            if len(valid_paths) >= 2:
                p1, p2 = rng.choice(valid_paths, size=2, replace=False)
                a = rng.choice(path_to_idx[p1])
                b = rng.choice(path_to_idx[p2])
                between_sims.append(float(cosine_similarity(
                    data["matched_emb"][a:a + 1], data["matched_emb"][b:b + 1])[0, 0]))

        within_sims = np.array(within_sims)
        between_sims = np.array(between_sims)
        delta = np.mean(within_sims) - np.mean(between_sims)
        results.append((rank, np.mean(within_sims), np.mean(between_sims), delta))

        ax.hist(within_sims, bins=40, alpha=0.6, color=RANK_COLORS[rank], label=f"Within (μ={np.mean(within_sims):.3f})")
        ax.hist(between_sims, bins=40, alpha=0.4, color="#999", label=f"Between (μ={np.mean(between_sims):.3f})")
        ax.set_title(f"{rank}\nΔ={delta:.4f}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Cosine similarity", fontsize=8)

    axes[0].set_ylabel("Frequency", fontsize=10)
    fig.suptitle("Within-Taxon vs Between-Taxon Cosine Similarity of OTU-level NTv3 Embeddings\n(larger Δ = more reliable fallback)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_within_between.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved fallback_within_between.{png,pdf}")
    pd.DataFrame(results, columns=["rank", "within_mean", "between_mean", "delta"]).to_csv(
        os.path.join(output_dir, "fallback_within_between.csv"), index=False)
    logger.info("Saved fallback_within_between.csv")
    for r, w, b, d in results:
        logger.info("  %s: within=%.4f  between=%.4f  Δ=%.4f", r, w, b, d)


# ===================================================================
# Figure 3: Original vs Fallback Reconstruction (info loss)
# ===================================================================
def fig_reconstruction(data, output_dir, n_sample=3000, seed=42):
    rng = np.random.default_rng(seed)

    # For each OTU, compute cosine similarity between its original NTv3 embedding
    # and the fallback embedding at each level
    ranks = ["phylum", "class", "order", "family", "genus", "species"]

    # Sample OTUs
    n = min(n_sample, len(data["otu_ids_matched"]))
    sample_idx = rng.choice(len(data["otu_ids_matched"]), size=n, replace=False)

    sims_by_rank = {r: [] for r in ranks}
    global_emb = data["global_emb"]

    for idx in tqdm(sample_idx, desc="Reconstruction"):
        oid = data["otu_ids_matched"][idx]
        orig = data["matched_emb"][idx:idx + 1]
        tax_str = data["otu_to_tax"][oid]
        parts = tax_str.replace("; ", ";").split(";")

        for level, rank in enumerate(ranks):
            if level >= len(parts):
                continue
            path = ";".join(parts[:level + 1])
            key = f"{rank}|{path}"
            fb = data["path_to_emb"].get(key)
            if fb is not None:
                sims_by_rank[rank].append(float(cosine_similarity(orig, fb.reshape(1, -1))[0, 0]))
        # Global fallback
        if global_emb is not None:
            sims_by_rank.setdefault("global", []).append(
                float(cosine_similarity(orig, global_emb.reshape(1, -1))[0, 0]))

    # Plot
    plot_ranks = [r for r in ranks + ["global"] if r in sims_by_rank and sims_by_rank[r]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data_to_plot = [sims_by_rank[r] for r in plot_ranks]
    colors = [RANK_COLORS.get(r, "#888") for r in plot_ranks]

    bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=False, widths=0.6)
    ax.set_xticklabels(plot_ranks)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)

    # Mean line
    means = [np.mean(d) for d in data_to_plot]
    ax.plot(range(1, len(plot_ranks) + 1), means, "D-", color="#333", linewidth=2, markersize=7, label="mean")

    # Text labels
    for i, (r, m) in enumerate(zip(plot_ranks, means)):
        ax.annotate(f"{m:.3f}", (i + 1, m), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Cosine similarity (original vs fallback)", fontsize=11)
    ax.set_title("Original NTv3 vs Fallback Embedding Similarity\n(higher = less information lost)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_reconstruction.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved fallback_reconstruction.{png,pdf}")
    recon_rows = []
    for r, vals in sims_by_rank.items():
        recon_rows.append({"rank": r, "mean_sim": np.mean(vals), "std_sim": np.std(vals), "n": len(vals)})
        logger.info("  %s: mean=%.4f ± %.4f", r, np.mean(vals), np.std(vals))
    pd.DataFrame(recon_rows).to_csv(os.path.join(output_dir, "fallback_reconstruction.csv"), index=False)
    logger.info("Saved fallback_reconstruction.csv")


# ===================================================================
# Figure 4: Silhouette Score by Rank
# ===================================================================
def fig_silhouette(data, output_dir, n_sample=20000, seed=42):
    rng = np.random.default_rng(seed)

    ranks = ["phylum", "class", "order", "family", "genus"]

    # For each rank, assign OTU to its taxon label, compute silhouette
    silhouettes = {}
    n_effective = {}

    for rank in ranks:
        # Group OTUs by taxonomy path at this rank
        path_to_idx = defaultdict(list)
        for idx, oid in enumerate(data["otu_ids_matched"]):
            tax_str = data["otu_to_tax"][oid]
            parts = tax_str.replace("; ", ";").split(";")
            level = ranks.index(rank)
            if level < len(parts):
                path = ";".join(parts[:level + 1])
                path_to_idx[path].append(idx)

        # Filter groups with >=5 members
        groups = {p: idxs for p, idxs in path_to_idx.items() if len(idxs) >= 5}
        if len(groups) < 2:
            silhouettes[rank] = np.nan
            n_effective[rank] = 0
            continue

        # Build X and y
        all_idx = []
        labels = []
        for label, (path, idxs) in enumerate(groups.items()):
            all_idx.extend(idxs)
            labels.extend([label] * len(idxs))

        # Sample to limit size
        if len(all_idx) > n_sample:
            sub_idx = rng.choice(len(all_idx), size=n_sample, replace=False)
            all_idx = [all_idx[i] for i in sub_idx]
            labels = [labels[i] for i in sub_idx]

        X = data["matched_emb"][all_idx]
        y = np.array(labels)
        s = silhouette_score(StandardScaler().fit_transform(X), y, metric="cosine", random_state=seed)
        silhouettes[rank] = s
        n_effective[rank] = len(groups)
        logger.info("  %s: silhouette=%.4f  (n=%d samples, %d groups)", rank, s, len(y), len(groups))

    # Plot
    valid_ranks = [r for r in ranks if not np.isnan(silhouettes[r])]
    vals = [silhouettes[r] for r in valid_ranks]
    colors = [RANK_COLORS[r] for r in valid_ranks]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(valid_ranks, vals, color=colors, edgecolor="#333", linewidth=1, width=0.55)
    for i, (r, v) in enumerate(zip(valid_ranks, vals)):
        ax.text(i, v + 0.01, f"{v:.4f}\n({n_effective[r]} groups)",
                ha="center", fontsize=9, fontweight="bold", linespacing=1.3)
    ax.set_ylabel("Silhouette Score (cosine)", fontsize=11)
    ax.set_title("Silhouette Score by Taxonomic Rank\n(higher = taxa better separated in embedding space)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0, color="#999", linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_silhouette.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved fallback_silhouette.{png,pdf}")
    sil_rows = []
    for r in ranks:
        sil_rows.append({
            "rank": r, "silhouette": silhouettes[r] if not np.isnan(silhouettes.get(r, np.nan)) else None,
            "n_groups": n_effective.get(r, 0),
        })
    pd.DataFrame(sil_rows).to_csv(os.path.join(output_dir, "fallback_silhouette.csv"), index=False)
    logger.info("Saved fallback_silhouette.csv")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate taxonomy fallback embeddings")
    parser.add_argument("--ntv3", default="ntv3_embeddings/otus_99_embeddings.h5")
    parser.add_argument("--fallback", default="ntv3_embeddings/otus_99_taxonomy_fallback.h5")
    parser.add_argument("--otu_info", default="/data6/zhanghaohong/databases/microbeatals/otus.info.tsv")
    parser.add_argument("--output_dir", default="figures/fallback/fallback_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    data = load_data(args)

    fig_coverage(data, args.output_dir)
    fig_within_between(data, args.output_dir)
    fig_reconstruction(data, args.output_dir)
    fig_silhouette(data, args.output_dir)

    print(f"\nDone. Figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
