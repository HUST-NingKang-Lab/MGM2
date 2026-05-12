#!/usr/bin/env python3
"""
Validate NTv3 embeddings via two complementary approaches:

  Method 2 — Sequence–Embedding Correlation:
      k-mer Jaccard similarity vs. NTv3 cosine similarity on random OTU pairs.

  Method 5 — k-NN Taxonomy Retrieval:
      For each OTU, find nearest neighbours in embedding space and report
      the fraction that share the same phylum/class/order/family/genus.

Output:
  {output_dir}/ntv3_seq_emb_corr.{png,pdf}   — Method 2 scatter + hexbin
  {output_dir}/ntv3_knn_taxonomy.{png,pdf}   — Method 5 retrieval-precision curves
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.interpolate import UnivariateSpline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(emb_path):
    """Load NTv3 embeddings, return {otu_id: np.ndarray}."""
    logger.info("Loading NTv3 embeddings from %s", emb_path)
    with h5py.File(emb_path, "r") as f:
        embeddings = f["embeddings"][:].astype(np.float32)
        raw_ids = [str(x) for x in f["otu_ids"][:]]
        # Clean: b'99_XXXXX' -> 99_XXXXX
        otu_ids = []
        for rid in raw_ids:
            t = rid.strip()
            while (t.startswith("b'") and t.endswith("'")) or (t.startswith('b"') and t.endswith('"')):
                t = t[2:-1]
            t = t.strip().strip("'").strip('"')
            otu_ids.append(t)
    logger.info("Loaded %d embeddings (dim=%d)", len(otu_ids), embeddings.shape[1])
    return embeddings, otu_ids


def load_sequences(fasta_path, otu_ids_set):
    """Load 16S sequences for OTUs of interest. Returns {otu_id: seq}."""
    logger.info("Loading sequences from %s", fasta_path)
    seqs = {}
    current_id = None
    current_seq = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None and current_id in otu_ids_set:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]  # e.g. >99_90652
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)
        if current_id is not None and current_id in otu_ids_set:
            seqs[current_id] = "".join(current_seq)
    logger.info("Loaded %d / %d sequences", len(seqs), len(otu_ids_set))
    return seqs


def load_taxonomy(tax_path, otu_ids_set):
    """Load taxonomy from otus.info.tsv, return {otu_id: tax_dict}."""
    logger.info("Loading taxonomy from %s", tax_path)
    df = pd.read_csv(tax_path, sep="\t", usecols=["OTU", "Tax"])
    ranks = ["kingdom", "phylum", "class", "order", "family", "genus"]

    tax = {}
    for _, row in df.iterrows():
        otu_str = str(row["OTU"])
        tax_str = str(row["Tax"]) if pd.notna(row["Tax"]) else ""
        parts = tax_str.replace("; ", ";").split(";")

        # Extract 99_ ID from semicolon-separated OTU field
        for chunk in otu_str.split(";"):
            chunk = chunk.strip()
            if chunk.startswith("99_") and chunk in otu_ids_set:
                tax_dict = {}
                for i, r in enumerate(ranks):
                    tax_dict[r] = parts[i].strip() if i < len(parts) and parts[i].strip() else "Unknown"
                tax[chunk] = tax_dict
                break

    logger.info("Matched taxonomy for %d OTUs", len(tax))
    return tax


# ---------------------------------------------------------------------------
# Method 2: Sequence (k-mer Jaccard) vs Embedding Cosine Similarity
# ---------------------------------------------------------------------------

def kmer_profile(seq, k=6):
    """Return set of k-mers for a sequence."""
    if len(seq) < k:
        return set()
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def jaccard(s1, s2):
    """Jaccard similarity between two sets."""
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def method2_seq_emb_corr(embeddings, otu_ids, fasta_path, output_dir,
                         n_sample=5000, n_pairs=15000, k=6, seed=42):
    """Scatter plot: k-mer Jaccard vs embedding cosine similarity."""
    logger.info("=== Method 2: Sequence–Embedding Correlation ===")
    rng = np.random.default_rng(seed)

    # Sample OTUs
    n_sample = min(n_sample, len(otu_ids))
    sample_idx = rng.choice(len(otu_ids), size=n_sample, replace=False)
    sample_idx.sort()
    sampled_ids = [otu_ids[i] for i in sample_idx]
    sampled_embs = embeddings[sample_idx]

    # Standardize embeddings so cosine similarity has meaningful dynamic range
    logger.info("Standardizing embeddings (StandardScaler per dimension)...")
    sampled_embs = StandardScaler().fit_transform(sampled_embs)

    # Load sequences for sampled OTUs
    id_set = set(sampled_ids)
    seqs = load_sequences(fasta_path, id_set)

    # Keep only OTUs with sequences
    valid = [(i, oid) for i, oid in enumerate(sampled_ids) if oid in seqs and len(seqs[oid]) >= 100]
    logger.info("OTUs with valid sequences: %d / %d", len(valid), n_sample)

    if len(valid) < 200:
        logger.error("Too few valid OTUs, aborting Method 2")
        return

    # Compute k-mer profiles
    logger.info("Computing k=%d profiles for %d OTUs...", k, len(valid))
    profiles = {}
    for idx, oid in tqdm(valid, desc="k-mer profiles"):
        profiles[oid] = kmer_profile(seqs[oid], k=k)

    # Sample random pairs
    n_pairs = min(n_pairs, len(valid) * (len(valid) - 1) // 2)
    logger.info("Sampling %d random pairs...", n_pairs)

    jaccard_sims = []
    cosine_sims = []
    valid_list = list(valid)  # (emb_idx, otu_id)

    pair_set = set()
    while len(pair_set) < n_pairs:
        a = rng.integers(0, len(valid_list))
        b = rng.integers(0, len(valid_list))
        if a == b:
            continue
        pair = (min(a, b), max(a, b))
        if pair in pair_set:
            continue
        pair_set.add(pair)

        idx_a, oid_a = valid_list[a]
        idx_b, oid_b = valid_list[b]

        jac = jaccard(profiles[oid_a], profiles[oid_b])
        cos = float(cosine_similarity(sampled_embs[idx_a:idx_a + 1], sampled_embs[idx_b:idx_b + 1])[0, 0])

        jaccard_sims.append(jac)
        cosine_sims.append(cos)

    jaccard_sims = np.array(jaccard_sims)
    cosine_sims = np.array(cosine_sims)

    # Correlations
    mask = jaccard_sims > 0  # exclude zero-overlap pairs for cleaner stats
    pr, pp = pearsonr(jaccard_sims[mask], cosine_sims[mask])
    sr, sp = spearmanr(jaccard_sims, cosine_sims)
    logger.info("Pearson r=%.4f (p=%.2e) | Spearman rho=%.4f (p=%.2e)", pr, pp, sr, sp)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter
    ax = axes[0]
    ax.scatter(jaccard_sims, cosine_sims, s=2, alpha=0.3, c="#3D549D", edgecolors="none", rasterized=True)
    ax.set_xlabel("k-mer Jaccard similarity (k=6)", fontsize=12)
    ax.set_ylabel("NTv3 embedding cosine similarity", fontsize=12)
    ax.set_title(f"Sequence vs. Embedding Similarity\n{n_pairs:,} OTU pairs (Pearson r={pr:.3f}, Spearman ρ={sr:.3f})",
                 fontsize=11, weight="bold")
    # Trend line
    idx_sort = np.argsort(jaccard_sims)
    window = max(1, len(idx_sort) // 50)
    trend_x = np.convolve(jaccard_sims[idx_sort], np.ones(window) / window, mode="valid")
    trend_y = np.convolve(cosine_sims[idx_sort], np.ones(window) / window, mode="valid")
    ax.plot(trend_x, trend_y, color="#D4782B", linewidth=2, label="moving avg")
    # Smooth spline
    spl = UnivariateSpline(jaccard_sims[idx_sort], cosine_sims[idx_sort], s=len(idx_sort) * 0.05)
    spl_x = np.linspace(jaccard_sims.min(), jaccard_sims.max(), 200)
    spl_y = spl(spl_x)
    ax.plot(spl_x, spl_y, color="#AC4431", linewidth=2, linestyle="--", label="spline")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: hexbin
    ax2 = axes[1]
    hb = ax2.hexbin(jaccard_sims, cosine_sims, gridsize=50, cmap="YlOrRd", mincnt=1, rasterized=True)
    plt.colorbar(hb, ax=ax2, label="Count")
    ax2.set_xlabel("k-mer Jaccard similarity (k=6)", fontsize=12)
    ax2.set_ylabel("NTv3 embedding cosine similarity", fontsize=12)
    ax2.set_title("Sequence vs. Embedding (density)", fontsize=11, weight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"ntv3_seq_emb_corr.{fmt}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved %s/ntv3_seq_emb_corr.{png,pdf}", output_dir)

    results = {"pearson_r": float(pr), "pearson_p": float(pp),
               "spearman_rho": float(sr), "spearman_p": float(sp),
               "n_pairs": n_pairs, "n_otus": len(valid)}
    with open(os.path.join(output_dir, "ntv3_seq_emb_corr.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Method 5: k-NN Taxonomy Retrieval
# ---------------------------------------------------------------------------

def method5_knn_taxonomy(embeddings, otu_ids, tax, output_dir,
                          n_sample=30000, k_max=10, seed=42):
    """k-NN retrieval precision by taxonomic rank."""
    logger.info("=== Method 5: k-NN Taxonomy Retrieval ===")
    rng = np.random.default_rng(seed)

    # Filter to OTUs with taxonomy
    tax_ids = [oid for oid in otu_ids if oid in tax]
    tax_idx = [i for i, oid in enumerate(otu_ids) if oid in tax]
    logger.info("OTUs with taxonomy: %d / %d", len(tax_ids), len(otu_ids))

    # Sample for efficiency
    n_sample = min(n_sample, len(tax_idx))
    sample = rng.choice(len(tax_idx), size=n_sample, replace=False)
    sample.sort()
    tax_idx_sample = [tax_idx[i] for i in sample]
    tax_ids_sample = [tax_ids[i] for i in sample]
    tax_embs = embeddings[tax_idx_sample]

    # Standardize embeddings
    logger.info("Standardizing embeddings (StandardScaler per dimension)...")
    tax_embs = StandardScaler().fit_transform(tax_embs)

    ranks = ["phylum", "class", "order", "family", "genus"]

    # Build k-NN index
    logger.info("Building k-NN index for %d OTUs...", n_sample)
    nn = NearestNeighbors(n_neighbors=k_max + 1, metric="cosine", n_jobs=-1)
    nn.fit(tax_embs)
    distances, indices = nn.kneighbors(tax_embs)

    # For each k, compute retrieval precision per rank
    precisions = {r: [] for r in ranks}

    for k in range(1, k_max + 1):
        for rank in ranks:
            correct = 0
            total = 0
            for i in range(n_sample):
                query_tax = tax[tax_ids_sample[i]].get(rank, "Unknown")
                neighbor_indices = indices[i, 1:k + 1]  # skip self (index 0)
                for ni in neighbor_indices:
                    neighbor_tax = tax[tax_ids_sample[ni]].get(rank, "Unknown")
                    if query_tax != "Unknown" and neighbor_tax != "Unknown" and query_tax == neighbor_tax:
                        correct += 1
                    total += 1
            precisions[rank].append(correct / total if total > 0 else 0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"phylum": "#4E7B48", "class": "#7CB159", "order": "#3D549D",
              "family": "#D4782B", "genus": "#AC4431"}
    markers = {"phylum": "o", "class": "s", "order": "D", "family": "^", "genus": "v"}

    ks = list(range(1, k_max + 1))
    for rank in ranks:
        ax.plot(ks, precisions[rank], color=colors[rank], marker=markers[rank],
                linewidth=2, markersize=7, label=f"{rank} ({precisions[rank][-1]:.2f} @k={k_max})")

    # Random baseline
    # Estimate random chance: 1 / (avg number of unique labels per rank)
    baseline = {}
    for rank in ranks:
        labels = [tax[tax_ids_sample[i]].get(rank, "Unknown") for i in range(n_sample)]
        labels = [l for l in labels if l != "Unknown"]
        n_unique = len(set(labels))
        baseline[rank] = 1.0 / n_unique if n_unique > 0 else 0
    ax.axhline(y=0, color="#999", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("k (number of nearest neighbours)", fontsize=12)
    ax.set_ylabel("Retrieval Precision", fontsize=12)
    ax.set_title(f"k-NN Taxonomy Retrieval in NTv3 Embedding Space\n({n_sample:,} OTUs, cosine distance)",
                 fontsize=12, weight="bold")
    ax.legend(fontsize=10, title="Taxonomic Rank (Precision@k=10)", title_fontsize=10)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Baseline annotation
    ax.text(0.99, 0.02,
            "Random baseline ≈ 1/N_unique (too small to show at this scale)",
            transform=ax.transAxes, fontsize=7, ha="right", color="#999", style="italic")

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"ntv3_knn_taxonomy.{fmt}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved %s/ntv3_knn_taxonomy.{png,pdf}", output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("k-NN Retrieval Precision Summary (@k=10)")
    print("=" * 60)
    for rank in ranks:
        print(f"  {rank:10s}: {precisions[rank][-1]:.4f}")
    print()

    results = {rank: precisions[rank][-1] for rank in ranks}
    # Also save per-k curves
    results["k_values"] = ks
    results["precision_curves"] = {r: precisions[r] for r in ranks}
    results["n_sample"] = n_sample
    with open(os.path.join(output_dir, "ntv3_knn_taxonomy.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate NTv3 embeddings (Method 2 + 5)")
    parser.add_argument("--emb_path", default="ntv3_embeddings/otus_99_embeddings.h5")
    parser.add_argument("--fasta_path", default="data/microbeatlas/otus.99.fasta")
    parser.add_argument("--tax_path", default="/data6/zhanghaohong/databases/microbeatals/otus.info.tsv")
    parser.add_argument("--n_sample_seq", type=int, default=5000, help="OTUs for Method 2 (default: 5000)")
    parser.add_argument("--n_pairs", type=int, default=15000, help="Random pairs for Method 2 (default: 15000)")
    parser.add_argument("--n_sample_nn", type=int, default=30000, help="OTUs for Method 5 (default: 30000)")
    parser.add_argument("--k_mer", type=int, default=6, help="k-mer size (default: 6)")
    parser.add_argument("--k_max", type=int, default=10, help="Max k for k-NN (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="figures/ntv3_validation", help="Output directory")
    parser.add_argument("--skip_method2", action="store_true")
    parser.add_argument("--skip_method5", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load embeddings
    embeddings, otu_ids = load_embeddings(args.emb_path)
    id_set = set(otu_ids)

    results = {}

    if not args.skip_method2:
        results["method2"] = method2_seq_emb_corr(
            embeddings, otu_ids, args.fasta_path, args.output_dir,
            n_sample=args.n_sample_seq, n_pairs=args.n_pairs,
            k=args.k_mer, seed=args.seed,
        )

    if not args.skip_method5:
        tax = load_taxonomy(args.tax_path, id_set)
        results["method5"] = method5_knn_taxonomy(
            embeddings, otu_ids, tax, args.output_dir,
            n_sample=args.n_sample_nn, k_max=args.k_max, seed=args.seed,
        )

    # Save combined results
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Outputs saved to {args.output_dir}/")
    if not args.skip_method2:
        print("  ntv3_seq_emb_corr.{png,pdf,json}")
    if not args.skip_method5:
        print("  ntv3_knn_taxonomy.{png,pdf,json}")
    print("  summary.json")


if __name__ == "__main__":
    main()
