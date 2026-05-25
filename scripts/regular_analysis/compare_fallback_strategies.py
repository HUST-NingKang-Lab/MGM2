#!/usr/bin/env python3
"""
Compare fallback strategies for OTU embedding approximation.

Strategies:
  A. Mean              — taxon mean (current baseline)
  B. Robust Mean       — taxon mean with outlier down-weighting (no query needed)
  C. Hier. Residual    — coarse_mean + Σ(fine_mean - coarse_mean)
  D. Nearest Neighbor  — single random OTU from same taxon
  E. Shrinkage         — (taxon_mean*N + parent_mean*Np) / (N + Np)

Evaluation: leave-one-out cosine similarity + retrieval precision.
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
STRATEGY_COLORS = {"Mean": "#7CB5EC", "Robust": "#4363D8", "Residual": "#4E7B48",
                   "Nearest": "#D4782B", "Shrinkage": "#AC4431"}


def load_data(args):
    logger.info("Loading data...")
    with h5py.File(args.ntv3, "r") as f:
        ntv3_emb = f["embeddings"][:].astype(np.float32)
        ntv3_ids = []
        for raw in f["otu_ids"][:]:
            t = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            t = t.strip().strip("'").strip('"')
            ntv3_ids.append(t)

    df = pd.read_csv(args.otu_info, sep="\t", usecols=["OTU", "Tax"])
    otu_to_tax = {}
    for _, row in df.iterrows():
        otu_str = str(row["OTU"]); tax_str = str(row["Tax"]) if pd.notna(row["Tax"]) else ""
        for chunk in otu_str.split(";"):
            if chunk.strip().startswith("99_") and tax_str:
                otu_to_tax[chunk.strip()] = tax_str; break

    idx_m, ids_m = [], []
    for i, oid in enumerate(ntv3_ids):
        if oid in otu_to_tax: idx_m.append(i); ids_m.append(oid)

    scaler = StandardScaler()
    matched_emb = scaler.fit_transform(ntv3_emb[idx_m])

    # rank → path → list[embedding_idx]
    r2p2o = {r: defaultdict(list) for r in RANKS}
    for idx, oid in enumerate(ids_m):
        parts = otu_to_tax[oid].replace("; ", ";").split(";")
        for lvl, rank in enumerate(RANKS):
            if lvl < len(parts):
                r2p2o[rank][";".join(parts[:lvl+1])].append(idx)

    return {"matched_emb": matched_emb, "otu_ids": ids_m, "otu_to_tax": otu_to_tax,
            "r2p2o": r2p2o, "scaler": scaler}


def get_loo_data(data, rank, min_otus=5, n_sample=500, seed=42):
    """Sample leave-one-out test cases: left-out OTU + rest of taxon."""
    rng = np.random.default_rng(seed)
    paths = [p for p, idxs in data["r2p2o"][rank].items() if len(idxs) >= min_otus]
    if len(paths) < 3:
        return None, None, None

    paths = rng.choice(paths, size=min(n_sample, len(paths)), replace=False)
    loo_idx_list, rest_idx_list, paths_list = [], [], []
    for p in paths:
        idxs = data["r2p2o"][rank][p]
        loo = rng.choice(idxs)
        rest = [i for i in idxs if i != loo]
        if rest:
            loo_idx_list.append(loo); rest_idx_list.append(rest); paths_list.append(p)
    return loo_idx_list, rest_idx_list, paths_list


def strategy_mean(loo_emb, rest_embs, rest_idx, data, rank, path):
    return rest_embs.mean(axis=0), "Mean"

def strategy_robust(loo_emb, rest_embs, rest_idx, data, rank, path):
    """Weighted mean down-weighting outliers within the taxon.
    Weights = cosine similarity to taxon centroid (taxonomy-only, no query)."""
    if len(rest_embs) <= 2:
        return rest_embs.mean(axis=0), "Robust"
    centroid = rest_embs.mean(axis=0)
    sim = cosine_similarity(centroid.reshape(1, -1), rest_embs)[0]
    # Softmax-style: exp(sim/T) with T=0.1 to sharpen
    weights = np.exp((sim - sim.max()) / 0.1)
    weights = weights / (weights.sum() + 1e-10)
    return (rest_embs * weights[:, None]).sum(axis=0), "Robust"

def strategy_residual(loo_emb, rest_embs, rest_idx, data, rank, path):
    """coarse_mean + Σ(fine_mean - coarse_mean)"""
    parts = path.split(";")
    rank_idx = RANKS.index(rank)
    if rank_idx < 2:
        return rest_embs.mean(axis=0), "Residual"

    emb_list = []
    for r_i, r in enumerate(RANKS[:rank_idx + 1]):
        p = ";".join(parts[:r_i + 1])
        idxs = data["r2p2o"][r][p]
        if idxs:
            emb_list.append(data["matched_emb"][idxs].mean(axis=0))
    # Sum residuals: fine_mean - parent_mean, then + base
    base = emb_list[0]  # kingdom
    result = base.copy()
    for i in range(1, len(emb_list)):
        result = result + (emb_list[i] - emb_list[i - 1])
    return result, "Residual"

def strategy_nearest(loo_emb, rest_embs, rest_idx, data, rank, path):
    rng = np.random.default_rng(hash(path) % 2**32)
    return rest_embs[rng.integers(len(rest_embs))], "Nearest"

def strategy_shrinkage(loo_emb, rest_embs, rest_idx, data, rank, path):
    parts = path.split(";")
    rank_idx = RANKS.index(rank)
    taxon_mean = rest_embs.mean(axis=0); Nt = len(rest_embs)
    if rank_idx == 0:
        return taxon_mean, "Shrinkage"
    parent_idx = data["r2p2o"][RANKS[rank_idx - 1]][";".join(parts[:-1])]
    parent_mean = data["matched_emb"][parent_idx].mean(axis=0); Np = len(parent_idx)
    return ((taxon_mean * Nt + parent_mean * Np) / (Nt + Np)), "Shrinkage"


def evaluate_strategies(data, output_dir, seed=42):
    ranks_eval = ["phylum", "class", "order", "family", "genus"]
    strategies = [
        ("Mean",      strategy_mean),
        ("Robust",    strategy_robust),
        ("Residual",  strategy_residual),
        ("Nearest",   strategy_nearest),
        ("Shrinkage", strategy_shrinkage),
    ]

    all_results = []

    for rank in ranks_eval:
        loo_idx, rest_idx, paths = get_loo_data(data, rank, seed=seed)
        if not loo_idx:
            continue

        rank_results = {sname: [] for sname, _ in strategies}
        for li, ri, p in tqdm(zip(loo_idx, rest_idx, paths), desc=f"LOO {rank}", total=len(loo_idx)):
            loo_emb = data["matched_emb"][li]
            rest_embs = data["matched_emb"][ri]

            for sname, sfunc in strategies:
                pred, _ = sfunc(loo_emb, rest_embs, ri, data, rank, p)
                sim = float(cosine_similarity(loo_emb.reshape(1, -1), pred.reshape(1, -1))[0, 0])
                rank_results[sname].append(sim)

        for sname in rank_results:
            sims = np.array(rank_results[sname])
            all_results.append({"rank": rank, "strategy": sname,
                                "mean_sim": np.mean(sims), "std_sim": np.std(sims),
                                "median_sim": np.median(sims), "n": len(sims)})
            logger.info("  %s/%s: mean=%.4f median=%.4f", rank, sname, np.mean(sims), np.median(sims))

    return pd.DataFrame(all_results)


def fig_loo_comparison(df, output_dir):
    ranks = sorted(df["rank"].unique())
    strategies = sorted(df["strategy"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: bar chart grouped by rank
    x = np.arange(len(ranks))
    n_strat = len(strategies)
    w = 0.8 / n_strat
    for i, sname in enumerate(strategies):
        sub = df[df["strategy"] == sname]
        vals = [sub[sub["rank"] == r]["mean_sim"].values[0] if r in sub["rank"].values else np.nan
                for r in ranks]
        bars = axes[0].bar(x + i * w - 0.4 + w / 2, vals, w, label=sname, color=STRATEGY_COLORS[sname],
                           edgecolor="#333", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                             f"{v:.3f}", ha="center", fontsize=7, rotation=90)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ranks, fontsize=10)
    axes[0].set_ylabel("Mean LOO cosine similarity", fontsize=11)
    axes[0].set_title("Leave-One-Out Reconstruction by Strategy", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    # Right: line plot by strategy across ranks
    for sname in strategies:
        sub = df[df["strategy"] == sname].sort_values("rank")
        axes[1].plot(range(len(sub)), sub["mean_sim"].values, "-o",
                     color=STRATEGY_COLORS[sname], linewidth=2, markersize=6, label=sname)
    axes[1].set_xticks(range(len(sub)))
    axes[1].set_xticklabels(sub["rank"].tolist(), fontsize=10)
    axes[1].set_ylabel("Mean LOO cosine similarity", fontsize=11)
    axes[1].set_title("Strategy Performance across Ranks", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    fig.suptitle("Fallback Strategy Comparison (leave-one-out, standardized cosine similarity)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_strategy_loo.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    df.to_csv(os.path.join(output_dir, "fallback_strategy_loo.csv"), index=False)
    logger.info("Saved fallback_strategy_loo.{png,pdf,csv}")


def fig_retrieval_comparison(data, output_dir, k=10, seed=42):
    """Use each strategy to generate fallback embedding, retrieve nearest OTUs."""
    rng = np.random.default_rng(seed)
    ranks_eval = ["phylum", "class", "order", "family", "genus"]
    strategies = [
        ("Mean",      strategy_mean),
        ("Robust",    strategy_robust),
        ("Residual",  strategy_residual),
        ("Nearest",   strategy_nearest),
        ("Shrinkage", strategy_shrinkage),
    ]

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(data["matched_emb"])

    results = []
    for rank in ranks_eval:
        loo_idx, rest_idx, paths = get_loo_data(data, rank, n_sample=200, seed=seed)
        if not loo_idx:
            continue

        strategy_prec = {s: [] for s, _ in strategies}
        for li, ri, p in tqdm(zip(loo_idx, rest_idx, paths), desc=f"Retrieval {rank}", total=len(loo_idx)):
            loo_emb = data["matched_emb"][li]
            rest_embs = data["matched_emb"][ri]

            for sname, sfunc in strategies:
                fb_emb, _ = sfunc(loo_emb, rest_embs, ri, data, rank, p)
                dists, indices = nn.kneighbors(fb_emb.reshape(1, -1), n_neighbors=k + 1)
                # What fraction of retrieved OTUs are in the same taxon?
                taxon_set = set(ri)
                hits = sum(1 for ni in indices[0] if ni in taxon_set)
                strategy_prec[sname].append(hits / k)

        for sname in strategy_prec:
            precs = np.array(strategy_prec[sname])
            results.append({"rank": rank, "strategy": sname,
                            f"precision@{k}": np.mean(precs), "std": np.std(precs)})
            logger.info("  %s/%s: precision@%d=%.4f", rank, sname, k, np.mean(precs))

    df_ret = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ranks_plot = sorted(df_ret["rank"].unique())
    strats_plot = sorted(df_ret["strategy"].unique())
    x = np.arange(len(ranks_plot))
    n_s = len(strats_plot)
    w = 0.8 / n_s
    for i, sname in enumerate(strats_plot):
        sub = df_ret[df_ret["strategy"] == sname]
        vals = [sub[sub["rank"] == r][f"precision@{k}"].values[0] if r in sub["rank"].values else 0
                for r in ranks_plot]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=sname, color=STRATEGY_COLORS[sname],
                      edgecolor="#333", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ranks_plot, fontsize=10)
    ax.set_ylabel(f"Precision@{k}", fontsize=11)
    ax.set_title(f"Fallback Retrieval Precision@{k} by Strategy\n(fallback as query → retrieve OTUs → check same-taxon)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for fmt in ["png", "pdf"]:
        fig.savefig(os.path.join(output_dir, f"fallback_strategy_retrieval.{fmt}"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    df_ret.to_csv(os.path.join(output_dir, "fallback_strategy_retrieval.csv"), index=False)
    logger.info("Saved fallback_strategy_retrieval.{png,pdf,csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntv3", default="ntv3_embeddings/otus_99_embeddings.h5")
    parser.add_argument("--otu_info", default="/data6/zhanghaohong/databases/microbeatals/otus.info.tsv")
    parser.add_argument("--output_dir", default="figures/fallback/fallback_strategy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_data(args)

    df = evaluate_strategies(data, args.output_dir)
    fig_loo_comparison(df, args.output_dir)
    fig_retrieval_comparison(data, args.output_dir)

    print(f"\nDone. Output saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
