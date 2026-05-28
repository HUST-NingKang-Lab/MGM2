#!/usr/bin/env python3
"""Non-circular reliability checks for the taxonomy fallback file.

The fallback file stores taxon means normalized to the NTv3 median norm, so
cosine(file fallback, full taxon mean) is a construction check and is expected
to be almost one.  The main reliability metric here is leave-one-out: hide OTUs
from a taxon, rebuild the same fallback prototype from the remaining OTUs, and
ask whether it is closer to the held-out OTU than to same-rank negative OTUs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (REPO_ROOT / "scripts", SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import build_taxonomy_ntv3_fallback as fb_build

RANKS = list(fb_build.DEFAULT_RANKS)
RANK_COLORS = {
    "kingdom": "#6D2A1F",
    "phylum": "#4E7B48",
    "class": "#7CB159",
    "order": "#3D549D",
    "family": "#D4782B",
    "genus": "#AC4431",
    "species": "#7B1FA2",
}

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def l2_normalize(values: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(values, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return values / denom


def decode_strings(values: Iterable[object]) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray, np.bytes_)):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def load_fallback(path: Path):
    logger.info("Loading fallback file from %s", path)
    with h5py.File(path, "r") as handle:
        taxon_emb = handle["taxon_embeddings"][:].astype(np.float32)
        taxon_keys = decode_strings(handle["taxon_keys"][:])
        taxon_ranks = decode_strings(handle["taxon_ranks"][:])
        taxon_paths = decode_strings(handle["taxon_paths"][:])
        taxon_counts = handle["taxon_counts"][:].astype(int)
        attrs = dict(handle.attrs.items())
    return taxon_emb, taxon_keys, taxon_ranks, taxon_paths, taxon_counts, attrs


def build_matched_taxonomy(args):
    embeddings, otu_ids = fb_build.load_ntv3_index(args.ntv3)
    taxonomy_by_id = fb_build.read_taxonomy_for_ntv3_ids(
        otu_info_path=args.otu_info,
        wanted_ids=set(otu_ids),
        otu_column=args.otu_column,
        taxonomy_column=args.taxonomy_column,
        species_column=args.species_column,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )

    matched_source_indices = []
    matched_ids = []
    rank_to_path_to_otus: Dict[str, Dict[str, List[int]]] = {rank: defaultdict(list) for rank in RANKS}
    for source_idx, otu_id in enumerate(otu_ids):
        taxonomy_pair = taxonomy_by_id.get(otu_id)
        if taxonomy_pair is None:
            continue
        taxonomy, species = taxonomy_pair
        paths = fb_build.parse_taxonomy(taxonomy, RANKS, species)
        if not paths:
            continue
        local_idx = len(matched_source_indices)
        matched_source_indices.append(source_idx)
        matched_ids.append(otu_id)
        for rank, path in paths:
            rank_to_path_to_otus[rank][path].append(local_idx)

    matched = embeddings[np.asarray(matched_source_indices, dtype=np.int64)].astype(np.float32)
    rank_to_all_otus = {
        rank: np.asarray(sorted({idx for indices in path_to_otus.values() for idx in indices}), dtype=np.int64)
        for rank, path_to_otus in rank_to_path_to_otus.items()
    }
    return matched, matched_ids, rank_to_path_to_otus, rank_to_all_otus


def normalized_taxon_mean(embeddings: np.ndarray, indices: Sequence[int], target_norm: float | None) -> np.ndarray:
    mean = embeddings[np.asarray(indices, dtype=np.int64)].mean(axis=0).astype(np.float32)
    if target_norm is not None:
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean * (target_norm / norm)
    return mean.astype(np.float32)


def cosine_one_to_many(query: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
    q = query / max(float(np.linalg.norm(query)), np.finfo(np.float32).eps)
    return matrix_norm @ q.astype(np.float32)


def construction_metrics(args, matched, rank_to_path_to_otus, fallback_emb, taxon_keys, taxon_ranks, taxon_paths, taxon_counts, target_norm):
    rows = []
    for i, (key, rank, path) in enumerate(zip(taxon_keys, taxon_ranks, taxon_paths)):
        otu_indices = rank_to_path_to_otus.get(rank, {}).get(path, [])
        if not otu_indices:
            continue
        expected = normalized_taxon_mean(matched, otu_indices, target_norm)
        diff = fallback_emb[i] - expected
        rel_l2 = float(np.linalg.norm(diff) / max(float(np.linalg.norm(expected)), np.finfo(np.float32).eps))
        cos = float(np.dot(fallback_emb[i], expected) / max(float(np.linalg.norm(fallback_emb[i]) * np.linalg.norm(expected)), np.finfo(np.float32).eps))
        rows.append(
            {
                "taxon_key": key,
                "rank": rank,
                "taxon_path": path,
                "taxon_label": path.split(";")[-1],
                "n_otus": len(otu_indices),
                "fallback_file_count": int(taxon_counts[i]),
                "count_match": int(taxon_counts[i]) == len(otu_indices),
                "file_to_rebuilt_rel_l2": rel_l2,
                "file_to_rebuilt_max_abs": float(np.max(np.abs(diff))),
                "file_to_rebuilt_cos": cos,
            }
        )
    return pd.DataFrame(rows)


def negative_pool_for_taxon(
    rank: str,
    path: str,
    own_indices: Sequence[int],
    rank_to_path_to_otus,
    rank_to_all_otus,
) -> tuple[np.ndarray, str]:
    own = set(int(i) for i in own_indices)
    rank_idx = RANKS.index(rank)
    parts = path.split(";")

    if rank_idx > 0 and len(parts) > 1:
        parent_rank = RANKS[rank_idx - 1]
        parent_path = ";".join(parts[:-1])
        parent_indices = rank_to_path_to_otus[parent_rank].get(parent_path, [])
        sibling = [int(idx) for idx in parent_indices if int(idx) not in own]
        if sibling:
            return np.asarray(sibling, dtype=np.int64), "sibling"

    same_rank = [int(idx) for idx in rank_to_all_otus[rank] if int(idx) not in own]
    return np.asarray(same_rank, dtype=np.int64), "same_rank"


def select_hard_negatives(
    rng,
    pool: np.ndarray,
    pred_norm: np.ndarray,
    matched_norm: np.ndarray,
    n_negatives: int,
    hard_candidate_size: int,
) -> np.ndarray:
    if len(pool) == 0:
        return np.zeros(0, dtype=np.int64)
    if len(pool) > hard_candidate_size:
        candidates = rng.choice(pool, size=hard_candidate_size, replace=False)
    else:
        candidates = pool
    scores = matched_norm[candidates] @ pred_norm
    keep = min(n_negatives, len(candidates))
    hard_local = np.argpartition(-scores, keep - 1)[:keep]
    hard_local = hard_local[np.argsort(-scores[hard_local])]
    return candidates[hard_local]


def loo_reliability(args, matched, rank_to_path_to_otus, rank_to_all_otus, target_norm):
    rng = np.random.default_rng(args.seed)
    matched_norm = l2_normalize(matched)
    rows = []

    for rank in RANKS:
        path_items = [(path, indices) for path, indices in rank_to_path_to_otus[rank].items() if len(indices) >= args.loo_min_otus]
        path_items.sort(key=lambda item: (-len(item[1]), item[0]))
        if args.max_taxa_per_rank and len(path_items) > args.max_taxa_per_rank:
            chosen = rng.choice(len(path_items), size=args.max_taxa_per_rank, replace=False)
            path_items = [path_items[int(i)] for i in chosen]
        logger.info("LOO reliability for %s: %d taxa", rank, len(path_items))

        for path, own_indices_list in path_items:
            own_indices = np.asarray(own_indices_list, dtype=np.int64)
            n_holdout = min(args.loo_per_taxon, len(own_indices))
            holdouts = rng.choice(own_indices, size=n_holdout, replace=False)
            negative_pool, negative_pool_type = negative_pool_for_taxon(rank, path, own_indices, rank_to_path_to_otus, rank_to_all_otus)
            if len(negative_pool) == 0:
                continue

            for holdout in holdouts:
                rest = own_indices[own_indices != holdout]
                if len(rest) == 0:
                    continue
                pred = normalized_taxon_mean(matched, rest, target_norm)
                pred_norm = pred / max(float(np.linalg.norm(pred)), np.finfo(np.float32).eps)
                heldout_cos = float(matched_norm[int(holdout)] @ pred_norm)
                negative_indices = select_hard_negatives(
                    rng,
                    negative_pool,
                    pred_norm,
                    matched_norm,
                    args.n_negatives,
                    args.hard_candidate_size,
                )
                negative_cos = matched_norm[negative_indices] @ pred_norm
                auc = float(np.mean(heldout_cos > negative_cos))
                rank_among_sample = int(1 + np.sum(negative_cos > heldout_cos))
                rows.append(
                    {
                        "rank": rank,
                        "taxon_path": path,
                        "taxon_label": path.split(";")[-1],
                        "n_otus": int(len(own_indices)),
                        "heldout_otu_index": int(holdout),
                        "heldout_cos": heldout_cos,
                        "negative_cos_median": float(np.median(negative_cos)),
                        "negative_cos_q95": float(np.quantile(negative_cos, 0.95)),
                        "delta_vs_negative_median": float(heldout_cos - np.median(negative_cos)),
                        "delta_vs_negative_q95": float(heldout_cos - np.quantile(negative_cos, 0.95)),
                        "auc_vs_same_rank_negatives": auc,
                        "rank_among_sample": rank_among_sample,
                        "n_negatives": int(len(negative_indices)),
                        "negative_pool_type": negative_pool_type,
                    }
                )

    return pd.DataFrame(rows)


def summarize_loo(loo: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if loo.empty:
        summary = pd.DataFrame()
    else:
        summary = loo.groupby("rank").agg(
            n_tests=("heldout_cos", "count"),
            n_taxa=("taxon_path", "nunique"),
            heldout_cos_median=("heldout_cos", "median"),
            heldout_cos_q05=("heldout_cos", lambda x: x.quantile(0.05)),
            negative_cos_median=("negative_cos_median", "median"),
            negative_pool_sibling_rate=("negative_pool_type", lambda x: float(np.mean(x == "sibling"))),
            delta_median=("delta_vs_negative_median", "median"),
            delta_q05=("delta_vs_negative_median", lambda x: x.quantile(0.05)),
            auc_median=("auc_vs_same_rank_negatives", "median"),
            auc_q05=("auc_vs_same_rank_negatives", lambda x: x.quantile(0.05)),
            success_auc_gt_0_5=("auc_vs_same_rank_negatives", lambda x: float(np.mean(x > 0.5))),
            success_auc_gt_0_9=("auc_vs_same_rank_negatives", lambda x: float(np.mean(x > 0.9))),
            top1_sample_rate=("rank_among_sample", lambda x: float(np.mean(x == 1))),
        ).reindex([rank for rank in RANKS if rank in set(loo["rank"])])
    summary.to_csv(output_dir / "fallback_file_loo_rank_summary.csv")
    return summary


def save_overview(construction: pd.DataFrame, loo: pd.DataFrame, summary: pd.DataFrame, args) -> None:
    rank_order = [rank for rank in RANKS if rank in set(construction["rank"]) or (not loo.empty and rank in set(loo["rank"]))]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.8), constrained_layout=True)
    ax_error, ax_auc, ax_delta, ax_success = axes.ravel()

    error_data = [construction.loc[construction["rank"] == rank, "file_to_rebuilt_rel_l2"].dropna().to_numpy() for rank in rank_order]
    bp = ax_error.boxplot(error_data, tick_labels=rank_order, patch_artist=True, showfliers=False)
    for patch, rank in zip(bp["boxes"], rank_order):
        patch.set_facecolor(RANK_COLORS.get(rank, "#777777"))
        patch.set_alpha(0.75)
    ax_error.set_yscale("log")
    ax_error.set_ylabel("relative L2(file, rebuilt full-taxon fallback)")
    ax_error.set_title("Construction check only: file matches the saved fallback rule")
    ax_error.tick_params(axis="x", rotation=30)
    ax_error.grid(axis="y", alpha=0.22)

    if loo.empty:
        for ax in [ax_auc, ax_delta, ax_success]:
            ax.text(0.5, 0.5, "No LOO cases", ha="center", va="center")
            ax.set_axis_off()
    else:
        loo_ranks = [rank for rank in RANKS if rank in set(loo["rank"])]
        auc_data = [loo.loc[loo["rank"] == rank, "auc_vs_same_rank_negatives"].dropna().to_numpy() for rank in loo_ranks]
        bp_auc = ax_auc.boxplot(auc_data, tick_labels=loo_ranks, patch_artist=True, showfliers=False)
        for patch, rank in zip(bp_auc["boxes"], loo_ranks):
            patch.set_facecolor(RANK_COLORS.get(rank, "#777777"))
            patch.set_alpha(0.75)
        ax_auc.axhline(0.5, color="#333333", linestyle="--", linewidth=1, label="random")
        ax_auc.set_ylim(0, 1.02)
        ax_auc.set_ylabel("LOO AUC vs same-rank negative OTUs")
        ax_auc.set_title("Non-circular fallback reliability")
        ax_auc.tick_params(axis="x", rotation=30)
        ax_auc.legend(fontsize=8)
        ax_auc.grid(axis="y", alpha=0.22)

        delta_data = [loo.loc[loo["rank"] == rank, "delta_vs_negative_median"].dropna().to_numpy() for rank in loo_ranks]
        bp_delta = ax_delta.boxplot(delta_data, tick_labels=loo_ranks, patch_artist=True, showfliers=False)
        for patch, rank in zip(bp_delta["boxes"], loo_ranks):
            patch.set_facecolor(RANK_COLORS.get(rank, "#777777"))
            patch.set_alpha(0.75)
        ax_delta.axhline(0.0, color="#333333", linestyle="--", linewidth=1)
        ax_delta.set_ylabel("held-out cosine - median negative cosine")
        ax_delta.set_title("LOO separation over same-rank negatives")
        ax_delta.tick_params(axis="x", rotation=30)
        ax_delta.grid(axis="y", alpha=0.22)

        x = np.arange(len(summary.index))
        width = 0.36
        ax_success.bar(x - width / 2, summary["success_auc_gt_0_5"], width=width, label="AUC > 0.5", color="#4E79A7", alpha=0.9)
        ax_success.bar(x + width / 2, summary["success_auc_gt_0_9"], width=width, label="AUC > 0.9", color="#2E7D32", alpha=0.9)
        ax_success.set_xticks(x)
        ax_success.set_xticklabels(summary.index, rotation=30, ha="right")
        ax_success.set_ylim(0, 1.02)
        ax_success.set_ylabel("fraction of LOO tests")
        ax_success.set_title("How often fallback beats same-rank negatives")
        ax_success.legend(fontsize=8)
        ax_success.grid(axis="y", alpha=0.22)

    for fmt in ["png", "pdf"]:
        out = args.output_dir / f"fallback_file_loo_reliability_overview.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", out)
    plt.close(fig)


def save_top20_genus_plot(loo: pd.DataFrame, args) -> None:
    genus = loo[loo["rank"] == "genus"].copy()
    if genus.empty:
        return
    grouped = genus.groupby("taxon_path").agg(
        taxon_label=("taxon_label", "first"),
        n_otus=("n_otus", "max"),
        auc_median=("auc_vs_same_rank_negatives", "median"),
        delta_median=("delta_vs_negative_median", "median"),
    )
    top = grouped.sort_values("n_otus", ascending=False).head(args.n_genera)
    top.to_csv(args.output_dir / "fallback_file_top20_genus_loo_metrics.csv")

    fig, ax1 = plt.subplots(figsize=(11.5, 5.8), constrained_layout=True)
    x = np.arange(len(top))
    ax1.bar(x, top["auc_median"], color="#4E79A7", alpha=0.9, label="median AUC")
    ax1.axhline(0.5, color="#333333", linestyle="--", linewidth=1)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("median LOO AUC")
    ax1.set_xticks(x)
    ax1.set_xticklabels(top["taxon_label"], rotation=75, ha="right", fontsize=8)
    ax1.set_title(f"Top {len(top)} genera: non-circular LOO fallback reliability")
    ax1.grid(axis="y", alpha=0.22)
    for fmt in ["png", "pdf"]:
        out = args.output_dir / f"fallback_file_top20_genus_loo_auc.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", out)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate taxonomy fallback file reliability with non-circular leave-one-out metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ntv3", type=Path, default=Path("ntv3_embeddings/otus_99_embeddings.h5"))
    parser.add_argument("--fallback", type=Path, default=Path("ntv3_embeddings/otus_99_taxonomy_fallback.h5"))
    parser.add_argument("--otu_info", type=Path, default=Path("/data6/zhanghaohong/databases/microbeatals/otus.info.tsv"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/fallback/reliability_summary"))
    parser.add_argument("--otu_column", default="OTU")
    parser.add_argument("--taxonomy_column", default="Tax")
    parser.add_argument("--species_column", default="RepSpecies")
    parser.add_argument("--chunk_size", type=int, default=200000)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--loo_min_otus", type=int, default=3)
    parser.add_argument("--loo_per_taxon", type=int, default=1)
    parser.add_argument("--n_negatives", type=int, default=128)
    parser.add_argument("--hard_candidate_size", type=int, default=4096)
    parser.add_argument("--max_taxa_per_rank", type=int, default=300, help="Taxa sampled per rank for hard-negative LOO; 0 means all eligible taxa")
    parser.add_argument("--n_genera", type=int, default=20, help="Only used for the companion top-genus LOO plot")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fallback_emb, taxon_keys, taxon_ranks, taxon_paths, taxon_counts, attrs = load_fallback(args.fallback)
    matched, _, rank_to_path_to_otus, rank_to_all_otus = build_matched_taxonomy(args)
    target_norm_attr = attrs.get("target_norm", "")
    target_norm = None if target_norm_attr == "" else float(target_norm_attr)

    construction = construction_metrics(
        args,
        matched,
        rank_to_path_to_otus,
        fallback_emb,
        taxon_keys,
        taxon_ranks,
        taxon_paths,
        taxon_counts,
        target_norm,
    )
    construction.to_csv(args.output_dir / "fallback_file_construction_check.csv", index=False)

    loo = loo_reliability(args, matched, rank_to_path_to_otus, rank_to_all_otus, target_norm)
    loo.to_csv(args.output_dir / "fallback_file_loo_reliability_metrics.csv", index=False)
    summary = summarize_loo(loo, args.output_dir)
    save_overview(construction, loo, summary, args)
    save_top20_genus_plot(loo, args)

    print(f"Done. Outputs written to {args.output_dir}")
    print("\nConstruction check, max relative L2 by rank:")
    print(construction.groupby("rank")["file_to_rebuilt_rel_l2"].max().reindex([rank for rank in RANKS if rank in set(construction["rank"])]).to_string())
    print("\nLOO reliability summary:")
    print(summary[["n_tests", "negative_pool_sibling_rate", "auc_median", "auc_q05", "delta_median", "success_auc_gt_0_9"]].to_string())


if __name__ == "__main__":
    main()
