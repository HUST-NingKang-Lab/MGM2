#!/usr/bin/env python3
"""Evaluate taxonomy fallback coverage on an abundance-table HDF5.

Example:
    python scripts/evaluate_taxonomy_fallback_coverage.py \
        --input_h5 /data6/zhanghaohong/projects/MGM2Benchmarks/MGnify_mst/expert/test_inputs.h5_raw.h5 \
        --key genus \
        --fallback_h5 ntv3_embeddings/otus_99_taxonomy_fallback.h5 \
        --output_dir outputs_analysis/taxonomy_fallback_mgnify_mst
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import pandas as pd

logger = logging.getLogger(__name__)

RANK_ORDER = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
KNOWN_PREFIXES = {"sk", "k", "d", "p", "c", "o", "f", "g", "s"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate how many taxonomy features can be resolved by fallback NTv3 taxon prototypes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/evaluate_taxonomy_fallback_coverage.py "
            "--input_h5 /data6/zhanghaohong/projects/MGM2Benchmarks/MGnify_mst/expert/test_inputs.h5_raw.h5 "
            "--key genus --fallback_h5 ntv3_embeddings/otus_99_taxonomy_fallback.h5 "
            "--output_dir outputs_analysis/taxonomy_fallback_mgnify_mst"
        ),
    )
    parser.add_argument("--input_h5", type=Path, required=True, help="Benchmark HDF5 abundance table")
    parser.add_argument("--key", default="genus", help="Top-level HDF5 group key containing axis1 taxonomy labels")
    parser.add_argument("--fallback_h5", type=Path, required=True, help="Taxonomy fallback HDF5 built by build_taxonomy_ntv3_fallback.py")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for coverage reports")
    parser.add_argument("--max_examples", type=int, default=50, help="Number of unmatched examples to save")
    return parser.parse_args()


def decode_h5_strings(values) -> List[str]:
    decoded = []
    for value in values:
        if isinstance(value, (bytes, bytearray)):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def clean_taxon_name(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"unclassified", "unknown", "none", "nan"}:
        return ""
    return value


def parse_prefixed_taxonomy(label: str) -> Dict[str, str]:
    """Parse MGnify/SILVA-style labels such as sk__Bacteria;k__;p__Firmicutes;g__Bacillus."""
    label = label.strip()
    if not label or label.lower() == "unclassified":
        return {}

    levels: List[str] = []
    for token in label.split(";"):
        token = token.strip()
        if "__" in token:
            prefix, name = token.split("__", 1)
            if prefix.strip().lower() not in KNOWN_PREFIXES:
                continue
            name = clean_taxon_name(name)
            if name:
                levels.append(name)
        else:
            name = clean_taxon_name(token)
            if name:
                levels.append(name)
    return {rank: value for rank, value in zip(RANK_ORDER, levels)}


def parse_plain_taxonomy(label: str) -> Dict[str, str]:
    """Parse plain MicrobeAtlas-style taxonomy: Bacteria;Firmicutes;Bacilli;..."""
    parts = [clean_taxon_name(part) for part in label.split(";")]
    parts = [part for part in parts if part]
    return {rank: value for rank, value in zip(RANK_ORDER, parts)}


def parse_taxonomy_label(label: str) -> Dict[str, str]:
    if "__" in label:
        return parse_prefixed_taxonomy(label)
    return parse_plain_taxonomy(label)


def build_cumulative_keys(rank_to_name: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return fallback keys from broad to specific while preserving missing-rank gaps."""
    keys: List[Tuple[str, str]] = []
    path: List[str] = []
    for rank in RANK_ORDER:
        name = rank_to_name.get(rank)
        if not name:
            continue
        path.append(name)
        keys.append((rank, f"{rank}|{';'.join(path)}"))
    return keys


def load_taxon_keys(fallback_h5: Path) -> set:
    with h5py.File(fallback_h5, "r") as handle:
        if "taxon_keys" not in handle:
            raise KeyError(f"{fallback_h5} does not contain dataset 'taxon_keys'")
        return set(decode_h5_strings(handle["taxon_keys"][:]))


def load_feature_labels(input_h5: Path, key: str) -> List[str]:
    axis1_path = f"{key}/axis1"
    with h5py.File(input_h5, "r") as handle:
        if axis1_path not in handle:
            available = list(handle.keys())
            raise KeyError(f"Missing dataset {axis1_path}; available top-level keys: {available}")
        return decode_h5_strings(handle[axis1_path][:])


def resolve_label(label: str, taxon_keys: set) -> Tuple[Optional[str], Optional[str], List[Tuple[str, str]]]:
    rank_to_name = parse_taxonomy_label(label)
    candidate_keys = build_cumulative_keys(rank_to_name)
    for rank, key in reversed(candidate_keys):
        if key in taxon_keys:
            return rank, key, candidate_keys
    return None, None, candidate_keys


def evaluate_coverage(labels: Sequence[str], taxon_keys: set, max_examples: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    rows = []
    rank_counts = Counter()
    unmatched_examples = []
    parsed_count = 0

    for label in labels:
        matched_rank, matched_key, candidate_keys = resolve_label(label, taxon_keys)
        if candidate_keys:
            parsed_count += 1
        if matched_rank is None:
            rank_counts["unmatched"] += 1
            if len(unmatched_examples) < max_examples:
                unmatched_examples.append(
                    {
                        "feature_label": label,
                        "most_specific_candidate": candidate_keys[-1][1] if candidate_keys else "",
                        "broadest_candidate": candidate_keys[0][1] if candidate_keys else "",
                    }
                )
        else:
            rank_counts[matched_rank] += 1
        rows.append(
            {
                "feature_label": label,
                "matched_rank": matched_rank or "unmatched",
                "matched_key": matched_key or "",
                "num_candidate_keys": len(candidate_keys),
            }
        )

    total = len(labels)
    summary = {
        "num_features": total,
        "parsed_features": parsed_count,
        "parsed_fraction": parsed_count / max(1, total),
        "matched_features": total - rank_counts["unmatched"],
        "matched_fraction": (total - rank_counts["unmatched"]) / max(1, total),
        "rank_counts": dict(rank_counts),
    }
    return pd.DataFrame(rows), summary, pd.DataFrame(unmatched_examples)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_feature_labels(args.input_h5, args.key)
    taxon_keys = load_taxon_keys(args.fallback_h5)
    coverage, summary, unmatched = evaluate_coverage(labels, taxon_keys, args.max_examples)

    coverage_path = args.output_dir / "taxonomy_fallback_feature_coverage.tsv"
    summary_path = args.output_dir / "taxonomy_fallback_coverage_summary.json"
    unmatched_path = args.output_dir / "taxonomy_fallback_unmatched_examples.tsv"

    coverage.to_csv(coverage_path, sep="\t", index=False)
    unmatched.to_csv(unmatched_path, sep="\t", index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved feature coverage to %s", coverage_path)
    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved unmatched examples to %s", unmatched_path)
    logger.info("Matched %s / %s features (%.2f%%)", summary["matched_features"], summary["num_features"], 100 * summary["matched_fraction"])
    logger.info("Rank counts: %s", summary["rank_counts"])


if __name__ == "__main__":
    main()
