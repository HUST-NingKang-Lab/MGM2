#!/usr/bin/env python3
"""Build a one-round pretraining sample/token manifest for MGM2-xlarge SAE training.

The script creates a sample-level split and a token-level manifest containing
CLS and all valid OTU tokens for each sampled pretraining sample. It does not
extract xlarge activations; later extraction should consume tokens.csv.gz.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


DEFAULT_DATA_FILE = Path("data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.pkl")
DEFAULT_METADATA = Path("/data6/zhanghaohong/databases/microbeatals/samples.env.info.tsv")
DEFAULT_OUTPUT_DIR = Path("data/sae/pretrain_xlarge_sample50k")
DEFAULT_PRIMARY_QUOTAS = {
    "animal": 10_000,
    "aquatic": 10_000,
    "soil": 10_000,
    "plant": 10_000,
    "<missing>": 5_000,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--metadata_file", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quota_scale", type=float, default=1.0, help="Scale default primary quotas and WGS supplement")
    parser.add_argument("--animal_quota", type=int, default=None)
    parser.add_argument("--aquatic_quota", type=int, default=None)
    parser.add_argument("--soil_quota", type=int, default=None)
    parser.add_argument("--plant_quota", type=int, default=None)
    parser.add_argument("--missing_quota", type=int, default=None)
    parser.add_argument("--wgs_supplement", type=int, default=5_000)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--test_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_bucket_fraction", type=float, default=0.25)
    parser.add_argument("--exclude_cls", action="store_true", help="Do not include CLS token rows")
    return parser.parse_args()


def primary_quotas_from_args(args) -> Dict[str, int]:
    scaled = {key: int(round(value * args.quota_scale)) for key, value in DEFAULT_PRIMARY_QUOTAS.items()}
    overrides = {
        "animal": args.animal_quota,
        "aquatic": args.aquatic_quota,
        "soil": args.soil_quota,
        "plant": args.plant_quota,
        "<missing>": args.missing_quota,
    }
    for key, value in overrides.items():
        if value is not None:
            scaled[key] = int(value)
    return scaled


def normalize_sample_id(value) -> str:
    return str(value).strip()


def parse_environment(raw: str) -> tuple[str, str]:
    raw = str(raw or "").strip()
    if not raw:
        return "<missing>", "<missing>"
    first_path = raw.split("|")[0]
    tokens = [token.strip() for token in first_path.split(";") if token.strip()]
    if not tokens:
        return "<missing>", "<missing>"
    primary = tokens[0]
    first2 = ";".join(tokens[:2]) if len(tokens) > 1 else primary
    return primary, first2


def load_pretrain_data(path: Path) -> Dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_metadata_for_samples(metadata_path: Path, sample_ids: set[str]) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with metadata_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            sample_id = normalize_sample_id(row["MAP_SID"])
            if sample_id not in sample_ids:
                continue
            primary, first2 = parse_environment(row.get("Environments", ""))
            rows[sample_id] = {
                "sample_id": sample_id,
                "primary_env": primary,
                "env_first2": first2,
                "technology": row.get("Technology", ""),
                "keywords": row.get("Keywords", ""),
                "project": row.get("Project", ""),
                "institution": row.get("Institution", ""),
            }
    missing = sample_ids.difference(rows)
    if missing:
        raise ValueError(f"metadata missing for {len(missing)} pretraining samples")
    return rows


def _shuffle_copy(items: Sequence[Dict], rng: random.Random) -> List[Dict]:
    copied = list(items)
    rng.shuffle(copied)
    return copied


def select_env_stratified(
    candidates: Sequence[Dict],
    target: int,
    rng: random.Random,
    max_bucket_fraction: float,
) -> List[Dict]:
    if len(candidates) <= target:
        return list(candidates)

    bucket_cap = max(1, int(round(target * max_bucket_fraction)))
    weighted_target = int(round(target * 0.7))
    selected: List[Dict] = []
    selected_ids = set()
    bucket_counts = Counter()

    for row in _shuffle_copy(candidates, rng):
        if len(selected) >= weighted_target:
            break
        bucket = row["env_first2"]
        if bucket_counts[bucket] >= bucket_cap:
            continue
        selected.append(row)
        selected_ids.add(row["sample_id"])
        bucket_counts[bucket] += 1

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for row in candidates:
        if row["sample_id"] not in selected_ids:
            buckets[row["env_first2"]].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)

    bucket_names = list(buckets)
    rng.shuffle(bucket_names)
    while len(selected) < target and bucket_names:
        progressed = False
        for bucket in list(bucket_names):
            rows = buckets[bucket]
            if not rows:
                bucket_names.remove(bucket)
                continue
            selected_row = rows.pop()
            selected.append(selected_row)
            selected_ids.add(selected_row["sample_id"])
            progressed = True
            if len(selected) >= target:
                break
        if not progressed:
            break

    if len(selected) < target:
        for row in _shuffle_copy(candidates, rng):
            if row["sample_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["sample_id"])
            if len(selected) >= target:
                break
    return selected


def assign_splits(rows: List[Dict], val_fraction: float, test_fraction: float, rng: random.Random) -> None:
    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_group[row["quota_group"]].append(row)
    for group_rows in by_group.values():
        rng.shuffle(group_rows)
        n = len(group_rows)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        for idx, row in enumerate(group_rows):
            if idx < n_test:
                row["split"] = "test"
            elif idx < n_test + n_val:
                row["split"] = "val"
            else:
                row["split"] = "train"


def build_sample_manifest(metadata: Dict[str, Dict[str, str]], args) -> List[Dict]:
    rng = random.Random(args.seed)
    selected: List[Dict] = []
    selected_ids = set()
    primary_quotas = primary_quotas_from_args(args)

    by_primary: Dict[str, List[Dict]] = defaultdict(list)
    for row in metadata.values():
        by_primary[row["primary_env"]].append(row)

    for primary, quota in primary_quotas.items():
        group_selected = select_env_stratified(
            by_primary.get(primary, []),
            quota,
            rng,
            args.max_bucket_fraction,
        )
        for row in group_selected:
            copied = dict(row)
            copied["quota_group"] = primary
            selected.append(copied)
            selected_ids.add(copied["sample_id"])

    remaining_wgs = [
        row
        for row in metadata.values()
        if row["sample_id"] not in selected_ids and row["technology"] == "WGS"
    ]
    non_animal_wgs = [row for row in remaining_wgs if row["primary_env"] != "animal"]
    animal_wgs = [row for row in remaining_wgs if row["primary_env"] == "animal"]
    wgs_selected = select_env_stratified(
        non_animal_wgs,
        min(args.wgs_supplement, len(non_animal_wgs)),
        rng,
        args.max_bucket_fraction,
    )
    if len(wgs_selected) < args.wgs_supplement:
        selected_wgs_ids = {row["sample_id"] for row in wgs_selected}
        needed = args.wgs_supplement - len(wgs_selected)
        fill = [row for row in animal_wgs if row["sample_id"] not in selected_wgs_ids]
        wgs_selected.extend(select_env_stratified(fill, needed, rng, args.max_bucket_fraction))

    for row in wgs_selected:
        copied = dict(row)
        copied["quota_group"] = "wgs_supplement"
        selected.append(copied)
        selected_ids.add(copied["sample_id"])

    assign_splits(selected, args.val_fraction, args.test_fraction, rng)
    return selected


def reverse_otu_map(otu_id_to_idx: Dict) -> Dict[int, str]:
    return {int(idx): str(otu_id) for otu_id, idx in otu_id_to_idx.items()}


def valid_token_arrays(data: Dict, sample_index: int):
    import numpy as np

    input_ids = np.asarray(data["input_ids"][sample_index])
    attention_mask = np.asarray(data["attention_mask"][sample_index])
    processed_abundances = np.asarray(data["abundance_values"][sample_index], dtype=np.float32)
    valid = (attention_mask == 1) & (input_ids != -1)
    positions = np.flatnonzero(valid)
    # Pretraining abundance_values are already log10(abundance) + 2, clipped to [-2, 2].
    # Convert back to a positive relative-abundance scale for metadata.
    relative_abundances = np.power(10.0, processed_abundances[positions] - 2.0).astype(np.float32)
    return (
        positions,
        input_ids[positions].astype(np.int64),
        processed_abundances[positions].astype(np.float32),
        relative_abundances,
    )


def build_token_rows_for_sample(
    data: Dict,
    sample_index: int,
    include_cls: bool,
) -> List[Dict]:
    positions, otu_indices, processed_abundances, relative_abundances = valid_token_arrays(data, sample_index)
    n_valid = len(positions)
    rows = []
    if include_cls:
        rows.append(
            {
                "token_position": 0,
                "token_type": "cls",
                "otu_index": -1,
                "abundance": 0.0,
                "processed_abundance": 0.0,
                "abundance_rank": 0,
            }
        )

    if n_valid == 0:
        return rows

    order_desc = sorted(range(n_valid), key=lambda idx: (-float(relative_abundances[idx]), int(positions[idx])))
    abundance_rank_by_local = {local_idx: rank for rank, local_idx in enumerate(order_desc, start=1)}
    for local_idx in range(n_valid):
        rows.append(
            {
                "token_position": int(positions[local_idx]),
                "token_type": "otu",
                "otu_index": int(otu_indices[local_idx]),
                "abundance": float(relative_abundances[local_idx]),
                "processed_abundance": float(processed_abundances[local_idx]),
                "abundance_rank": int(abundance_rank_by_local[local_idx]),
            }
        )
    return rows


def write_outputs(data: Dict, sample_manifest: List[Dict], args) -> Dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = [normalize_sample_id(sample_id) for sample_id in data["samples"]]
    sample_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
    idx_to_otu_id = reverse_otu_map(data["otu_id_to_idx"])
    include_cls = not args.exclude_cls

    sample_fields = [
        "sample_id",
        "sample_index",
        "split",
        "quota_group",
        "primary_env",
        "env_first2",
        "technology",
        "project",
        "institution",
        "keywords",
        "nonzero_otu_count",
        "selected_token_count",
    ]
    token_fields = [
        "sample_id",
        "sample_index",
        "split",
        "quota_group",
        "primary_env",
        "env_first2",
        "technology",
        "token_type",
        "token_position",
        "otu_index",
        "otu_id",
        "abundance",
        "processed_abundance",
        "abundance_rank",
    ]

    split_counts = Counter()
    group_counts = Counter()
    token_type_counts = Counter()
    token_count_total = 0
    with (output_dir / "samples.csv").open("w", newline="") as sample_handle, gzip.open(
        output_dir / "tokens.csv.gz", "wt", newline=""
    ) as token_handle:
        sample_writer = csv.DictWriter(sample_handle, fieldnames=sample_fields)
        token_writer = csv.DictWriter(token_handle, fieldnames=token_fields)
        sample_writer.writeheader()
        token_writer.writeheader()

        for row in sample_manifest:
            sample_id = row["sample_id"]
            sample_index = sample_to_index[sample_id]
            positions, _, _, _ = valid_token_arrays(data, sample_index)
            token_rows = build_token_rows_for_sample(data, sample_index, include_cls=include_cls)

            sample_out = {
                **row,
                "sample_index": sample_index,
                "nonzero_otu_count": len(positions),
                "selected_token_count": len(token_rows),
            }
            sample_writer.writerow({field: sample_out.get(field, "") for field in sample_fields})
            split_counts[row["split"]] += 1
            group_counts[(row["quota_group"], row["split"])] += 1

            for token_row in token_rows:
                token_type_counts[token_row["token_type"]] += 1
                token_count_total += 1
                token_writer.writerow(
                    {
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "split": row["split"],
                        "quota_group": row["quota_group"],
                        "primary_env": row["primary_env"],
                        "env_first2": row["env_first2"],
                        "technology": row["technology"],
                        "token_type": token_row["token_type"],
                        "token_position": token_row["token_position"],
                        "otu_index": token_row["otu_index"],
                        "otu_id": idx_to_otu_id.get(token_row["otu_index"], "CLS" if token_row["token_type"] == "cls" else ""),
                        "abundance": token_row["abundance"],
                        "processed_abundance": token_row["processed_abundance"],
                        "abundance_rank": token_row["abundance_rank"],
                    }
                )

    summary = {
        "data_file": str(args.data_file),
        "metadata_file": str(args.metadata_file),
        "seed": args.seed,
        "quota_scale": args.quota_scale,
        "primary_quotas": primary_quotas_from_args(args),
        "wgs_supplement": args.wgs_supplement,
        "include_cls": include_cls,
        "token_policy": "all_valid_tokens",
        "num_samples": len(sample_manifest),
        "num_tokens": token_count_total,
        "split_counts": dict(split_counts),
        "quota_split_counts": {f"{group}|{split}": count for (group, split), count in sorted(group_counts.items())},
        "token_type_counts": dict(token_type_counts),
        "primary_env_counts": dict(Counter(row["primary_env"] for row in sample_manifest)),
        "technology_counts": dict(Counter(row["technology"] for row in sample_manifest)),
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def main():
    args = parse_args()
    if args.val_fraction < 0 or args.test_fraction < 0 or args.val_fraction + args.test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1")
    data = load_pretrain_data(args.data_file)
    sample_ids = {normalize_sample_id(sample_id) for sample_id in data["samples"]}
    metadata = load_metadata_for_samples(args.metadata_file, sample_ids)
    sample_manifest = build_sample_manifest(metadata, args)
    summary = write_outputs(data, sample_manifest, args)
    print(f"Saved samples: {args.output_dir / 'samples.csv'}")
    print(f"Saved tokens: {args.output_dir / 'tokens.csv.gz'}")
    print(f"Saved summary: {args.output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
