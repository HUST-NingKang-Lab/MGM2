#!/usr/bin/env python3
"""Build a compact MGM2 drug perturbation dataset from raw MicrobeDrugPerturbation HDF5 files."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
from tqdm import tqdm

from src.data.DrugPerturbationDataset import DrugPerturbationDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build compact preprocessed data for MGM2 drug perturbation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, required=True, help="MicrobeDrugPerturbation raw data directory")
    parser.add_argument("--ntv3_embedding_path", type=Path, required=True, help="NTv3 embedding HDF5 path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/downstreams/drug_perturbation/drug_perturbation_top1024.pkl"),
        help="Output pickle path",
    )
    parser.add_argument("--baseline_dir", type=Path, default=Path("outputs_downstreams/drug_perturbation/baseline"))
    parser.add_argument("--examples_csv", type=Path, default=None, help="Canonical baseline examples.csv")
    parser.add_argument("--selected_otus_tsv", type=Path, default=None, help="Canonical baseline selected_otus.tsv")
    parser.add_argument("--max_otus", type=int, default=1024)
    parser.add_argument("--min_abundance", type=float, default=1e-4)
    parser.add_argument("--output_min", type=float, default=-2.0)
    parser.add_argument("--output_max", type=float, default=2.0)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min_read_sum", type=int, default=100)
    parser.add_argument("--min_control_read_sum", type=int, default=100)
    return parser.parse_args()




def _selected_profile_matrix(dataset: DrugPerturbationDataset):
    needed_samples = []
    for example in dataset.examples:
        needed_samples.append(int(example["sample_idx"]))
        needed_samples.extend(int(idx) for idx in example["control_indices"])
    sample_indices = sorted(set(needed_samples))
    selected_idx = dataset.selected_otu_indices.astype(int)
    sorted_otu = np.sort(selected_idx)
    order = np.argsort(selected_idx)
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    logger.info("Loading selected OTU matrix: %d OTUs x %d samples", len(selected_idx), len(sample_indices))
    with h5py.File(dataset.otu_counts_path, "r") as handle:
        block = handle["rel_abundance"][sorted_otu, :][:, sample_indices]
    matrix = block[inverse_order].astype(np.float32)
    sample_pos = {sample_idx: pos for pos, sample_idx in enumerate(sample_indices)}
    return matrix, sample_pos


def _build_fixed_otu_item(dataset: DrugPerturbationDataset, example: Dict[str, Any], matrix: np.ndarray, sample_pos: Dict[int, int]) -> Dict[str, Any]:
    after_eval_raw = matrix[:, sample_pos[int(example["sample_idx"])]].astype(np.float32)
    control_pos = [sample_pos[int(idx)] for idx in example["control_indices"]]
    before_eval_raw = matrix[:, control_pos].mean(axis=1).astype(np.float32)
    before_present_raw = before_eval_raw > 0
    after_present_raw = after_eval_raw > 0
    before_raw = np.where(before_eval_raw > 0, before_eval_raw, dataset.min_abundance).astype(np.float32)
    after_raw = np.where(after_eval_raw > 0, after_eval_raw, dataset.min_abundance).astype(np.float32)
    before_values = dataset.preprocessor.transform(before_raw)
    after_values = dataset.preprocessor.transform(after_raw)
    input_ids = np.concatenate([np.array([-1], dtype=np.int64), dataset.selected_otu_indices.astype(np.int64)])
    before_values = np.concatenate([np.array([0.0], dtype=np.float32), before_values.astype(np.float32)])
    after_values = np.concatenate([np.array([0.0], dtype=np.float32), after_values.astype(np.float32)])
    before_raw = np.concatenate([np.array([0.0], dtype=np.float32), before_raw])
    after_raw = np.concatenate([np.array([0.0], dtype=np.float32), after_raw])
    before_eval_raw = np.concatenate([np.array([0.0], dtype=np.float32), before_eval_raw])
    after_eval_raw = np.concatenate([np.array([0.0], dtype=np.float32), after_eval_raw])
    before_present_mask = np.concatenate([np.array([False]), before_present_raw])
    after_present_mask = np.concatenate([np.array([False]), after_present_raw])
    supervision_mask = np.ones_like(before_present_mask, dtype=bool)
    supervision_mask[0] = False
    ntv3_indices = [dataset.otu_id_to_ntv3_idx.get(otu_id, -1) for otu_id in dataset.selected_otu_ids]
    return {
        "sample_id": example["sample_id"],
        "donor": example["donor"],
        "drug_key": example["drug_key"],
        "otu_ids": list(dataset.selected_otu_ids),
        "ntv3_indices": np.asarray(ntv3_indices, dtype=np.int64),
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": np.ones(len(input_ids), dtype=np.int64),
        "before_abundance_values": before_values.astype(np.float32),
        "after_abundance_values": after_values.astype(np.float32),
        "before_abundance_raw": before_raw.astype(np.float32),
        "after_abundance_raw": after_raw.astype(np.float32),
        "before_abundance_eval_raw": before_eval_raw.astype(np.float32),
        "after_abundance_eval_raw": after_eval_raw.astype(np.float32),
        "before_present_mask": before_present_mask.astype(bool),
        "after_present_mask": after_present_mask.astype(bool),
        "supervision_mask": supervision_mask.astype(bool),
        "drug_embedding": dataset.drug_embeddings[example["drug_idx"]].astype(np.float32),
        "donor_id": int(example["donor_id"]),
    }




def _fixed_embedding_matrix(dataset: DrugPerturbationDataset) -> np.ndarray:
    if dataset.selected_otu_ids is None:
        raise ValueError("fixed embedding matrix requires selected OTUs")
    embeddings = np.zeros((len(dataset.selected_otu_ids) + 1, dataset.ntv3_embeddings.shape[1]), dtype=np.float32)
    for row_idx, otu_id in enumerate(dataset.selected_otu_ids, start=1):
        ntv3_idx = dataset.otu_id_to_ntv3_idx.get(otu_id)
        if ntv3_idx is not None:
            embeddings[row_idx] = dataset.ntv3_embeddings[ntv3_idx]
    return embeddings


def _example_metadata(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": example["sample_id"],
        "donor": example["donor"],
        "donor_id": int(example["donor_id"]),
        "drug_key": example["drug_key"],
    }


def main() -> None:
    args = parse_args()
    examples_csv = args.examples_csv or args.baseline_dir / "examples.csv"
    selected_otus_tsv = args.selected_otus_tsv or args.baseline_dir / "selected_otus.tsv"
    if not examples_csv.exists():
        raise FileNotFoundError(f"Canonical examples file not found: {examples_csv}")
    if not selected_otus_tsv.exists():
        raise FileNotFoundError(f"Canonical selected OTU file not found: {selected_otus_tsv}")

    dataset = DrugPerturbationDataset(
        data_dir=args.data_dir,
        ntv3_embedding_path=args.ntv3_embedding_path,
        max_otus=args.max_otus,
        min_abundance=args.min_abundance,
        output_min=args.output_min,
        output_max=args.output_max,
        normalize=args.normalize,
        min_read_sum=args.min_read_sum,
        min_control_read_sum=args.min_control_read_sum,
        cache_aligned_examples=False,
        examples_csv=examples_csv,
        selected_otus_tsv=selected_otus_tsv,
    )

    logger.info("Building %d compact drug perturbation examples", len(dataset.examples))
    if dataset.selected_otu_indices is not None:
        profile_matrix, sample_pos = _selected_profile_matrix(dataset)
        items = [
            _build_fixed_otu_item(dataset, example, profile_matrix, sample_pos)
            for example in tqdm(dataset.examples, desc="drug perturbation")
        ]
    else:
        items = [dataset._build_item_arrays(example) for example in tqdm(dataset.examples, desc="drug perturbation")]
    examples = [_example_metadata(example) for example in dataset.examples]
    payload = {
        "format": "mgm2_drug_perturbation",
        "format_version": 1,
        "samples": [example["sample_id"] for example in examples],
        "examples": examples,
        "items": items,
        "donor_to_id": dataset.donor_to_id,
        "drug_embedding_dim": int(dataset.drug_embeddings.shape[1]),
        "selected_otus": dataset.selected_otus.to_dict(orient="list") if dataset.selected_otus is not None else None,
        "fixed_embeddings": _fixed_embedding_matrix(dataset) if dataset.selected_otu_ids is not None else None,
        "ntv3_embedding_path": str(args.ntv3_embedding_path),
        "build_config": {
            "data_dir": str(args.data_dir),
            "baseline_dir": str(args.baseline_dir),
            "examples_csv": str(examples_csv),
            "selected_otus_tsv": str(selected_otus_tsv),
            "max_otus": args.max_otus,
            "min_abundance": args.min_abundance,
            "output_min": args.output_min,
            "output_max": args.output_max,
            "normalize": args.normalize,
            "min_read_sum": args.min_read_sum,
            "min_control_read_sum": args.min_control_read_sum,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote %d examples to %s", len(items), args.output)


if __name__ == "__main__":
    main()
