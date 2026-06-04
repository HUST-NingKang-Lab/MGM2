#!/usr/bin/env python3
"""Export filtered wastewater abundance tables as per-plant Micro Prophet inputs."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.data.WastewaterDataset import load_wastewater_windows
from src.metrics.wastewater import normalize_abundance


DEFAULT_DATASET = Path("data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl")
DEFAULT_OUTPUT_DIR = Path("data/downstreams/wastewater/microprophet_by_plant")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--data_root", type=Path, default=None, help="Override organized wastewater data root stored in the dataset")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_splits", nargs="*", default=["train", "val"])
    parser.add_argument("--test_splits", nargs="*", default=["test"])
    parser.add_argument("--plants", nargs="*", default=None)
    parser.add_argument(
        "--time_mode",
        choices=["global", "split"],
        default="global",
        help="global keeps one time index across train/test; split restarts time at 1 in each file.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_").lower()
    return slug or "plant"


def collect_samples_by_plant(dataset: Dict, plants: Iterable[str] | None = None, data_root: Path | None = None) -> Dict[str, List[Dict]]:
    selected = set(plants) if plants else None
    data_root = Path(data_root) if data_root is not None else Path(dataset["data_root"])
    asv_universes = dataset["asv_universes"]
    by_plant: Dict[str, List[Dict]] = {}
    for plant in dataset["plants"]:
        plant = str(plant)
        if selected is not None and plant not in selected:
            continue
        asv_ids = list(asv_universes[plant])
        kept_asvs = [asv for asv in asv_ids if asv != "__OTHER__"]
        records = []
        for split in ("train", "val", "test"):
            split_dir = data_root / plant / split
            abundance = pd.read_csv(split_dir / "ASVtable.csv", index_col=0)
            abundance.index = abundance.index.astype(str)
            metadata = pd.read_csv(split_dir / "metadata.csv")
            if "Sample" not in metadata.columns:
                metadata = metadata.rename(columns={metadata.columns[0]: "Sample"})
            metadata["Sample"] = metadata["Sample"].astype(str)
            metadata["Date"] = pd.to_datetime(metadata["Date"], errors="raise")
            sample_ids = metadata["Sample"].tolist()
            table = abundance.loc[:, sample_ids].astype(np.float32).T / 100.0

            missing_kept = [asv for asv in kept_asvs if asv not in table.columns]
            if missing_kept:
                raise ValueError(f"{plant}/{split}: selected ASVs missing from source table: {missing_kept[:5]}")
            selected_values = table.loc[:, kept_asvs].to_numpy(dtype=np.float32)
            if "__OTHER__" in asv_ids:
                other_columns = [asv for asv in table.columns.astype(str).tolist() if asv not in set(kept_asvs)]
                other_values = table.loc[:, other_columns].sum(axis=1).to_numpy(dtype=np.float32)[:, None]
                selected_values = np.concatenate([selected_values, other_values], axis=1)
            selected_values = normalize_abundance(selected_values).astype(np.float32)

            for idx, row in metadata.iterrows():
                records.append(
                    {
                        "plant": plant,
                        "sample_id": str(row["Sample"]),
                        "date": row["Date"].strftime("%Y-%m-%d"),
                        "split": split,
                        "asv_ids": asv_ids,
                        "abundance": selected_values[idx],
                    }
                )
        records.sort(key=lambda item: (pd.Timestamp(item["date"]), item["sample_id"]))
        by_plant[plant] = records
    return dict(sorted(by_plant.items()))


def _sample_frame(records: List[Dict], subject_id: int, splits: set[str], time_mode: str) -> pd.DataFrame:
    selected = [record for record in records if record["split"] in splits]
    if not selected:
        return pd.DataFrame()

    all_asvs = list(records[0]["asv_ids"])
    global_time = {record["sample_id"]: idx + 1 for idx, record in enumerate(records)}
    rows = []
    for local_idx, record in enumerate(selected, start=1):
        row = {
            "subject_id": subject_id,
            "time": global_time[record["sample_id"]] if time_mode == "global" else local_idx,
        }
        row.update({asv_id: float(value) for asv_id, value in zip(all_asvs, record["abundance"])})
        rows.append(row)
    columns = ["subject_id", "time", *all_asvs]
    return pd.DataFrame(rows, columns=columns)


def export_microprophet_by_plant(
    dataset: Dict,
    output_dir: Path,
    train_splits: Iterable[str],
    test_splits: Iterable[str],
    plants: Iterable[str] | None = None,
    data_root: Path | None = None,
    time_mode: str = "global",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_plant = collect_samples_by_plant(dataset, plants=plants, data_root=data_root)
    train_splits_set = set(train_splits)
    test_splits_set = set(test_splits)

    mapping_rows = []
    for subject_id, (plant, records) in enumerate(by_plant.items(), start=1):
        plant_slug = slugify(plant)
        plant_dir = output_dir / plant_slug
        plant_dir.mkdir(parents=True, exist_ok=True)

        train = _sample_frame(records, subject_id, train_splits_set, time_mode)
        test = _sample_frame(records, subject_id, test_splits_set, time_mode)
        if train.empty or test.empty:
            raise ValueError(f"{plant}: empty train or test export")

        train.to_csv(plant_dir / "train.csv", sep="\t", index=False)
        test.to_csv(plant_dir / "test.csv", sep="\t", index=False)
        mapping_rows.append(
            {
                "subject_id": subject_id,
                "plant": plant,
                "plant_slug": plant_slug,
                "train_path": str(plant_dir / "train.csv"),
                "test_path": str(plant_dir / "test.csv"),
                "n_train_samples": len(train),
                "n_test_samples": len(test),
                "n_taxa": len(train.columns) - 2,
                "train_splits": "|".join(sorted(train_splits_set)),
                "test_splits": "|".join(sorted(test_splits_set)),
            }
        )

    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(output_dir / "subject_id_mapping.csv", index=False)
    return mapping


def main():
    args = parse_args()
    dataset = load_wastewater_windows(args.dataset)
    mapping = export_microprophet_by_plant(
        dataset,
        output_dir=args.output_dir,
        train_splits=args.train_splits,
        test_splits=args.test_splits,
        plants=args.plants,
        data_root=args.data_root,
        time_mode=args.time_mode,
    )
    print(f"Saved Micro Prophet inputs under: {args.output_dir}")
    print(f"Saved subject mapping: {args.output_dir / 'subject_id_mapping.csv'}")
    print(mapping.to_string(index=False))


if __name__ == "__main__":
    main()
