"""Utilities for building wastewater time-series forecasting datasets."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")
OTHER_BUCKET = "__OTHER__"


@dataclass(frozen=True)
class PlantSeries:
    plant: str
    asv_ids: List[str]
    sample_ids: List[str]
    dates: List[str]
    split_labels: List[str]
    abundance: np.ndarray
    metadata: pd.DataFrame


def discover_plants(data_root: Path) -> List[str]:
    """Return plant directories that contain train/val/test ASV tables."""
    root = Path(data_root)
    plants = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "combined":
            continue
        if all((child / split / "ASVtable.csv").exists() for split in SPLITS):
            plants.append(child.name)
    return plants


def _read_split(plant_dir: Path, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = plant_dir / split
    abundance = pd.read_csv(split_dir / "ASVtable.csv", index_col=0)
    metadata = pd.read_csv(split_dir / "metadata.csv")
    if "Sample" not in metadata.columns:
        first = metadata.columns[0]
        metadata = metadata.rename(columns={first: "Sample"})
    if "Date" not in metadata.columns:
        raise ValueError(f"metadata is missing Date column: {split_dir / 'metadata.csv'}")
    sample_ids = [str(sample) for sample in metadata["Sample"].tolist()]
    missing = [sample for sample in sample_ids if sample not in abundance.columns]
    if missing:
        raise ValueError(f"{plant_dir.name}/{split} metadata samples missing from ASV table: {missing[:5]}")
    abundance = abundance.loc[:, sample_ids]
    return abundance, metadata


def _select_asv_universe(
    abundance: pd.DataFrame,
    top_k_asvs: int = 0,
    add_other_bucket: bool = False,
) -> Tuple[List[str], np.ndarray]:
    values = abundance.to_numpy(dtype=np.float32).T / 100.0
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    if top_k_asvs and top_k_asvs > 0 and top_k_asvs < abundance.shape[0]:
        mean_abundance = values.mean(axis=0)
        keep_idx = np.argsort(-mean_abundance)[:top_k_asvs]
        keep_idx = np.sort(keep_idx)
        kept = values[:, keep_idx]
        asv_ids = [str(abundance.index[idx]) for idx in keep_idx]
        if add_other_bucket:
            other_idx = np.setdiff1d(np.arange(values.shape[1]), keep_idx)
            other = values[:, other_idx].sum(axis=1, keepdims=True)
            kept = np.concatenate([kept, other], axis=1)
            asv_ids.append(OTHER_BUCKET)
        values = kept
    else:
        asv_ids = [str(asv) for asv in abundance.index.tolist()]
    return asv_ids, values.astype(np.float32)


def _load_plant_series(
    data_root: Path,
    plant: str,
    top_k_asvs: int = 0,
    add_other_bucket: bool = False,
) -> PlantSeries:
    plant_dir = Path(data_root) / plant
    split_tables = []
    split_metadata = []
    split_labels = []
    for split in SPLITS:
        abundance, metadata = _read_split(plant_dir, split)
        split_tables.append(abundance)
        split_metadata.append(metadata)
        split_labels.extend([split] * len(metadata))

    all_asvs = split_tables[0].index.astype(str)
    for split, table in zip(SPLITS[1:], split_tables[1:]):
        if list(table.index.astype(str)) != list(all_asvs):
            raise ValueError(f"{plant}/{split} ASV order does not match train split")

    abundance = pd.concat(split_tables, axis=1)
    metadata = pd.concat(split_metadata, axis=0, ignore_index=True)
    metadata["Sample"] = metadata["Sample"].astype(str)
    metadata["Date"] = pd.to_datetime(metadata["Date"], errors="raise")
    order = np.argsort(metadata["Date"].to_numpy())
    if not np.all(order == np.arange(len(order))):
        metadata = metadata.iloc[order].reset_index(drop=True)
        abundance = abundance.loc[:, metadata["Sample"].tolist()]
        split_labels = [split_labels[idx] for idx in order]

    asv_ids, values = _select_asv_universe(abundance, top_k_asvs=top_k_asvs, add_other_bucket=add_other_bucket)
    totals = values.sum(axis=1, keepdims=True)
    values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)

    return PlantSeries(
        plant=plant,
        asv_ids=asv_ids,
        sample_ids=metadata["Sample"].astype(str).tolist(),
        dates=[date.strftime("%Y-%m-%d") for date in metadata["Date"]],
        split_labels=split_labels,
        abundance=values.astype(np.float32),
        metadata=metadata,
    )


def _window_split(split_labels: Sequence[str], target_start: int, horizon: int) -> Optional[str]:
    target_splits = split_labels[target_start : target_start + horizon]
    if len(target_splits) != horizon:
        return None
    first = target_splits[0]
    if all(split == first for split in target_splits):
        return first
    return None


def _metadata_subset(metadata: pd.DataFrame, rows: Sequence[int], columns: Sequence[str]) -> Dict[str, List]:
    selected = {}
    for column in columns:
        if column in metadata.columns:
            values = metadata.iloc[list(rows)][column].tolist()
            selected[column] = [None if pd.isna(value) else value for value in values]
    return selected


def build_wastewater_windows(
    data_root: Path,
    context_len: int = 10,
    horizon: int = 10,
    plants: Optional[Sequence[str]] = None,
    top_k_asvs: int = 0,
    add_other_bucket: bool = False,
    metadata_columns: Sequence[str] = ("week_mean_temperature", "SVI", "DSVI", "Period"),
) -> Dict:
    """Build plant-local causal forecasting windows from organized wastewater data."""
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    root = Path(data_root)
    selected_plants = list(plants) if plants else discover_plants(root)
    if not selected_plants:
        raise ValueError(f"no plant directories found under {root}")

    windows = []
    asv_universes = {}
    plant_summaries = {}
    split_window_counts = {split: 0 for split in SPLITS}
    warnings = []

    for plant in selected_plants:
        series = _load_plant_series(
            root,
            plant,
            top_k_asvs=top_k_asvs,
            add_other_bucket=add_other_bucket,
        )
        asv_universes[plant] = series.asv_ids
        plant_counts = {split: 0 for split in SPLITS}
        n_samples = len(series.sample_ids)
        max_start = n_samples - context_len - horizon + 1
        for context_start in range(max(0, max_start)):
            context_end = context_start + context_len
            target_start = context_end
            target_end = target_start + horizon
            split = _window_split(series.split_labels, target_start, horizon)
            if split is None:
                continue

            context_rows = list(range(context_start, context_end))
            target_rows = list(range(target_start, target_end))
            window = {
                "plant": plant,
                "split": split,
                "context_sample_ids": [series.sample_ids[idx] for idx in context_rows],
                "target_sample_ids": [series.sample_ids[idx] for idx in target_rows],
                "context_dates": [series.dates[idx] for idx in context_rows],
                "target_dates": [series.dates[idx] for idx in target_rows],
                "asv_ids": series.asv_ids,
                "context_abundance": series.abundance[context_rows].copy(),
                "target_abundance": series.abundance[target_rows].copy(),
                "context_metadata": _metadata_subset(series.metadata, context_rows, metadata_columns),
                "target_calendar": {
                    "date": [series.dates[idx] for idx in target_rows],
                    "days_from_last_context": [
                        int((series.metadata.iloc[idx]["Date"] - series.metadata.iloc[context_end - 1]["Date"]).days)
                        for idx in target_rows
                    ],
                },
            }
            windows.append(window)
            plant_counts[split] += 1
            split_window_counts[split] += 1

        for split in SPLITS:
            if plant_counts[split] == 0:
                warnings.append(f"{plant}: no {split} windows for context_len={context_len}, horizon={horizon}")
        plant_summaries[plant] = {
            "num_samples": n_samples,
            "num_asvs": len(series.asv_ids),
            "split_samples": {split: int(sum(label == split for label in series.split_labels)) for split in SPLITS},
            "split_windows": plant_counts,
            "date_start": series.dates[0] if series.dates else "",
            "date_end": series.dates[-1] if series.dates else "",
        }

    return {
        "version": 1,
        "data_root": str(root),
        "context_len": int(context_len),
        "horizon": int(horizon),
        "plants": selected_plants,
        "top_k_asvs": int(top_k_asvs),
        "add_other_bucket": bool(add_other_bucket),
        "asv_universes": asv_universes,
        "windows": windows,
        "summary": {
            "num_plants": len(selected_plants),
            "num_windows": len(windows),
            "split_window_counts": split_window_counts,
            "plants": plant_summaries,
            "warnings": warnings,
        },
    }


def save_wastewater_windows(dataset: Dict, output_path: Path) -> None:
    """Save a wastewater window dataset pickle."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_wastewater_windows(path: Path) -> Dict:
    """Load a wastewater window dataset pickle."""
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
