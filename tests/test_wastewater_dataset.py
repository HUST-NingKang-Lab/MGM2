import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.WastewaterDataset import build_wastewater_windows, save_wastewater_windows
from src.metrics.wastewater import (
    aggregate_forecast_metrics,
    bray_curtis_distance,
    compute_forecast_metrics,
    normalize_abundance,
    topk_overlap,
)


def _write_split(base: Path, split: str, sample_ids, values, start_offset: int):
    split_dir = base / split
    split_dir.mkdir(parents=True, exist_ok=True)
    asv_ids = ["ASV1", "ASV2", "ASV3"]
    pd.DataFrame(values, index=asv_ids, columns=sample_ids).to_csv(split_dir / "ASVtable.csv")
    dates = pd.date_range("2020-01-01", periods=start_offset + len(sample_ids), freq="7D")[start_offset:]
    pd.DataFrame(
        {
            "Sample": sample_ids,
            "Date": [date.strftime("%Y-%m-%d") for date in dates],
            "Plant": ["PlantA"] * len(sample_ids),
            "week_mean_temperature": np.arange(len(sample_ids), dtype=float),
        }
    ).to_csv(split_dir / "metadata.csv", index=False)
    pd.DataFrame(
        {
            "OTU": asv_ids,
            "Genus": ["g__A", "g__B", "g__C"],
        }
    ).to_csv(split_dir / "taxonomy.csv", index=False)


def _make_organized_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data_organized"
    plant = root / "PlantA"
    train_samples = [f"S{i}" for i in range(8)]
    val_samples = [f"S{i}" for i in range(8, 11)]
    test_samples = [f"S{i}" for i in range(11, 16)]

    def values(offset, n):
        cols = []
        for i in range(n):
            first = 70 - offset - i
            cols.append([first, 20 + i, 10 + offset])
        return np.asarray(cols, dtype=float).T

    _write_split(plant, "train", train_samples, values(0, len(train_samples)), start_offset=0)
    _write_split(plant, "val", val_samples, values(5, len(val_samples)), start_offset=len(train_samples))
    _write_split(
        plant,
        "test",
        test_samples,
        values(10, len(test_samples)),
        start_offset=len(train_samples) + len(val_samples),
    )
    return root


def test_build_wastewater_windows_preserves_causal_split_boundaries(tmp_path):
    root = _make_organized_dataset(tmp_path)
    dataset = build_wastewater_windows(
        data_root=root,
        context_len=3,
        horizon=2,
        plants=["PlantA"],
    )

    assert dataset["context_len"] == 3
    assert dataset["horizon"] == 2
    assert dataset["plants"] == ["PlantA"]
    assert dataset["asv_universes"]["PlantA"] == ["ASV1", "ASV2", "ASV3"]

    split_counts = dataset["summary"]["split_window_counts"]
    assert split_counts["train"] == 4
    assert split_counts["val"] == 2
    assert split_counts["test"] == 4

    val_window = next(window for window in dataset["windows"] if window["split"] == "val")
    assert val_window["context_sample_ids"] == ["S5", "S6", "S7"]
    assert val_window["target_sample_ids"] == ["S8", "S9"]
    assert val_window["context_abundance"].shape == (3, 3)
    assert val_window["target_abundance"].shape == (2, 3)
    assert np.allclose(val_window["target_abundance"].sum(axis=1), 1.0)


def test_build_wastewater_windows_supports_top_k_with_other_bucket(tmp_path):
    root = _make_organized_dataset(tmp_path)
    dataset = build_wastewater_windows(
        data_root=root,
        context_len=3,
        horizon=2,
        plants=["PlantA"],
        top_k_asvs=2,
        add_other_bucket=True,
    )

    assert dataset["asv_universes"]["PlantA"] == ["ASV1", "ASV2", "__OTHER__"]
    first = dataset["windows"][0]
    assert first["context_abundance"].shape[-1] == 3
    assert np.allclose(first["context_abundance"].sum(axis=1), 1.0)


def test_save_wastewater_windows_roundtrips_pickle(tmp_path):
    root = _make_organized_dataset(tmp_path)
    dataset = build_wastewater_windows(root, context_len=3, horizon=2, plants=["PlantA"])
    output = tmp_path / "windows.pkl"
    save_wastewater_windows(dataset, output)

    with output.open("rb") as handle:
        loaded = pickle.load(handle)

    assert loaded["summary"]["num_windows"] == dataset["summary"]["num_windows"]


def test_wastewater_metrics_handle_normalized_forecasts():
    y_true = np.asarray([[[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]], dtype=np.float32)
    y_pred = np.asarray([[[7.0, 2.0, 1.0], [0.1, 0.1, 0.8]]], dtype=np.float32)

    normalized = normalize_abundance(y_pred)
    assert np.allclose(normalized.sum(axis=-1), 1.0)
    assert np.isclose(bray_curtis_distance(y_true[0, 0], normalized[0, 0]), 0.0)
    assert np.isclose(topk_overlap(y_true[0, 1], normalized[0, 1], k=1), 1.0)

    metrics = compute_forecast_metrics(y_true, y_pred, top_k_values=(1, 2))
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert "top1_overlap" in metrics
    assert "top2_overlap" in metrics

    by_horizon = aggregate_forecast_metrics(
        [
            {
                "plant": "PlantA",
                "split": "test",
                "y_true": y_true,
                "y_pred": y_pred,
            }
        ],
        top_k_values=(1,),
    )
    assert list(by_horizon["horizon"]) == [1, 2]
    assert by_horizon.loc[0, "split"] == "test"
