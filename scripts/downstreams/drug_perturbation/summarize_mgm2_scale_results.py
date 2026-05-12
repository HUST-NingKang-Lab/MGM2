#!/usr/bin/env python3
"""Summarize MGM2 drug perturbation scale runs into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRIC_ALIASES = {
    "val_loss": ["val/loss", "val/loss_epoch"],
    "val_delta_mse": ["val/delta_mse", "val/delta_mse_epoch"],
    "val_after_mse": ["val/after_mse", "val/after_mse_epoch"],
    "val_raw_bray_curtis": ["val/raw_bray_curtis", "val/raw_bray_curtis_epoch"],
}


FIELDNAMES = [
    "model_scale",
    "split_mode",
    "seed",
    "run_name",
    "output_dir",
    "pretrained_checkpoint",
    "config_json",
    "metrics_csv",
    "best_epoch",
    "best_step",
    "best_val_loss",
    "best_val_delta_mse",
    "best_val_after_mse",
    "best_val_raw_bray_curtis",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "freeze_backbone",
    "num_drug_tokens",
    "cross_attention_heads",
    "max_otus",
    "batch_size",
    "learning_rate",
    "backbone_learning_rate",
    "max_epochs",
]


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _version_key(path: Path) -> tuple[int, float]:
    match = re.search(r"version_(\d+)", str(path))
    version = int(match.group(1)) if match else -1
    return version, path.stat().st_mtime


def resolve_metrics_csv(run_dir: Path, metrics_csv: Path | None) -> Path:
    if metrics_csv is not None:
        if not metrics_csv.exists():
            raise FileNotFoundError(metrics_csv)
        return metrics_csv

    candidates = sorted(run_dir.glob("logs/*/version_*/metrics.csv"), key=_version_key, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No Lightning metrics.csv found under {run_dir / 'logs'}")
    return candidates[0]


def _metric(row: Dict[str, str], aliases: Iterable[str]) -> float | None:
    for name in aliases:
        value = _to_float(row.get(name))
        if value is not None:
            return value
    return None


def best_validation_row(metrics_csv: Path, monitor: str) -> Dict[str, Any]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if monitor not in METRIC_ALIASES:
        raise ValueError(f"Unsupported monitor {monitor!r}; choose one of {sorted(METRIC_ALIASES)}")

    best_row: Dict[str, str] | None = None
    best_value: float | None = None
    for row in rows:
        value = _metric(row, METRIC_ALIASES[monitor])
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_row = row
            best_value = value

    if best_row is None or best_value is None:
        raise ValueError(f"No validation metric {monitor} found in {metrics_csv}")

    return {
        "best_epoch": _to_int(best_row.get("epoch")),
        "best_step": _to_int(best_row.get("step")),
        "best_val_loss": _metric(best_row, METRIC_ALIASES["val_loss"]),
        "best_val_delta_mse": _metric(best_row, METRIC_ALIASES["val_delta_mse"]),
        "best_val_after_mse": _metric(best_row, METRIC_ALIASES["val_after_mse"]),
        "best_val_raw_bray_curtis": _metric(best_row, METRIC_ALIASES["val_raw_bray_curtis"]),
    }


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    args_path = run_dir / "drug_perturbation_args.json"
    config_path = run_dir / "mgm2_config.json"
    args = load_json(args_path) if args_path.exists() else {}
    config = load_json(config_path) if config_path.exists() else {}
    model = config.get("model", {})
    training = config.get("training", {})
    return {
        "pretrained_checkpoint": args.get("pretrained_checkpoint", ""),
        "config_json": str(config_path) if config_path.exists() else args.get("config", ""),
        "freeze_backbone": args.get("freeze_backbone", ""),
        "num_drug_tokens": args.get("num_drug_tokens", ""),
        "cross_attention_heads": args.get("cross_attention_heads", ""),
        "max_otus": args.get("max_otus", ""),
        "batch_size": args.get("batch_size", training.get("batch_size", "")),
        "learning_rate": args.get("learning_rate", training.get("learning_rate", "")),
        "backbone_learning_rate": args.get("backbone_learning_rate", ""),
        "max_epochs": args.get("max_epochs", training.get("max_epochs", "")),
        "hidden_size": model.get("hidden_size", ""),
        "num_hidden_layers": model.get("num_hidden_layers", ""),
        "num_attention_heads": model.get("num_attention_heads", ""),
        "intermediate_size": model.get("intermediate_size", ""),
    }


def read_summary(summary_csv: Path) -> List[Dict[str, str]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary(summary_csv: Path, rows: List[Dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row.get("model_scale", ""), row.get("split_mode", ""), str(row.get("seed", ""))))
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def upsert_row(summary_csv: Path, new_row: Dict[str, Any]) -> None:
    key = (str(new_row["model_scale"]), str(new_row["split_mode"]), str(new_row["seed"]))
    rows = []
    replaced = False
    for row in read_summary(summary_csv):
        row_key = (str(row.get("model_scale", "")), str(row.get("split_mode", "")), str(row.get("seed", "")))
        if row_key == key:
            rows.append(new_row)
            replaced = True
        else:
            rows.append(row)
    if not replaced:
        rows.append(new_row)
    write_summary(summary_csv, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one MGM2 drug perturbation scale run")
    parser.add_argument("--run_dir", type=Path, required=True, help="Drug perturbation run directory")
    parser.add_argument("--summary_csv", type=Path, required=True, help="Output summary CSV")
    parser.add_argument("--model_scale", required=True, help="Stage-2 scale name")
    parser.add_argument("--split_mode", required=True, help="Evaluation split mode")
    parser.add_argument("--seed", required=True, help="Random seed")
    parser.add_argument("--run_name", default="", help="Human-readable run name")
    parser.add_argument("--metrics_csv", type=Path, default=None, help="Optional explicit Lightning metrics.csv")
    parser.add_argument("--monitor", choices=sorted(METRIC_ALIASES), default="val_loss")
    args = parser.parse_args()

    metrics_csv = resolve_metrics_csv(args.run_dir, args.metrics_csv)
    row: Dict[str, Any] = {
        "model_scale": args.model_scale,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "run_name": args.run_name,
        "output_dir": str(args.run_dir),
        "metrics_csv": str(metrics_csv),
    }
    row.update(load_run_metadata(args.run_dir))
    row.update(best_validation_row(metrics_csv, args.monitor))
    upsert_row(args.summary_csv, row)
    print(f"Updated {args.summary_csv} with {args.model_scale}/{args.split_mode}/seed{args.seed}")


if __name__ == "__main__":
    main()
