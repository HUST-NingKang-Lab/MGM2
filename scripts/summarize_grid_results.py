#!/usr/bin/env python3
"""Summarize one MGM2 grid-search run into a stage-level CSV.

Example:
    python scripts/summarize_grid_results.py \
        --metrics_csv outputs_grid/stage1/run_001/logs/demo/version_0/metrics.csv \
# If version_0 is missing, the script automatically falls back to the latest sibling version_*/metrics.csv.
        --summary_csv outputs_grid/stage1/summary.csv \
        --run_name run_001 \
        --output_dir outputs_grid/stage1/run_001 \
        --config_json outputs_grid/stage1/run_001/config.json \
        --checkpoint_dir outputs_grid/stage1/run_001/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict




def _version_sort_key(path: Path) -> tuple[int, float]:
    parent_name = path.parent.name
    try:
        version_idx = int(parent_name.split('_')[1]) if parent_name.startswith('version_') else -1
    except (IndexError, ValueError):
        version_idx = -1
    return version_idx, path.stat().st_mtime


def resolve_metrics_csv(metrics_csv: Path) -> Path:
    if metrics_csv.exists():
        return metrics_csv

    version_parent = metrics_csv.parent.parent if metrics_csv.parent.name.startswith('version_') else metrics_csv.parent
    candidates = sorted(version_parent.glob('version_*/metrics.csv'), key=_version_sort_key, reverse=True)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f'Could not find metrics.csv at {metrics_csv} or any sibling version_* directory under {version_parent}'
    )

def _to_float(value: str | None) -> float | None:
    if value is None or value == '':
        return None
    return float(value)


def extract_best_metrics(metrics_csv: Path) -> Dict[str, float]:
    metrics_csv = resolve_metrics_csv(metrics_csv)
    with metrics_csv.open('r', encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))

    best_row = None
    best_total_loss = None
    for row in rows:
        total_loss = _to_float(row.get('val/total_loss'))
        if total_loss is None:
            continue
        if best_total_loss is None or total_loss < best_total_loss:
            best_total_loss = total_loss
            best_row = row

    if best_row is None or best_total_loss is None:
        raise ValueError(f'No validation metrics found in {metrics_csv}')

    return {
        'best_val_total_loss': best_total_loss,
        'best_val_bin_accuracy': _to_float(best_row.get('val/bin_accuracy')) or 0.0,
        'best_val_regression_mse': _to_float(best_row.get('val/regression_mse')) or 0.0,
    }


def find_best_checkpoint(checkpoint_dir: Path) -> str:
    checkpoints = sorted(checkpoint_dir.glob('*.ckpt'), key=lambda path: path.stat().st_mtime, reverse=True)
    preferred = [path for path in checkpoints if path.name != 'last.ckpt']
    selected = preferred[0] if preferred else (checkpoints[0] if checkpoints else None)
    return str(selected) if selected else ''


def load_summary_hparams(config_json: Path) -> Dict[str, object]:
    with config_json.open('r', encoding='utf-8') as handle:
        config = json.load(handle)

    return {
        'learning_rate': config.get('training', {}).get('learning_rate'),
        'weight_decay': config.get('training', {}).get('weight_decay'),
        'mask_prob': config.get('training', {}).get('mask_prob'),
        'lambda_regression': config.get('loss', {}).get('lambda_regression'),
        'qwen3_loss_weight': config.get('qwen3_alignment', {}).get('loss_weight'),
        'hidden_size': config.get('model', {}).get('hidden_size'),
        'num_hidden_layers': config.get('model', {}).get('num_hidden_layers'),
        'num_attention_heads': config.get('model', {}).get('num_attention_heads'),
        'intermediate_size': config.get('model', {}).get('intermediate_size'),
    }


def append_summary_row(summary_csv: Path, row: Dict[str, object]) -> None:
    fieldnames = [
        'run_name',
        'output_dir',
        'best_checkpoint_path',
        'best_val_total_loss',
        'best_val_bin_accuracy',
        'best_val_regression_mse',
        'learning_rate',
        'weight_decay',
        'mask_prob',
        'lambda_regression',
        'qwen3_loss_weight',
        'hidden_size',
        'num_hidden_layers',
        'num_attention_heads',
        'intermediate_size',
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()
    with summary_csv.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description='Append one MGM2 grid-search run to a stage summary CSV')
    parser.add_argument('--metrics_csv', type=Path, required=True, help='Lightning metrics.csv path')
    parser.add_argument('--summary_csv', type=Path, required=True, help='Summary CSV to append to')
    parser.add_argument('--run_name', required=True, help='Human-readable run name')
    parser.add_argument('--output_dir', type=Path, required=True, help='Run output directory')
    parser.add_argument('--config_json', type=Path, required=True, help='Run config JSON path')
    parser.add_argument('--checkpoint_dir', type=Path, required=True, help='Checkpoint directory')
    args = parser.parse_args()

    row = {
        'run_name': args.run_name,
        'output_dir': str(args.output_dir),
        'best_checkpoint_path': find_best_checkpoint(args.checkpoint_dir),
    }
    row.update(extract_best_metrics(args.metrics_csv))
    row.update(load_summary_hparams(args.config_json))
    append_summary_row(args.summary_csv, row)


if __name__ == '__main__':
    main()
