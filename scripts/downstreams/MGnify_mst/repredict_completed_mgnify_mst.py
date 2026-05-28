#!/usr/bin/env python3
"""Re-run MGnify MST predictions for completed MGM2 downstream classifiers.

This is intended to repair prediction CSVs produced by multi-GPU DDP runs where
only one rank's test shard was saved. The script runs prediction on a single GPU
and writes a complete `test_predictions.csv` for selected completed runs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pandas as pd
import pytorch_lightning as pl
import torch

from src.configs.Config import MGM2Config
from src.data.CommunityClassificationDataset import (
    create_community_classification_dataloaders,
    create_community_classification_test_loader,
)
from src.training.CommunityClassification import save_prediction_table
from src.training.CommunityClassificationModule import CommunityClassificationModule


# DEFAULT_RUNS = [
#     ("scale_small", "env_level_1"),
#     ("scale_medium", "env_level_1"),
#     ("scale_large", "env_level_1"),
#     ("scale_large", "env_level_2"),
# ]
DEFAULT_RUNS = [
    ("scale_small", "env_level_2"),
    ("scale_small", "env_level_3"),
    ("scale_small", "env_level_4"),
    ("scale_small", "env_level_5"),
    ("scale_medium", "env_level_2"),
    ("scale_medium", "env_level_3"),
    ("scale_medium", "env_level_4"),
    ("scale_medium", "env_level_5"),
]
DEFAULT_SEEDS = [11, 22, 33, 44, 55]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-predict completed MGnify MST MGM2 classifiers on a single GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data/downstreams/MGnify_mst"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs_downstreams/MGnify_mst"))
    parser.add_argument("--runs", nargs="*", default=None, help="Optional runs as scale:level, e.g. scale_large:env_level_2")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--sample_id_column", default="SampleID")
    parser.add_argument("--pooling_mode", choices=["mean_pool", "cls"], default="cls")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--backup_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prediction_name", default="test_predictions.csv")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def parse_runs(raw_runs: list[str] | None) -> list[tuple[str, str]]:
    if not raw_runs:
        return DEFAULT_RUNS
    runs = []
    for raw in raw_runs:
        if ":" not in raw:
            raise ValueError(f"Run must be formatted as scale:level, got {raw!r}")
        scale, level = raw.split(":", 1)
        runs.append((scale, level))
    return runs


def class_names_from_mapping(label_to_id: dict[str, int]) -> list[str]:
    return [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]


def find_classification_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    candidates = sorted(ckpt_dir.glob("community-*.ckpt"))
    candidates = [path for path in candidates if "last" not in path.name]
    if candidates:
        def parse_loss(path: Path) -> float:
            match = re.search(r"val/loss=([\d.]+)", str(path))
            return float(match.group(1)) if match else float("inf")

        return min(candidates, key=parse_loss)
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    raise FileNotFoundError(f"No community checkpoint or last.ckpt found in {ckpt_dir}")


def backup_prediction_file(path: Path) -> None:
    if not path.exists():
        return
    backup_path = path.with_name(path.stem + ".rank0_shard" + path.suffix)
    if backup_path.exists():
        return
    shutil.copy2(path, backup_path)
    print(f"Backed up existing shard to {backup_path}")


def repredict_one(args: argparse.Namespace, scale: str, level: str, seed: int) -> None:
    run_dir = args.output_root / scale / level / f"seed_{seed}"
    config_path = run_dir / "config.json"
    label_path = run_dir / "label_to_id.json"
    prediction_path = run_dir / args.prediction_name
    if not config_path.exists() or not label_path.exists():
        print(f"SKIP {scale}/{level}/seed_{seed}: missing config or label_to_id")
        return
    ckpt_path = find_classification_checkpoint(run_dir)
    with label_path.open(encoding="utf-8") as handle:
        label_to_id = {str(label): int(idx) for label, idx in json.load(handle).items()}
    class_names = class_names_from_mapping(label_to_id)
    config = MGM2Config.load(str(config_path))
    config.training.batch_size = args.batch_size
    config.training.num_workers = args.num_workers
    config.training.devices = [args.gpu]
    config.training.precision = args.precision

    train_data_file = args.data_dir / "train.pkl"
    test_data_file = args.data_dir / "test.pkl"
    train_metadata_file = args.data_dir / "train_meta_split.csv"
    test_metadata_file = args.data_dir / "test_meta_split.csv"
    train_embedding = args.data_dir / "train.resolved_embeddings.h5"
    test_embedding = args.data_dir / "test.resolved_embeddings.h5"

    print(f"Repredict {scale}/{level}/seed_{seed} from {ckpt_path}")
    if args.dry_run:
        return

    pl.seed_everything(config.seed, workers=True)
    train_loader, _, _ = create_community_classification_dataloaders(
        data_path=train_data_file,
        metadata_path=train_metadata_file,
        ntv3_embedding_path=train_embedding,
        abundance_bin_config=config.abundance.__dict__,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_id_column=args.sample_id_column,
        label_column=level,
        val_ratio=args.val_ratio,
        seed=config.seed,
    )
    test_loader = create_community_classification_test_loader(
        data_path=test_data_file,
        metadata_path=test_metadata_file,
        ntv3_embedding_path=test_embedding,
        abundance_bin_config=config.abundance.__dict__,
        label_to_id=label_to_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_id_column=args.sample_id_column,
        label_column=level,
        abundance_preprocessor=train_loader.dataset.abundance_preprocessor,
        bin_manager=train_loader.dataset.bin_manager,
    )
    module = CommunityClassificationModule(
        config=config,
        num_classes=len(label_to_id),
        class_names=class_names,
        freeze_backbone=True,
        pretrained_checkpoint=None,
        pooling_mode=args.pooling_mode,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[args.gpu] if torch.cuda.is_available() else 1,
        precision=args.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    prediction_batches = trainer.predict(module, test_loader, ckpt_path=str(ckpt_path))
    if args.backup_existing:
        backup_prediction_file(prediction_path)
    save_prediction_table(prediction_batches, class_names, prediction_path)
    written = pd.read_csv(prediction_path, usecols=["SampleID"])
    print(f"Finished {scale}/{level}/seed_{seed}: {len(written):,} rows -> {prediction_path}")


def main() -> None:
    args = parse_args()
    for scale, level in parse_runs(args.runs):
        for seed in args.seeds:
            repredict_one(args, scale, level, seed)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
