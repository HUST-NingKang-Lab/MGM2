#!/usr/bin/env python3
"""Train MGM2 cross-attention models for drug perturbation prediction."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
script_dir = project_root / "scripts" / "downstreams" / "drug_perturbation"
sys.path.insert(0, str(script_dir))

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Subset

import baseline as baseline_utils  # noqa: E402
from src.configs.Config import MGM2Config
from src.data.DrugPerturbationDataset import DrugPerturbationCollator, DrugPerturbationDataset, DrugPerturbationPreprocessedDataset
from src.training.DrugPerturbationModule import DrugPerturbationLightningModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a drug-conditioned MGM2 perturbation model with cross-attention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=None, help="MicrobeDrugPerturbation raw data directory")
    parser.add_argument("--preprocessed_data", type=Path, default=None, help="Prebuilt compact drug perturbation pickle")
    parser.add_argument("--baseline_dir", type=Path, default=Path("outputs_downstreams/drug_perturbation/baseline"))
    parser.add_argument("--ntv3_embedding_path", type=Path, required=True, help="NTv3 embedding HDF5")
    parser.add_argument("--config", type=Path, default=project_root / "src" / "configs" / "mgm2_default.json")
    parser.add_argument("--pretrained_checkpoint", type=Path, default=None, help="Optional MGM2 pretraining checkpoint")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_downstreams/drug_perturbation/mgm2_cross_attention"))
    parser.add_argument("--experiment_name", type=str, default="drug_perturbation_cross_attention")
    parser.add_argument("--split_mode", choices=["pair", "drug", "donor"], default="pair")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--max_otus", type=int, default=1024)
    parser.add_argument("--max_folds", type=int, default=0, help="Fold cap for LOO; 0 means all folds")
    parser.add_argument("--loo_eval", action=argparse.BooleanOptionalAction, default=False, help="Train/evaluate leave-one-group folds")
    parser.add_argument("--pseudocount", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_aligned_examples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--backbone_learning_rate", type=float, default=None)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_drug_tokens", type=int, default=4)
    parser.add_argument("--cross_attention_heads", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def group_labels(dataset, mode: str) -> List[str]:
    labels = []
    for example in dataset.examples:
        if mode == "pair":
            labels.append(f"{example['donor']}::{example['drug_key']}")
        elif mode == "drug":
            labels.append(example["drug_key"])
        elif mode == "donor":
            labels.append(example["donor"])
        else:
            raise ValueError(f"Unsupported split mode: {mode}")
    return labels


def examples_frame(dataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Run": [example.get("sample_id", "") for example in dataset.examples],
            "Donor": [example.get("donor", "") for example in dataset.examples],
            "drug_key": [example.get("drug_key", "") for example in dataset.examples],
        }
    )


def train_val_indices(labels: List[str], val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=object)
    unique = np.asarray(pd.unique(labels), dtype=object)
    if len(unique) < 2:
        raise ValueError("Need at least two groups for train/validation split")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    n_val = min(n_val, len(unique) - 1)
    val_groups = set(unique[:n_val])
    val_idx = np.flatnonzero(np.asarray([label in val_groups for label in labels]))
    train_idx = np.flatnonzero(np.asarray([label not in val_groups for label in labels]))
    return train_idx, val_idx


def inner_train_val_indices(labels: List[str], train_idx: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    train_labels = [labels[int(idx)] for idx in train_idx]
    inner_train_pos, inner_val_pos = train_val_indices(train_labels, val_fraction, seed)
    return train_idx[inner_train_pos], train_idx[inner_val_pos]


def selected_otus_frame(dataset) -> pd.DataFrame:
    selected = getattr(dataset, "selected_otus", None)
    if selected is None:
        raise ValueError("LOO prediction export requires a fixed selected OTU table in the dataset")
    return selected.reset_index(drop=True)


def _h5_string_array(values: List[str]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def write_prediction_h5(output_path: Path, selected_otus: pd.DataFrame, metadata_frames: List[pd.DataFrame], prediction_chunks: Dict[str, List[np.ndarray]]) -> None:
    if not metadata_frames:
        return
    metadata = pd.concat(metadata_frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("otu_ids", data=_h5_string_array(selected_otus["otu_id"].astype(str).to_list()), dtype=string_dtype)
        handle.create_dataset("otu_idx", data=selected_otus["otu_idx"].astype(int).to_numpy())
        handle.attrs["prediction_space"] = "raw_relative_abundance"
        meta_group = handle.create_group("metadata")
        for column in ["split_mode", "group", "Run", "Donor", "drug_key"]:
            meta_group.create_dataset(column, data=_h5_string_array(metadata[column].astype(str).to_list()), dtype=string_dtype)
        meta_group.create_dataset("fold", data=metadata["fold"].astype(int).to_numpy())
        pred_group = handle.create_group("predictions")
        for method, chunks in sorted(prediction_chunks.items()):
            if chunks:
                pred_group.create_dataset(method, data=np.concatenate(chunks, axis=0).astype(np.float32), compression="gzip")


def load_pretrained_backbone(module: DrugPerturbationLightningModule, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    backbone_state = {}
    for key, value in state_dict.items():
        if key.startswith("model.backbone."):
            backbone_state[key[len("model.backbone.") :]] = value
        elif key.startswith("backbone."):
            backbone_state[key[len("backbone.") :]] = value
    if not backbone_state:
        raise ValueError(f"No MGM2 backbone weights found in checkpoint: {checkpoint_path}")
    missing, unexpected = module.model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"Loaded pretrained backbone from {checkpoint_path}")
    print(f"Backbone missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")


def make_loaders(dataset, train_idx: np.ndarray, val_idx: np.ndarray, args) -> Tuple[DataLoader, DataLoader]:
    collator = DrugPerturbationCollator()
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def make_module(config: MGM2Config, dataset, args) -> DrugPerturbationLightningModule:
    module = DrugPerturbationLightningModule(
        config=config,
        drug_embedding_dim=dataset.drug_embeddings.shape[1],
        num_donors=len(dataset.donor_to_id),
        num_drug_tokens=args.num_drug_tokens,
        cross_attention_heads=args.cross_attention_heads,
        freeze_backbone=args.freeze_backbone,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
    )
    if args.pretrained_checkpoint is not None:
        load_pretrained_backbone(module, args.pretrained_checkpoint)
    return module


def make_trainer(args, config: MGM2Config, run_dir: Path, run_name: str) -> pl.Trainer:
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="drug-perturbation-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss", patience=8, mode="min"),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    return pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=[
            TensorBoardLogger(save_dir=run_dir / "logs", name=run_name),
            CSVLogger(save_dir=run_dir / "logs", name=run_name),
        ],
        gradient_clip_val=config.training.gradient_clip_val,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )


def collect_predictions(module: DrugPerturbationLightningModule, dataset, indices: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    collator = DrugPerturbationCollator()
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "shuffle": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size, **loader_kwargs)
    preds, true_profiles, baseline_profiles = [], [], []
    module.eval()
    device = module.device
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = module(tensor_batch)
            pred_raw = module._transformed_to_raw(outputs["pred_after_abundance"]).detach().cpu().numpy()[:, 1:]
            true_raw = tensor_batch["after_abundance_eval_raw"].detach().cpu().numpy()[:, 1:]
            before_raw = tensor_batch["before_abundance_eval_raw"].detach().cpu().numpy()[:, 1:]
            preds.append(np.clip(pred_raw, 0.0, None).astype(np.float32))
            true_profiles.append(true_raw.astype(np.float32))
            baseline_profiles.append(before_raw.astype(np.float32))
    return np.concatenate(preds, axis=0), np.concatenate(true_profiles, axis=0), np.concatenate(baseline_profiles, axis=0)


def run_single_train_val(args, config: MGM2Config, dataset) -> None:
    labels = group_labels(dataset, args.split_mode)
    train_idx, val_idx = train_val_indices(labels, args.val_fraction, args.seed)
    train_loader, val_loader = make_loaders(dataset, train_idx, val_idx, args)
    print(f"Examples: {len(dataset)} | train: {len(train_idx)} | val: {len(val_idx)}")
    module = make_module(config, dataset, args)
    trainer = make_trainer(args, config, args.output_dir, args.experiment_name)
    trainer.fit(module, train_loader, val_loader)


def run_loo(args, config: MGM2Config, dataset) -> None:
    examples = examples_frame(dataset)
    labels = group_labels(dataset, args.split_mode)
    selected_otus = selected_otus_frame(dataset)
    fold_frames = []
    prediction_metadata: List[pd.DataFrame] = []
    prediction_chunks: Dict[str, List[np.ndarray]] = {"mgm2_cross_attention": []}
    max_folds = None if args.max_folds == 0 else args.max_folds

    for fold_idx, (train_idx, test_idx, group_label) in enumerate(baseline_utils.iter_leave_one_groups(examples, args.split_mode)):
        if max_folds is not None and fold_idx >= max_folds:
            break
        fold_train_idx, fold_val_idx = inner_train_val_indices(labels, train_idx, args.val_fraction, args.seed + fold_idx)
        fold_dir = args.output_dir / "folds" / f"fold_{fold_idx:04d}"
        fold_name = f"{args.experiment_name}_fold{fold_idx:04d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_loader, val_loader = make_loaders(dataset, fold_train_idx, fold_val_idx, args)
        print(f"{args.split_mode} fold {fold_idx + 1}: {group_label} train={len(fold_train_idx)} val={len(fold_val_idx)} test={len(test_idx)}", flush=True)
        module = make_module(config, dataset, args)
        trainer = make_trainer(args, config, fold_dir, fold_name)
        trainer.fit(module, train_loader, val_loader)
        pred, true, before = collect_predictions(module, dataset, test_idx, args)
        prediction_chunks["mgm2_cross_attention"].append(pred)
        test_examples = examples.iloc[test_idx].reset_index(drop=True)
        prediction_metadata.append(
            pd.DataFrame(
                {
                    "split_mode": args.split_mode,
                    "fold": int(fold_idx),
                    "group": group_label,
                    "Run": test_examples["Run"].astype(str).to_list(),
                    "Donor": test_examples["Donor"].astype(str).to_list(),
                    "drug_key": test_examples["drug_key"].astype(str).to_list(),
                }
            )
        )
        metrics = baseline_utils.evaluate_predictions(
            {"mgm2_cross_attention": pred},
            true,
            before,
            pseudocount=args.pseudocount,
        )
        metrics.insert(0, "n_test", len(test_idx))
        metrics.insert(0, "group", group_label)
        metrics.insert(0, "fold", fold_idx)
        metrics.insert(0, "split_mode", args.split_mode)
        fold_frames.append(metrics)

    if not fold_frames:
        raise RuntimeError("No LOO folds were evaluated")
    fold_metrics = pd.concat(fold_frames, ignore_index=True)
    summary = baseline_utils.summarize_metrics(fold_metrics)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    write_prediction_h5(args.output_dir / f"predicted_abundance_{args.split_mode}.h5", selected_otus, prediction_metadata, prediction_chunks)
    print(summary.to_string(index=False))


def main():
    args = parse_args()
    if args.preprocessed_data is None and args.data_dir is None:
        raise ValueError("Either --preprocessed_data or --data_dir must be provided")
    if args.data_dir is not None and not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if args.preprocessed_data is not None and not args.preprocessed_data.exists():
        raise FileNotFoundError(args.preprocessed_data)
    if not args.ntv3_embedding_path.exists():
        raise FileNotFoundError(args.ntv3_embedding_path)
    config = MGM2Config.load(str(args.config))
    config.output_dir = str(args.output_dir)
    config.experiment_name = args.experiment_name
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.devices = args.devices
    config.training.accelerator = args.accelerator
    config.training.precision = args.precision
    config.training.learning_rate = args.learning_rate
    config.seed = args.seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(args.output_dir / "mgm2_config.json"))
    with (args.output_dir / "drug_perturbation_args.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)

    pl.seed_everything(args.seed, workers=True)
    if args.preprocessed_data is not None:
        dataset = DrugPerturbationPreprocessedDataset(
            preprocessed_data=args.preprocessed_data,
            ntv3_embedding_path=args.ntv3_embedding_path,
        )
    else:
        dataset = DrugPerturbationDataset(
            data_dir=args.data_dir,
            ntv3_embedding_path=args.ntv3_embedding_path,
            max_otus=args.max_otus,
            min_abundance=config.abundance.min_abundance,
            output_min=config.abundance.output_min,
            output_max=config.abundance.output_max,
            normalize=config.abundance.normalize,
            cache_aligned_examples=args.cache_aligned_examples,
            examples_csv=args.baseline_dir / "examples.csv",
            selected_otus_tsv=args.baseline_dir / "selected_otus.tsv",
        )
    if args.loo_eval:
        run_loo(args, config, dataset)
    else:
        run_single_train_val(args, config, dataset)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
