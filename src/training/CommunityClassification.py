#!/usr/bin/env python3
"""Fine-tune MGM2 for downstream community classification.

Example:
    python src/training/CommunityClassification.py \
        --train_data_file data/downstreams/MGnify_mst/train.pkl \
        --train_metadata_file /data9/zhanghaohong/projects/MGM2Benchmarks/MGnify_mst/expert/train_meta.csv \
        --test_data_file data/downstreams/MGnify_mst/test.pkl \
        --test_metadata_file /data9/zhanghaohong/projects/MGM2Benchmarks/MGnify_mst/expert/test_meta.csv \
        --ntv3_embedding_path data/downstreams/MGnify_mst/train.resolved_embeddings.h5 \
        --sample_id_column sample_id \
        --label_column biome \
        --pooling_mode mean_pool \
        --pretrained_checkpoint outputs_grid/stage1/best.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
default_config_path = project_root / 'src' / 'configs' / 'mgm2_default.json'
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import pandas as pd
import torch

from src.configs.Config import MGM2Config
from src.data.CommunityClassificationDataset import (
    create_community_classification_dataloaders,
    create_community_classification_test_loader,
)
from src.training.CommunityClassificationModule import CommunityClassificationModule


def _optional_path(value):
    if value is None:
        return None
    return Path(value)


def _require_existing_path(path_value, argument_name: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f'{argument_name} not found: {path}')
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune MGM2 for downstream community classification with metadata-stratified train/val split',
        epilog=(
            'Example:\n'
            '  python src/training/CommunityClassification.py '
            '--train_data_file data/downstreams/MGnify_mst/train.pkl '
            '--train_metadata_file /path/to/train_meta.csv '
            '--test_data_file data/downstreams/MGnify_mst/test.pkl '
            '--test_metadata_file /path/to/test_meta.csv '
            '--ntv3_embedding_path data/downstreams/MGnify_mst/train.resolved_embeddings.h5 '
            '--sample_id_column sample_id --label_column biome --pooling_mode mean_pool'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--train_data_file', type=Path, required=True, help='Training MGM2 pkl built by src/data/build_dataset.py')
    parser.add_argument('--train_metadata_file', type=Path, required=True, help='Training metadata CSV/TSV/HDF5 containing labels')
    parser.add_argument('--test_data_file', type=Path, default=None, help='Optional held-out test MGM2 pkl')
    parser.add_argument('--test_metadata_file', type=Path, default=None, help='Optional held-out test metadata CSV/TSV/HDF5')
    parser.add_argument('--ntv3_embedding_path', type=Path, required=True, help='Embedding HDF5 used by the pkl files')
    parser.add_argument('--test_ntv3_embedding_path', type=Path, default=None, help='Optional embedding HDF5 for test pkl if train/test were resolved separately')
    parser.add_argument('--config', type=Path, default=None, help='Config JSON; defaults to src/configs/mgm2_default.json')
    parser.add_argument('--pretrained_checkpoint', type=Path, default=None, help='Optional MGM2 pretraining checkpoint; only backbone weights are loaded')
    parser.add_argument('--sample_id_column', type=str, default='sample_id', help='Metadata sample id column')
    parser.add_argument('--label_column', type=str, required=True, help='Metadata class label column')
    parser.add_argument('--metadata_h5_key', type=str, default=None, help='Pandas HDF5 key for train metadata, if metadata is HDF5')
    parser.add_argument('--test_metadata_h5_key', type=str, default=None, help='Pandas HDF5 key for test metadata, if metadata is HDF5')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio inside train metadata, stratified by label')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate override')
    parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay override')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='GPU devices')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--freeze_backbone', action='store_true', help='Train only the classification head')
    parser.add_argument(
        '--pooling_mode',
        type=str,
        choices=['mean_pool', 'cls'],
        default='mean_pool',
        help='Sample-level pooling for classification: mean_pool averages valid CLS/OTU tokens; cls uses only CLS',
    )
    parser.add_argument('--output_dir', type=Path, default=Path('./outputs_community_classification'), help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='community_classification', help='Experiment name')
    parser.add_argument('--prediction_file', type=Path, default=None, help='Optional output CSV for held-out test predictions')
    parser.add_argument('--resume', type=Path, default=None, help='Resume classification checkpoint')
    parser.add_argument('--fast_dev_run', action='store_true', help='Run one train/val/test batch for smoke testing')
    return parser.parse_args()


def create_config(args) -> MGM2Config:
    config_path = _optional_path(args.config)
    if config_path is not None:
        config_path = _require_existing_path(config_path, '--config')
    else:
        config_path = _require_existing_path(default_config_path, 'default config')
    config = MGM2Config.load(str(config_path))

    config.data_path = str(_require_existing_path(args.train_data_file, '--train_data_file'))
    config.ntv3_embedding_path = str(_require_existing_path(args.ntv3_embedding_path, '--ntv3_embedding_path'))
    if args.test_ntv3_embedding_path is not None:
        _require_existing_path(args.test_ntv3_embedding_path, '--test_ntv3_embedding_path')
    config.output_dir = str(args.output_dir)
    config.experiment_name = args.experiment_name
    config.training.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.devices = args.devices
    config.training.num_workers = args.num_workers
    config.training.val_ratio = args.val_ratio
    config.training.train_ratio = 1.0 - args.val_ratio
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.pretrained_checkpoint is not None:
        _require_existing_path(args.pretrained_checkpoint, '--pretrained_checkpoint')
    if args.resume is not None:
        _require_existing_path(args.resume, '--resume')
    _require_existing_path(args.train_metadata_file, '--train_metadata_file')
    if args.test_data_file is not None:
        _require_existing_path(args.test_data_file, '--test_data_file')
    if args.test_metadata_file is not None:
        _require_existing_path(args.test_metadata_file, '--test_metadata_file')
    if (args.test_data_file is None) != (args.test_metadata_file is None):
        raise ValueError('--test_data_file and --test_metadata_file must be provided together')
    return config


def create_callbacks(config: MGM2Config):
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(config.output_dir) / 'checkpoints',
            filename='community-{epoch:02d}-{val/loss:.4f}-{val/accuracy:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            verbose=True,
        ),
        EarlyStopping(monitor='val/loss', patience=5, mode='min', verbose=True),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar(),
    ]
    return callbacks


def summarize_run(args, config: MGM2Config, label_to_id) -> str:
    lines = [
        f'Experiment: {config.experiment_name}',
        f'Output directory: {config.output_dir}',
        f'Train data: {args.train_data_file}',
        f'Train metadata: {args.train_metadata_file}',
        f'Label column: {args.label_column}',
        f'Validation ratio: {args.val_ratio}',
        f'Classes: {len(label_to_id)}',
        f'Pooling mode: {args.pooling_mode}',
        f'Pretrained checkpoint: {args.pretrained_checkpoint or "none"}',
        f'Freeze backbone: {args.freeze_backbone}',
    ]
    if args.test_data_file:
        lines.extend([f'Test data: {args.test_data_file}', f'Test metadata: {args.test_metadata_file}'])
    return '\n'.join(lines)


def _safe_score_column_label(class_name) -> str:
    safe_label = ''.join(char if str(char).isalnum() else '_' for char in str(class_name)).strip('_')
    return safe_label or 'class'


def save_prediction_table(prediction_batches, class_names, output_path: Path):
    rows = []
    for batch in prediction_batches:
        sample_ids = batch['sample_id']
        label_ids = batch['label_id'].detach().cpu().tolist()
        pred_ids = batch['pred_id'].detach().cpu().tolist()
        probabilities = batch['probability'].detach().cpu().tolist()
        logits = batch.get('logits')
        all_probabilities = batch.get('probabilities')
        logit_rows = logits.detach().cpu().tolist() if logits is not None else [None] * len(sample_ids)
        probability_rows = all_probabilities.detach().cpu().tolist() if all_probabilities is not None else [None] * len(sample_ids)
        for sample_id, label_id, pred_id, probability, logit_row, probability_row in zip(
            sample_ids,
            label_ids,
            pred_ids,
            probabilities,
            logit_rows,
            probability_rows,
        ):
            row = {
                'SampleID': sample_id,
                'label_id': int(label_id),
                'label': class_names[int(label_id)],
                'pred_id': int(pred_id),
                'prediction': class_names[int(pred_id)],
                'probability': float(probability),
                'correct': int(label_id) == int(pred_id),
            }
            if logit_row is not None:
                for class_id, (class_name, logit_value) in enumerate(zip(class_names, logit_row)):
                    row[f'logit_{class_id}_{_safe_score_column_label(class_name)}'] = float(logit_value)
            if probability_row is not None:
                for class_id, (class_name, probability_value) in enumerate(zip(class_names, probability_row)):
                    row[f'prob_{class_id}_{_safe_score_column_label(class_name)}'] = float(probability_value)
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f'Wrote {len(rows):,} predictions to {output_path}')


def main():
    args = parse_args()
    config = create_config(args)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(Path(config.output_dir) / 'config.json')

    pl.seed_everything(config.seed, workers=True)
    print('Creating stratified train/validation dataloaders...')
    train_loader, val_loader, label_to_id = create_community_classification_dataloaders(
        data_path=args.train_data_file,
        metadata_path=args.train_metadata_file,
        ntv3_embedding_path=args.ntv3_embedding_path,
        abundance_bin_config=config.abundance.__dict__,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        sample_id_column=args.sample_id_column,
        label_column=args.label_column,
        val_ratio=args.val_ratio,
        seed=config.seed,
        metadata_h5_key=args.metadata_h5_key,
    )
    class_names = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    with open(Path(config.output_dir) / 'label_to_id.json', 'w', encoding='utf-8') as handle:
        json.dump(label_to_id, handle, indent=2, ensure_ascii=False)

    test_loader = None
    if args.test_data_file is not None:
        print('Creating held-out test dataloader...')
        test_ntv3_embedding_path = args.test_ntv3_embedding_path or args.ntv3_embedding_path
        test_loader = create_community_classification_test_loader(
            data_path=args.test_data_file,
            metadata_path=args.test_metadata_file,
            ntv3_embedding_path=test_ntv3_embedding_path,
            abundance_bin_config=config.abundance.__dict__,
            label_to_id=label_to_id,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            sample_id_column=args.sample_id_column,
            label_column=args.label_column,
            metadata_h5_key=args.test_metadata_h5_key,
            abundance_preprocessor=train_loader.dataset.abundance_preprocessor,
            bin_manager=train_loader.dataset.bin_manager,
        )

    print()
    print(summarize_run(args, config, label_to_id))
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    if test_loader is not None:
        print(f'Test batches: {len(test_loader)}')
    print()

    module = CommunityClassificationModule(
        config=config,
        num_classes=len(label_to_id),
        class_names=class_names,
        freeze_backbone=args.freeze_backbone,
        pretrained_checkpoint=str(args.pretrained_checkpoint) if args.pretrained_checkpoint else None,
        pooling_mode=args.pooling_mode,
    )

    total_params = sum(parameter.numel() for parameter in module.parameters())
    trainable_params = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        callbacks=create_callbacks(config),
        logger=[
            TensorBoardLogger(save_dir=Path(config.output_dir) / 'logs', name=config.experiment_name),
            CSVLogger(save_dir=Path(config.output_dir) / 'logs', name=config.experiment_name),
        ],
        fast_dev_run=args.fast_dev_run,
        deterministic=False,
        log_every_n_steps=10,
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=str(args.resume) if args.resume else None)
    if test_loader is not None:
        trainer.test(module, test_loader, ckpt_path='best')
        prediction_file = args.prediction_file or (Path(config.output_dir) / 'test_predictions.csv')
        prediction_batches = trainer.predict(module, test_loader, ckpt_path='best')
        save_prediction_table(prediction_batches, class_names, prediction_file)
    print(f'\nCommunity classification training complete. Results saved to {config.output_dir}')


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
