#!/usr/bin/env python3
"""Extract unlabeled community embeddings from an MGM2 Lightning checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract sample-level MGM2 embeddings without metadata or labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python scripts/extract_community_embeddings.py --model_dir outputs_mgm2/run '
            '--data_file data/communities.pkl --output outputs/community_embeddings.h5\n'
            '  python scripts/extract_community_embeddings.py --checkpoint outputs/checkpoints/last.ckpt '
            '--config outputs/config.json --data_file data/communities.pkl '
            '--embedding_strategy mean_pool --output outputs/community_embeddings.npz'
        ),
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model_dir', type=Path, help='Model run directory containing config and checkpoint files')
    model_group.add_argument('--checkpoint', type=Path, help='Trusted MGM2 Lightning checkpoint')
    parser.add_argument('--config', type=Path, default=None, help='Optional config JSON; inferred when omitted')
    parser.add_argument('--data_file', type=Path, required=True, help='Preprocessed MGM2 dataset (.pkl)')
    parser.add_argument(
        '--ntv3_embedding_path',
        type=Path,
        default=None,
        help='NTv3 embedding HDF5; defaults to ntv3_embedding_path stored in the dataset pickle',
    )
    parser.add_argument('--output', type=Path, required=True, help='Output .h5, .hdf5, .npz, or .csv file')
    parser.add_argument(
        '--embedding_strategy',
        choices=['cls', 'mean_pool', 'qwen3_proj'],
        default='cls',
        help='Sample representation to export',
    )
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--disable_progress', action='store_true')
    return parser.parse_args()


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f'{description} not found: {resolved}')
    return resolved


def _find_config_from_checkpoint(checkpoint: Path) -> Path:
    for directory in checkpoint.resolve().parents:
        candidate = directory / 'config.json'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'Could not infer config.json from {checkpoint}; pass --config explicitly'
    )


def resolve_model_files(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.model_dir is not None:
        model_dir = args.model_dir.expanduser().resolve()
        if not model_dir.is_dir():
            raise NotADirectoryError(f'Model directory not found: {model_dir}')
        checkpoint_candidates = (
            model_dir / 'last.ckpt',
            model_dir / 'model.ckpt',
            model_dir / 'checkpoints' / 'last.ckpt',
        )
        checkpoint = next((path for path in checkpoint_candidates if path.is_file()), None)
        if checkpoint is None:
            searched = ', '.join(str(path) for path in checkpoint_candidates)
            raise FileNotFoundError(f'No checkpoint found in --model_dir; searched: {searched}')
        config = args.config if args.config is not None else model_dir / 'config.json'
    else:
        checkpoint = _require_file(args.checkpoint, 'Checkpoint')
        config = args.config if args.config is not None else _find_config_from_checkpoint(checkpoint)

    return _require_file(checkpoint, 'Checkpoint'), _require_file(config, 'Config')


def _resolve_embedding_path(data: Dict[str, Any], data_file: Path, explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        return _require_file(explicit_path, 'NTv3 embedding file')

    stored_path = data.get('ntv3_embedding_path')
    if not stored_path:
        raise ValueError(
            'The dataset does not contain ntv3_embedding_path; pass --ntv3_embedding_path explicitly'
        )
    stored_path = Path(str(stored_path)).expanduser()
    candidates = [stored_path] if stored_path.is_absolute() else [stored_path, data_file.parent / stored_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'NTv3 embedding file stored in dataset was not found: {stored_path}')


def _iter_abundances(data: Dict[str, Any]) -> Iterable[np.ndarray]:
    if 'otu_data' in data:
        for sample_id in data['samples']:
            yield np.asarray(data['otu_data'][sample_id]['abundance_values'], dtype=np.float32)
    else:
        for abundances in data.get('abundance_values', []):
            yield np.asarray(abundances, dtype=np.float32)


def build_inference_loader(
    config,
    data: Dict[str, Any],
    ntv3_embedding_path: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    from src.data.Dataset import BatchCollatorMGM2, MicrobiomeDatasetMGM2
    from src.data.Preprocessing import AbundancePreprocessor, BinManager

    preprocessor = AbundancePreprocessor(
        min_abundance=config.abundance.min_abundance,
        output_min=config.abundance.output_min,
        output_max=config.abundance.output_max,
        normalize=config.abundance.normalize,
    )
    preprocessor.fit(list(_iter_abundances(data)) if preprocessor.normalize else [])
    bin_manager = BinManager(
        num_bins=config.abundance.num_bins,
        method=config.abundance.binning_method,
        fixed_thresholds=config.abundance.fixed_thresholds,
        enable_regression=config.abundance.enable_regression,
        regression_top_k_bins=config.abundance.regression_top_k_bins,
    )
    bin_manager.fit([])

    dataset = MicrobiomeDatasetMGM2(
        preprocessed_data=data,
        abundance_preprocessor=preprocessor,
        bin_manager=bin_manager,
        mask_prob=0.0,
        ntv3_embedding_path=str(ntv3_embedding_path),
        qwen3_embedding_path=None,
        enable_qwen3_alignment=False,
        is_training=False,
        deterministic_masking=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=BatchCollatorMGM2(),
        pin_memory=pin_memory,
    )


def load_model(checkpoint: Path, config, device: torch.device):
    from src.training.TrainerModule import MGM2LightningModule

    module = MGM2LightningModule.load_from_checkpoint(
        str(checkpoint),
        config=config,
        map_location=device,
        weights_only=False,
    )
    module.eval()
    module.freeze()
    module.to(device)
    return module


def _select_embeddings(
    outputs: Dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    if strategy == 'cls':
        return outputs['cls_hidden']
    if strategy == 'qwen3_proj':
        if 'qwen3_proj' not in outputs:
            raise ValueError('qwen3_proj requires a checkpoint whose config enables Qwen3 alignment')
        return outputs['qwen3_proj']

    token_mask = attention_mask.clone()
    token_mask[:, 0] = 0
    float_mask = token_mask.unsqueeze(-1).to(outputs['hidden_states'].dtype)
    denominator = float_mask.sum(dim=1).clamp_min(1.0)
    return (outputs['hidden_states'] * float_mask).sum(dim=1) / denominator


def _progress(iterable, total: int, disabled: bool):
    if disabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.info('tqdm is unavailable; continuing without a progress bar')
        return iterable
    return tqdm(iterable, total=total, desc='Extracting community embeddings')


def extract_embeddings(
    module,
    loader: DataLoader,
    strategy: str,
    device: torch.device,
    disable_progress: bool,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in _progress(loader, len(loader), disable_progress):
            model_inputs = {
                key: batch[key].to(device, non_blocking=loader.pin_memory)
                for key in ('input_ids', 'embeddings', 'abundance_values', 'attention_mask')
            }
            outputs = module.model(**model_inputs)
            selected = _select_embeddings(outputs, model_inputs['attention_mask'], strategy)
            chunks.append(selected.detach().cpu().numpy().astype(np.float32, copy=False))
    if not chunks:
        raise ValueError('The input dataset contains no samples')
    return np.concatenate(chunks, axis=0)


def _sample_id_text(sample_id: Any) -> str:
    if isinstance(sample_id, (bytes, bytearray, np.bytes_)):
        return sample_id.decode('utf-8', errors='replace')
    return str(sample_id)


def save_embeddings(
    output: Path,
    embeddings: np.ndarray,
    sample_ids: List[str],
    attrs: Dict[str, Any],
) -> None:
    output = output.expanduser().resolve()
    suffix = output.suffix.lower()
    if suffix not in {'.h5', '.hdf5', '.npz', '.csv'}:
        raise ValueError('--output must end in .h5, .hdf5, .npz, or .csv')
    output.parent.mkdir(parents=True, exist_ok=True)

    if suffix in {'.h5', '.hdf5'}:
        string_dtype = h5py.string_dtype(encoding='utf-8')
        with h5py.File(output, 'w') as handle:
            handle.create_dataset('embeddings', data=embeddings, compression='gzip')
            handle.create_dataset(
                'sample_ids',
                data=np.asarray(sample_ids, dtype=object),
                dtype=string_dtype,
            )
            for key, value in attrs.items():
                handle.attrs[key] = value
    elif suffix == '.npz':
        np.savez_compressed(
            output,
            embeddings=embeddings,
            sample_ids=np.asarray(sample_ids, dtype=str),
            **{key: np.asarray(value) for key, value in attrs.items()},
        )
    else:
        frame = pd.DataFrame(embeddings, columns=[f'embedding_{idx}' for idx in range(embeddings.shape[1])])
        frame.insert(0, 'sample_id', sample_ids)
        frame.to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.batch_size <= 0:
        raise ValueError('--batch_size must be positive')
    if args.num_workers < 0:
        raise ValueError('--num_workers must be non-negative')

    checkpoint, config_path = resolve_model_files(args)
    data_file = _require_file(args.data_file, 'Dataset')

    from src.configs.Config import MGM2Config
    from src.data.utils import load_preprocessed_data

    config = MGM2Config.load(str(config_path))
    data = load_preprocessed_data(str(data_file))
    if 'samples' not in data:
        raise KeyError("Preprocessed dataset must contain a 'samples' field")
    ntv3_embedding_path = _resolve_embedding_path(data, data_file, args.ntv3_embedding_path)
    config.data_path = str(data_file)
    config.ntv3_embedding_path = str(ntv3_embedding_path)

    if args.embedding_strategy == 'qwen3_proj' and not config.qwen3_alignment.enabled:
        raise ValueError('qwen3_proj requires qwen3_alignment.enabled=true in the model config')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device requested but CUDA is unavailable: {args.device}')

    loader = build_inference_loader(
        config=config,
        data=data,
        ntv3_embedding_path=ntv3_embedding_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    logger.info('Loading trusted Lightning checkpoint: %s', checkpoint)
    module = load_model(checkpoint, config, device)
    embeddings = extract_embeddings(
        module=module,
        loader=loader,
        strategy=args.embedding_strategy,
        device=device,
        disable_progress=args.disable_progress,
    )
    sample_ids = [_sample_id_text(sample_id) for sample_id in data['samples']]
    if embeddings.shape[0] != len(sample_ids):
        raise RuntimeError(
            f'Extracted {embeddings.shape[0]} embeddings for {len(sample_ids)} input samples'
        )

    attrs = {
        'model_class': 'MGM2LightningModule',
        'model_checkpoint': str(checkpoint),
        'model_config': str(config_path),
        'embedding_strategy': args.embedding_strategy,
        'source_data_file': str(data_file),
        'ntv3_embedding_path': str(ntv3_embedding_path),
        'hidden_size': int(config.model.hidden_size),
    }
    save_embeddings(args.output, embeddings, sample_ids, attrs)
    logger.info('Saved %s embeddings with shape %s to %s', args.embedding_strategy, embeddings.shape, args.output)


if __name__ == '__main__':
    main()
