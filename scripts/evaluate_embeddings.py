#!/usr/bin/env python3
"""Evaluate MGM2 sample embeddings from a pretrained checkpoint."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


LABEL_KEYWORDS = ('aquatic', 'animal', 'soil', 'plant')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate MGM2 sample embeddings from a checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python scripts/evaluate_embeddings.py --checkpoint outputs/checkpoints/last.ckpt '
            '--data_file data/pretrain_dataset.pkl --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5 '
            '--metadata_file samples.env.info.tsv --sample_id_column MAP_SID --label_column Environments '
            '--embedding_strategy cls --output_dir outputs_eval/run_x'
        ),
    )
    parser.add_argument('--checkpoint', type=Path, required=True, help='Lightning checkpoint to evaluate')
    parser.add_argument('--config', type=Path, default=None, help='Optional config JSON. Defaults to the nearest config.json found above the checkpoint path')
    parser.add_argument('--data_file', type=Path, required=True, help='Preprocessed dataset (.pkl)')
    parser.add_argument('--ntv3_embedding_path', type=Path, required=True, help='NTv3 embeddings (.h5)')
    parser.add_argument('--metadata_file', type=Path, required=True, help='Metadata TSV/CSV with sample labels')
    parser.add_argument('--sample_id_column', type=str, required=True, help='Metadata column holding sample IDs')
    parser.add_argument('--label_column', type=str, required=True, help='Metadata label column used for evaluation')
    parser.add_argument('--metadata_sep', type=str, default='\t', help='Metadata separator, default tab')
    parser.add_argument('--embedding_strategy', choices=['cls', 'mean_pool', 'qwen3_proj'], default='cls')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory to write evaluation outputs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_samples_per_label', type=int, default=1000)
    parser.add_argument('--min_samples_per_label', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--no_umap', action='store_true', help='Skip UMAP even if umap-learn is installed')
    parser.add_argument('--disable_progress', action='store_true', help='Disable tqdm progress bars during evaluation')
    return parser.parse_args()


def select_sample_embeddings(outputs: Dict[str, torch.Tensor], attention_mask: torch.Tensor, strategy: str) -> np.ndarray:
    if strategy == 'cls':
        return outputs['cls_hidden'].detach().cpu().numpy().astype(np.float32)
    if strategy == 'qwen3_proj':
        if 'qwen3_proj' not in outputs:
            raise ValueError('qwen3_proj requested but checkpoint/config does not enable Qwen3 alignment')
        return outputs['qwen3_proj'].detach().cpu().numpy().astype(np.float32)
    if strategy != 'mean_pool':
        raise ValueError(f'Unsupported embedding strategy: {strategy}')

    hidden_states = outputs['hidden_states']
    token_mask = attention_mask.clone()
    if token_mask.size(1) > 0:
        token_mask[:, 0] = 0  # exclude CLS from mean pooling
    mask = token_mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    mean_pool = (hidden_states * mask).sum(dim=1) / denom
    return mean_pool.detach().cpu().numpy().astype(np.float32)


def filter_metadata_by_keywords(metadata: pd.DataFrame, label_column: str, keywords: Iterable[str]) -> pd.DataFrame:
    keyword_list = tuple(keyword.lower() for keyword in keywords)
    label_series = metadata[label_column].astype(str).str.lower()
    mask = label_series.apply(lambda value: any(keyword in value for keyword in keyword_list))
    return metadata[mask].copy()


def sample_metadata_balanced(
    metadata: pd.DataFrame,
    label_column: str,
    max_samples_per_label: int,
    seed: int,
    min_samples_per_label: int = 1,
) -> pd.DataFrame:
    grouped_frames: List[pd.DataFrame] = []
    for _, group in metadata.groupby(label_column, sort=True):
        if len(group) < min_samples_per_label:
            continue
        n = min(len(group), max_samples_per_label)
        grouped_frames.append(group.sample(n=n, random_state=seed) if len(group) > n else group.copy())
    if not grouped_frames:
        raise ValueError('No labels have enough samples after metadata filtering')
    return pd.concat(grouped_frames, axis=0).reset_index(drop=True)


def _resolve_config_path(checkpoint_path: Path, explicit_config: Optional[Path]) -> Path:
    if explicit_config is not None:
        resolved_config = explicit_config.resolve()
        if resolved_config.exists():
            return resolved_config
        raise FileNotFoundError(f'Config not found: {resolved_config}')

    resolved_checkpoint = checkpoint_path.resolve()
    for ancestor in resolved_checkpoint.parents:
        candidate = ancestor / 'config.json'
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f'Could not infer config.json from checkpoint path {resolved_checkpoint}; pass --config explicitly'
    )


def _prepare_eval_components(config):
    from src.data.Preprocessing import AbundancePreprocessor, BinManager
    from src.data.utils import load_preprocessed_data

    data = load_preprocessed_data(config.data_path)
    abundance_preprocessor = AbundancePreprocessor(
        min_abundance=config.abundance.min_abundance,
        output_min=config.abundance.output_min,
        output_max=config.abundance.output_max,
        normalize=config.abundance.normalize,
    )
    abundance_preprocessor.fit([])
    bin_manager = BinManager(
        num_bins=config.abundance.num_bins,
        method=config.abundance.binning_method,
        fixed_thresholds=config.abundance.fixed_thresholds,
        enable_regression=config.abundance.enable_regression,
        regression_top_k_bins=config.abundance.regression_top_k_bins,
    )
    bin_manager.fit([])
    return data, abundance_preprocessor, bin_manager


def _subset_preprocessed_data(data: Dict, sample_ids: Iterable[str]) -> Dict:
    sample_id_set = set(sample_ids)
    subset = {}
    n_samples = len(data['samples'])
    selected_samples = [sample_id for sample_id in data['samples'] if sample_id in sample_id_set]
    for key, value in data.items():
        if key == 'samples':
            subset[key] = selected_samples
        elif key == 'otu_data' and isinstance(value, dict):
            subset[key] = {sample_id: value[sample_id] for sample_id in selected_samples}
        elif isinstance(value, list) and len(value) == n_samples:
            index_map = {sample_id: idx for idx, sample_id in enumerate(data['samples'])}
            subset[key] = [value[index_map[sample_id]] for sample_id in selected_samples]
        else:
            subset[key] = value
    return subset


def _wrap_progress(iterable, total: Optional[int], desc: str, disable: bool = False):
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        logger.info('tqdm is not installed; proceeding without progress bars for %s', desc)
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _load_model(checkpoint_path: Path, config, device: torch.device):
    from src.training.TrainerModule import MGM2LightningModule

    # PyTorch 2.6 defaults torch.load(..., weights_only=True), but Lightning checkpoints
    # store trusted MGM2Config objects in hyperparameters and must be loaded as full checkpoints.
    module = MGM2LightningModule.load_from_checkpoint(
        str(checkpoint_path),
        config=config,
        map_location=device,
        weights_only=False,
    )
    module.eval()
    module.freeze()
    module.to(device)
    return module


def _build_eval_loader(config, sample_ids: List[str], batch_size: int, num_workers: int):
    from src.data.Dataset import BatchCollatorMGM2, MicrobiomeDatasetMGM2

    data, abundance_preprocessor, bin_manager = _prepare_eval_components(config)
    subset_data = _subset_preprocessed_data(data, sample_ids)
    dataset = MicrobiomeDatasetMGM2(
        preprocessed_data=subset_data,
        abundance_preprocessor=abundance_preprocessor,
        bin_manager=bin_manager,
        mask_prob=config.training.mask_prob,
        ntv3_embedding_path=config.ntv3_embedding_path,
        qwen3_embedding_path=config.qwen3_alignment.embedding_path if config.qwen3_alignment.enabled else None,
        qwen3_projection_dim=config.qwen3_alignment.projection_dim,
        enable_qwen3_alignment=config.qwen3_alignment.enabled,
        is_training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=BatchCollatorMGM2(),
        pin_memory=True,
    )
    return loader, subset_data['samples']


def _extract_embeddings(
    module,
    loader: DataLoader,
    sample_ids: List[str],
    strategy: str,
    device: torch.device,
    show_progress: bool = True,
):
    all_embeddings = []
    collected_ids: List[str] = []
    offset = 0
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    batch_iterator = _wrap_progress(
        loader,
        total=total_batches,
        desc='Extracting embeddings',
        disable=not show_progress,
    )
    with torch.no_grad():
        for batch in batch_iterator:
            batch_size = batch['input_ids'].size(0)
            batch_on_device = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = module.model(**module._extract_model_inputs(batch_on_device))
            embeddings = select_sample_embeddings(outputs, batch_on_device['attention_mask'], strategy=strategy)
            all_embeddings.append(embeddings)
            collected_ids.extend(sample_ids[offset:offset + batch_size])
            offset += batch_size
    return np.concatenate(all_embeddings, axis=0), collected_ids


def _compute_label_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, davies_bouldin_score, silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier

    metrics = {'num_samples': int(len(labels)), 'num_labels': int(len(np.unique(labels)))}
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return metrics
    metrics['silhouette'] = float(silhouette_score(embeddings, labels))
    metrics['davies_bouldin'] = float(davies_bouldin_score(embeddings, labels))
    label_counts = pd.Series(labels).value_counts()
    n_splits = min(5, int(label_counts.min()))
    if n_splits >= 2:
        clf = KNeighborsClassifier(n_neighbors=min(5, max(1, len(labels) - 1)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        pred = cross_val_predict(clf, embeddings, labels, cv=cv)
        metrics['knn_balanced_accuracy'] = float(balanced_accuracy_score(labels, pred))
    return metrics


def _save_plot(coords: np.ndarray, labels: List[str], output_path: Path, title: str):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    matplotlib.rcParams['pdf.fonttype'] = 42

    df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'label': labels})
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', s=25, alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    base_path = output_path.with_suffix('')
    plt.savefig(base_path.with_suffix('.png'), dpi=300)
    plt.savefig(base_path.with_suffix('.pdf'))
    plt.close()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    config_path = _resolve_config_path(checkpoint_path, args.config.resolve() if args.config else None)
    from src.configs.Config import MGM2Config
    from src.data.utils import load_preprocessed_data

    config = MGM2Config.load(str(config_path))
    config.data_path = str(args.data_file.resolve())
    config.ntv3_embedding_path = str(args.ntv3_embedding_path.resolve())
    if args.embedding_strategy == 'qwen3_proj' and not config.qwen3_alignment.enabled:
        raise ValueError('qwen3_proj evaluation requires a checkpoint/config with qwen3_alignment.enabled=true')

    metadata = pd.read_csv(args.metadata_file, sep=args.metadata_sep)
    if args.sample_id_column not in metadata.columns or args.label_column not in metadata.columns:
        raise KeyError('metadata is missing the requested sample_id_column or label_column')
    metadata = metadata[[args.sample_id_column, args.label_column]].dropna().copy()
    metadata[args.sample_id_column] = metadata[args.sample_id_column].astype(str)
    metadata[args.label_column] = metadata[args.label_column].astype(str)

    data = load_preprocessed_data(config.data_path)
    dataset_sample_ids = set(map(str, data['samples']))
    metadata = metadata[metadata[args.sample_id_column].isin(dataset_sample_ids)].copy()
    if metadata.empty:
        raise ValueError('No metadata rows matched sample IDs in the dataset')

    metadata = filter_metadata_by_keywords(metadata, args.label_column, LABEL_KEYWORDS)
    if metadata.empty:
        raise ValueError(f'No metadata rows matched label keywords: {LABEL_KEYWORDS}')
    logger.info('Retained %d samples after keyword filtering with %s', len(metadata), ','.join(LABEL_KEYWORDS))

    metadata = sample_metadata_balanced(
        metadata,
        label_column=args.label_column,
        max_samples_per_label=args.max_samples_per_label,
        min_samples_per_label=args.min_samples_per_label,
        seed=args.seed,
    )
    sample_ids = metadata[args.sample_id_column].tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Building evaluation dataloader for %d samples', len(sample_ids))
    loader, ordered_sample_ids = _build_eval_loader(config, sample_ids, args.batch_size, args.num_workers)
    metadata = metadata.set_index(args.sample_id_column).loc[ordered_sample_ids].reset_index()
    labels = metadata[args.label_column].tolist()

    device = torch.device(args.device)
    logger.info('Loading checkpoint from %s', checkpoint_path)
    module = _load_model(checkpoint_path, config, device)
    logger.info('Extracting %s embeddings on %s', args.embedding_strategy, device)
    embeddings, ordered_sample_ids = _extract_embeddings(
        module,
        loader,
        ordered_sample_ids,
        args.embedding_strategy,
        device,
        show_progress=not args.disable_progress,
    )

    logger.info('Computing label metrics')
    metrics = _compute_label_metrics(embeddings, np.asarray(labels))
    metrics.update(
        {
            'embedding_strategy': args.embedding_strategy,
            'checkpoint': str(checkpoint_path),
            'config': str(config_path),
        }
    )

    np.savez_compressed(
        args.output_dir / 'embeddings.npz',
        embeddings=embeddings,
        sample_ids=np.asarray(ordered_sample_ids),
        labels=np.asarray(labels),
    )
    embedding_df = pd.DataFrame(embeddings)
    embedding_df.insert(0, 'label', labels)
    embedding_df.insert(0, 'sample_id', ordered_sample_ids)
    embedding_df.to_csv(args.output_dir / 'embedding_table.csv', index=False)
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    if embeddings.shape[0] >= 2:
        logger.info('Saving PCA visualization')
        from sklearn.decomposition import PCA

        pca_coords = PCA(n_components=2, random_state=args.seed).fit_transform(embeddings)
        _save_plot(pca_coords, labels, args.output_dir / 'pca.png', f'PCA of {args.embedding_strategy} embeddings')
    if not args.no_umap and embeddings.shape[0] >= 2:
        try:
            import umap  # type: ignore

            logger.info('Saving UMAP visualization')
            umap_coords = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=args.seed).fit_transform(embeddings)
            _save_plot(umap_coords, labels, args.output_dir / 'umap.png', f'UMAP of {args.embedding_strategy} embeddings')
        except ImportError:
            from sklearn.manifold import TSNE

            logger.info('UMAP unavailable; saving t-SNE visualization instead')
            tsne_coords = TSNE(n_components=2, random_state=args.seed, init='pca', learning_rate='auto').fit_transform(embeddings)
            _save_plot(tsne_coords, labels, args.output_dir / 'tsne.png', f't-SNE of {args.embedding_strategy} embeddings')

    logger.info('Saved evaluation outputs to %s', args.output_dir)
    logger.info('Metrics: %s', json.dumps(metrics, ensure_ascii=False))


if __name__ == '__main__':
    main()
