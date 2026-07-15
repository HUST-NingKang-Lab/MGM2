#!/usr/bin/env python3
"""Assemble MGM2 checkpoints and embedding resources for Hugging Face upload."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SCALES = ('small', 'medium', 'large', 'xlarge')


HF_FRONT_MATTER = """---
library_name: pytorch
pipeline_tag: feature-extraction
tags:
- microbiome
- microbial-community
- transformer
- feature-extraction
---

"""


def render_model_card() -> str:
    return HF_FRONT_MATTER + (PROJECT_ROOT / 'README.md').read_text()


GIT_ATTRIBUTES = """*.ckpt filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
"""


def parse_args() -> argparse.Namespace:
    project_root = PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description='Prepare a drag-and-drop Hugging Face model repository for MGM2',
    )
    parser.add_argument(
        '--stage2_root',
        type=Path,
        default=project_root / 'outputs_grid/stage2/20260505_094404',
        help='Stage-2 directory containing scale_<name>/config.json and checkpoints/last.ckpt',
    )
    parser.add_argument(
        '--reference_embedding',
        type=Path,
        default=project_root / 'ntv3_embeddings/otus_99_embeddings.h5',
    )
    parser.add_argument(
        '--fallback_embedding',
        type=Path,
        default=project_root / 'ntv3_embeddings/otus_99_taxonomy_fallback.h5',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=project_root / 'huggingface_upload/MGM2',
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy embedding resources instead of using space-saving hard links',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace files in an existing output directory',
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{description} not found: {path}')
    return path


def place_artifact(source: Path, destination: Path, copy_files: bool, overwrite: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f'Output already exists: {destination}; pass --overwrite to replace it')
        destination.unlink()

    if not copy_files:
        try:
            os.link(source, destination)
            return 'hardlink'
        except OSError:
            pass

    shutil.copy2(source, destination)
    return 'copy'


def write_inference_checkpoint(source: Path, destination: Path, overwrite: bool) -> str:
    """Write a portable checkpoint containing only weights and loader metadata."""
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f'Output already exists: {destination}; pass --overwrite to replace it')
        destination.unlink()

    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if 'state_dict' not in checkpoint:
        raise KeyError(f'Lightning checkpoint has no state_dict: {source}')

    release_checkpoint = {
        'state_dict': checkpoint['state_dict'],
        'pytorch-lightning_version': checkpoint.get('pytorch-lightning_version', 'unknown'),
        'mgm2_release': {
            'checkpoint_selection': 'last',
            'format': 'inference_only_v1',
        },
    }
    torch.save(release_checkpoint, destination)
    del checkpoint
    del release_checkpoint
    return 'inference_checkpoint'


def sanitized_config(source: Path, scale: str) -> Dict:
    config = json.loads(source.read_text())
    config['data_path'] = ''
    config['ntv3_embedding_path'] = '../../embeddings/reference_ntv3_embeddings.h5'
    config['output_dir'] = './outputs'
    config['experiment_name'] = f'MGM2-{scale}'
    config['qwen3_alignment']['embedding_path'] = ''
    config['training']['devices'] = [0]
    config['training']['num_workers'] = 0
    return config


def iter_manifest_files(root: Path) -> Iterable[Dict]:
    for path in sorted(item for item in root.rglob('*') if item.is_file() and item.name != 'manifest.json'):
        yield {
            'path': path.relative_to(root).as_posix(),
            'size_bytes': path.stat().st_size,
        }


def assert_portable_config(config: Dict, scale: str) -> None:
    serialized = json.dumps(config)
    forbidden = ('/data', '/home/', 'zhanghaohong', 'microbeatlas_keywords_emb')
    matches = [token for token in forbidden if token in serialized]
    if matches:
        raise ValueError(f'Sanitized {scale} config still contains local values: {matches}')


def validate_model_release(model_dir: Path) -> None:
    import torch

    from src.configs.Config import MGM2Config
    from src.training.TrainerModule import MGM2LightningModule

    checkpoint_path = model_dir / 'model.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    forbidden_keys = {'hyper_parameters', 'optimizer_states', 'lr_schedulers', 'callbacks', 'loops'}
    leaked_keys = forbidden_keys.intersection(checkpoint)
    if leaked_keys:
        raise ValueError(f'Inference checkpoint contains training-only keys: {sorted(leaked_keys)}')
    if 'state_dict' not in checkpoint:
        raise KeyError(f'Inference checkpoint has no state_dict: {checkpoint_path}')
    del checkpoint

    config = MGM2Config.load(str(model_dir / 'config.json'))
    module = MGM2LightningModule.load_from_checkpoint(
        str(checkpoint_path),
        config=config,
        map_location='cpu',
        weights_only=False,
    )
    expected_hidden_size = int(config.model.hidden_size)
    if int(module.model.hidden_size) != expected_hidden_size:
        raise ValueError(
            f'Loaded hidden size {module.model.hidden_size} does not match config {expected_hidden_size}'
        )
    del module


def validate_embedding_release(reference_path: Path, fallback_path: Path) -> None:
    import h5py

    with h5py.File(reference_path, 'r') as handle:
        missing = {'embeddings', 'otu_ids'}.difference(handle.keys())
        if missing:
            raise KeyError(f'Reference embedding is missing datasets: {sorted(missing)}')
    with h5py.File(fallback_path, 'r') as handle:
        required = {'taxon_embeddings', 'taxon_keys', 'taxon_ranks', 'taxon_paths', 'global_embedding'}
        missing = required.difference(handle.keys())
        if missing:
            raise KeyError(f'Taxonomy fallback is missing datasets: {sorted(missing)}')


def build_bundle(args: argparse.Namespace) -> Tuple[Path, Dict[str, str]]:
    stage2_root = args.stage2_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    reference_embedding = require_file(args.reference_embedding, 'Reference NTv3 embedding')
    fallback_embedding = require_file(args.fallback_embedding, 'Taxonomy fallback embedding')
    pipeline_image = require_file(PROJECT_ROOT / 'pipeline.png', 'Pipeline figure')

    if output.exists() and not args.overwrite:
        raise FileExistsError(f'Output directory already exists: {output}; pass --overwrite to rebuild it')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pipeline_image, output / 'pipeline.png')

    placement_modes: Dict[str, str] = {}
    for scale in SCALES:
        source_dir = stage2_root / f'scale_{scale}'
        checkpoint = require_file(source_dir / 'checkpoints/last.ckpt', f'{scale} last checkpoint')
        config_path = require_file(source_dir / 'config.json', f'{scale} config')

        model_dir = output / 'models' / scale
        model_dir.mkdir(parents=True, exist_ok=True)
        config = sanitized_config(config_path, scale)
        assert_portable_config(config, scale)
        (model_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        mode = write_inference_checkpoint(
            checkpoint,
            model_dir / 'model.ckpt',
            overwrite=args.overwrite,
        )
        placement_modes[f'models/{scale}/model.ckpt'] = mode

    embedding_dir = output / 'embeddings'
    placement_modes['embeddings/reference_ntv3_embeddings.h5'] = place_artifact(
        reference_embedding,
        embedding_dir / 'reference_ntv3_embeddings.h5',
        copy_files=args.copy,
        overwrite=args.overwrite,
    )
    placement_modes['embeddings/taxonomy_fallback.h5'] = place_artifact(
        fallback_embedding,
        embedding_dir / 'taxonomy_fallback.h5',
        copy_files=args.copy,
        overwrite=args.overwrite,
    )

    for scale in SCALES:
        validate_model_release(output / 'models' / scale)
    validate_embedding_release(
        embedding_dir / 'reference_ntv3_embeddings.h5',
        embedding_dir / 'taxonomy_fallback.h5',
    )

    (output / 'README.md').write_text(render_model_card())
    (output / '.gitattributes').write_text(GIT_ATTRIBUTES)

    manifest = {
        'bundle_format': 'mgm2_huggingface_v1',
        'checkpoint_selection': 'last',
        'scales': list(SCALES),
        'artifact_placement': placement_modes,
        'validation': {
            'models': 'MGM2LightningModule.load_from_checkpoint',
            'embeddings': 'required_hdf5_datasets',
        },
    }
    manifest['files'] = list(iter_manifest_files(output))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output, placement_modes


def main() -> None:
    args = parse_args()
    output, placement_modes = build_bundle(args)
    print(f'Prepared MGM2 Hugging Face bundle: {output}')
    for relative_path, mode in placement_modes.items():
        print(f'  {mode:8s} {relative_path}')
    print('Upload the contents of this directory to one Hugging Face model repository.')


if __name__ == '__main__':
    main()
