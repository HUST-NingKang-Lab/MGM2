#!/usr/bin/env python3
"""Split MGnify MST hierarchical metadata labels into per-level columns.

Example:
    python scripts/downstreams/MGnify_mst/split_mgnify_mst_metadata.py
"""

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


def split_label(value: object) -> List[str]:
    """Split root:environment:... or root;environment;... labels and remove root."""
    if pd.isna(value):
        parts = []
    else:
        parts = [part.strip() for part in re.split(r'[;:]', str(value)) if part.strip()]
    if parts and parts[0].lower() == 'root':
        parts = parts[1:]
    return parts


def split_metadata(
    input_path: Path,
    output_path: Path,
    sample_id_column: str = 'SampleID',
    label_column: str = 'Env',
    max_depth: int = 5,
) -> pd.DataFrame:
    metadata = pd.read_csv(input_path)
    if sample_id_column not in metadata.columns or label_column not in metadata.columns:
        raise KeyError(f'{input_path} must contain {sample_id_column!r} and {label_column!r}')

    labels = metadata[label_column].map(split_label)
    output = metadata[[sample_id_column, label_column]].copy()
    for level_idx in range(max_depth):
        output[f'env_level_{level_idx + 1}'] = labels.map(
            lambda parts, idx=level_idx: parts[idx] if idx < len(parts) else 'unknown'
        )
    output['env_depth'] = labels.map(len)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description='Split MGnify MST metadata Env hierarchy into level columns')
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=Path('/data9/zhanghaohong/projects/MGM2Benchmarks/MGnify_mst/expert'),
        help='Directory containing train_meta.csv and test_meta.csv',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/downstreams/MGnify_mst'),
        help='Directory for split metadata CSV files',
    )
    parser.add_argument('--sample_id_column', default='SampleID', help='Sample id column in metadata')
    parser.add_argument('--label_column', default='Env', help='Hierarchical label column in metadata')
    parser.add_argument('--max_depth', type=int, default=5, help='Number of non-root hierarchy levels to write')
    return parser.parse_args()


def main():
    args = parse_args()
    for split in ('train', 'test'):
        input_path = args.input_dir / f'{split}_meta.csv'
        output_path = args.output_dir / f'{split}_meta_split.csv'
        output = split_metadata(
            input_path=input_path,
            output_path=output_path,
            sample_id_column=args.sample_id_column,
            label_column=args.label_column,
            max_depth=args.max_depth,
        )
        print(f'{split}: wrote {len(output):,} rows to {output_path}')
        print(output.filter(regex=r'^env_level_').nunique().to_string())


if __name__ == '__main__':
    main()
