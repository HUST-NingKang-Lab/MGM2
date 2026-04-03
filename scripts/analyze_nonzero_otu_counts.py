#!/usr/bin/env python3
"""Analyze non-zero OTU count distribution and top-K abundance truncation trade-offs.

Example:
    python scripts/analyze_nonzero_otu_counts.py \
        --data_file data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.pkl \
        --output_dir outputs_analysis/nonzero_otu_counts
"""

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

from src.data.utils import load_preprocessed_data

logger = logging.getLogger(__name__)
DEFAULT_TOP_K_VALUES = (512, 768, 1024, 1536, 2048)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze non-zero OTU count distribution and top-K truncation trade-offs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Example:\n'
            '  python scripts/analyze_nonzero_otu_counts.py '
            '--data_file data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.pkl '
            '--output_dir outputs_analysis/nonzero_otu_counts'
        ),
    )
    parser.add_argument('--data_file', type=Path, required=True, help='Path to the MGM2 preprocessed dataset (.pkl)')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory for summary tables and plots')
    parser.add_argument('--hist_bins', type=int, default=50, help='Number of bins for the non-zero-count histogram')
    parser.add_argument(
        '--top_k_values',
        type=int,
        nargs='+',
        default=list(DEFAULT_TOP_K_VALUES),
        help='Candidate top-K OTU truncation values to analyze',
    )
    parser.add_argument(
        '--floor_abundance',
        type=float,
        default=1e-4,
        help='Abundance floor used later in preprocessing; used to diagnose floor-heavy tails',
    )
    return parser.parse_args()


def compute_nonzero_otu_counts(data: Dict) -> pd.DataFrame:
    sample_ids = data.get('samples', [])
    otu_data = data.get('otu_data', {})
    rows: List[Dict[str, int]] = []
    for sample_id in sample_ids:
        sample = otu_data[sample_id]
        input_ids = np.asarray(sample['input_ids'])
        nonzero_count = int(np.sum(input_ids != -1))
        rows.append({'sample_id': str(sample_id), 'nonzero_otu_count': nonzero_count})
    return pd.DataFrame(rows)


def summarize_counts(count_table: pd.DataFrame) -> Dict[str, float]:
    counts = count_table['nonzero_otu_count'].to_numpy(dtype=np.int64)
    return {
        'num_samples': int(len(counts)),
        'mean': float(np.mean(counts)),
        'std': float(np.std(counts)),
        'min': int(np.min(counts)),
        'p25': float(np.percentile(counts, 25)),
        'median': float(np.percentile(counts, 50)),
        'p75': float(np.percentile(counts, 75)),
        'p90': float(np.percentile(counts, 90)),
        'p95': float(np.percentile(counts, 95)),
        'p99': float(np.percentile(counts, 99)),
        'max': int(np.max(counts)),
    }


def _iter_positive_abundances(data: Dict) -> Iterable[np.ndarray]:
    for sample_id in data.get('samples', []):
        sample = data['otu_data'][sample_id]
        abundances = np.asarray(sample['abundance_values'], dtype=np.float32)
        positive = abundances[abundances > 0]
        yield positive


def summarize_top_k_tradeoffs(data: Dict, top_k_values: List[int], floor_abundance: float) -> pd.DataFrame:
    unique_top_k = sorted({int(value) for value in top_k_values if int(value) > 0})
    if not unique_top_k:
        raise ValueError('top_k_values must contain at least one positive integer')

    positive_abundances = list(_iter_positive_abundances(data))
    nonzero_counts = np.array([len(values) for values in positive_abundances], dtype=np.int64)
    total_samples = max(1, len(nonzero_counts))
    rows = []

    for top_k in unique_top_k:
        kth_values: List[float] = []
        for values in positive_abundances:
            if len(values) < top_k:
                continue
            kth_values.append(float(np.partition(values, len(values) - top_k)[len(values) - top_k]))

        kth_array = np.asarray(kth_values, dtype=np.float32)
        eligible_samples = int(len(kth_array))
        truncated_samples = int(np.sum(nonzero_counts > top_k))
        floor_samples = int(np.sum(kth_array <= floor_abundance)) if eligible_samples else 0

        row = {
            'top_k_otus': int(top_k),
            'eligible_samples': eligible_samples,
            'eligible_fraction': float(eligible_samples / total_samples),
            'truncated_samples': truncated_samples,
            'truncated_fraction': float(truncated_samples / total_samples),
            'floor_or_below_samples': floor_samples,
            'floor_or_below_fraction': float(floor_samples / eligible_samples) if eligible_samples else 0.0,
        }
        if eligible_samples:
            row.update(
                {
                    'kth_abundance_mean': float(np.mean(kth_array)),
                    'kth_abundance_median': float(np.percentile(kth_array, 50)),
                    'kth_abundance_p10': float(np.percentile(kth_array, 10)),
                    'kth_abundance_p25': float(np.percentile(kth_array, 25)),
                    'kth_abundance_p75': float(np.percentile(kth_array, 75)),
                    'kth_abundance_p90': float(np.percentile(kth_array, 90)),
                }
            )
        else:
            row.update(
                {
                    'kth_abundance_mean': float('nan'),
                    'kth_abundance_median': float('nan'),
                    'kth_abundance_p10': float('nan'),
                    'kth_abundance_p25': float('nan'),
                    'kth_abundance_p75': float('nan'),
                    'kth_abundance_p90': float('nan'),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def _pick_first_top_k(frame: pd.DataFrame) -> Optional[int]:
    if frame.empty:
        return None
    return int(frame.iloc[0]['top_k_otus'])


def build_top_k_recommendations(tradeoff_table: pd.DataFrame) -> Dict[str, Optional[int]]:
    balanced = tradeoff_table[
        (tradeoff_table['truncated_fraction'] <= 0.05)
        & (tradeoff_table['floor_or_below_fraction'] <= 0.25)
    ]
    conservative = tradeoff_table[tradeoff_table['truncated_fraction'] <= 0.01]
    aggressive = tradeoff_table[
        (tradeoff_table['truncated_fraction'] <= 0.12)
        & (tradeoff_table['floor_or_below_fraction'] <= 0.10)
    ]
    return {
        'balanced': _pick_first_top_k(balanced),
        'conservative': _pick_first_top_k(conservative),
        'aggressive': _pick_first_top_k(aggressive),
    }


def save_histogram(count_table: pd.DataFrame, output_dir: Path, hist_bins: int):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    matplotlib.rcParams['pdf.fonttype'] = 42

    plt.figure(figsize=(10, 6))
    sns.histplot(count_table['nonzero_otu_count'], bins=hist_bins, kde=False)
    plt.xlabel('Non-zero OTUs per sample')
    plt.ylabel('Number of samples')
    plt.title('Distribution of non-zero OTU counts')
    plt.tight_layout()
    plt.savefig(output_dir / 'nonzero_otu_count_hist.png', dpi=300)
    plt.savefig(output_dir / 'nonzero_otu_count_hist.pdf')
    plt.close()


def save_top_k_tradeoff_plot(tradeoff_table: pd.DataFrame, output_dir: Path):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams['pdf.fonttype'] = 42

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        tradeoff_table['top_k_otus'],
        tradeoff_table['truncated_fraction'],
        marker='o',
        label='Truncated sample fraction',
    )
    ax1.plot(
        tradeoff_table['top_k_otus'],
        tradeoff_table['floor_or_below_fraction'],
        marker='s',
        label='Rank-K <= floor fraction',
    )
    ax1.set_xlabel('top_k_otus')
    ax1.set_ylabel('Fraction of samples')
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(
        tradeoff_table['top_k_otus'],
        tradeoff_table['kth_abundance_median'],
        color='tab:green',
        marker='^',
        label='Median rank-K abundance',
    )
    ax2.set_ylabel('Median rank-K abundance')
    ax2.set_yscale('log')

    fig.tight_layout()
    fig.savefig(output_dir / 'top_k_tradeoff.png', dpi=300)
    fig.savefig(output_dir / 'top_k_tradeoff.pdf')
    plt.close(fig)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_preprocessed_data(str(args.data_file))
    count_table = compute_nonzero_otu_counts(data)
    if count_table.empty:
        raise ValueError('No samples were found in the provided dataset')

    summary = summarize_counts(count_table)
    tradeoff_table = summarize_top_k_tradeoffs(data, args.top_k_values, args.floor_abundance)
    recommendations = build_top_k_recommendations(tradeoff_table)

    count_table.to_csv(args.output_dir / 'nonzero_otu_count_table.csv', index=False)
    tradeoff_table.to_csv(args.output_dir / 'top_k_tradeoff_summary.csv', index=False)

    (args.output_dir / 'nonzero_otu_count_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    (args.output_dir / 'top_k_tradeoff_summary.json').write_text(
        json.dumps(
            {
                'floor_abundance': args.floor_abundance,
                'top_k_values': [int(v) for v in tradeoff_table['top_k_otus'].tolist()],
                'recommendations': recommendations,
                'rows': tradeoff_table.to_dict(orient='records'),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    save_histogram(count_table, args.output_dir, args.hist_bins)
    save_top_k_tradeoff_plot(tradeoff_table, args.output_dir)

    logger.info('Saved non-zero OTU count analysis to %s', args.output_dir)
    logger.info('Count summary: %s', json.dumps(summary, ensure_ascii=False))
    logger.info('Top-K recommendations: %s', json.dumps(recommendations, ensure_ascii=False))
    print(json.dumps({'count_summary': summary, 'top_k_recommendations': recommendations}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
