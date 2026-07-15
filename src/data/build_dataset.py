#!/usr/bin/env python3
"""
Build MGM2 preprocessed datasets from CSV/TSV/HDF5/BIOM community tables.

Optional preprocessing can keep only the top-K OTUs by relative abundance per sample.

Example:
    python src/data/build_dataset.py         --otu_table data/otu_table.csv         --format csv         --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5         --output data/pretrain_dataset.pkl

    python src/data/build_dataset.py         --otu_table data/otu_table.biom         --format biom         --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5         --output data/pretrain_dataset.pkl

Notes:
- Only non-zero OTUs are stored for each sample to reduce dataset size.
- The output stores raw relative abundance values; log preprocessing happens later in Dataset.py.
- The generated pickle is the direct input for src/training/Pretrain.py.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

try:
    from biom import load_table
    BIOM_AVAILABLE = True
except ImportError:
    BIOM_AVAILABLE = False
    logging.warning('BIOM library not available. Install with: pip install biom-format')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MGM2DatasetBuilder:
    """Builds MGM2 training datasets from OTU tables and NTv3 embeddings."""

    TAXON_RANKS = ('kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species')
    TAXON_PREFIX_TO_RANK = {
        'sk': 'kingdom',
        'k': 'kingdom',
        'd': 'kingdom',
        'p': 'phylum',
        'c': 'class',
        'o': 'order',
        'f': 'family',
        'g': 'genus',
        's': 'species',
    }

    def __init__(
        self,
        ntv3_embedding_path: str,
        embedding_dim: int = 1536,
        top_k_otus: int = 0,
        fallback_embedding_path: Optional[str] = None,
        resolved_embedding_path: Optional[str] = None,
    ):
        if top_k_otus < 0:
            raise ValueError('top_k_otus must be non-negative')
        self.ntv3_embedding_path = ntv3_embedding_path
        self.embedding_dim = embedding_dim
        self.top_k_otus = top_k_otus
        self.fallback_embedding_path = fallback_embedding_path
        self.resolved_embedding_path = resolved_embedding_path
        self.otu_id_to_idx: Dict[str, int] = {}
        self.embedding_resolution: Dict = {}

    @staticmethod
    def _validate_non_empty_samples(all_samples: List[str], output_path: str) -> None:
        if not all_samples:
            raise ValueError(
                f'No valid samples were produced for output {output_path}. '
                'Check OTU IDs, abundance values, and NTv3 vocabulary overlap.'
            )

    def load_ntv3_embeddings(self):
        """Load NTv3 OTU ids and build the OTU-id-to-index map."""
        logger.info(f'Loading NTv3 embeddings from {self.ntv3_embedding_path}')
        with h5py.File(self.ntv3_embedding_path, 'r') as f:
            otu_ids = f['otu_ids'][:].astype(str)
            self.otu_id_to_idx = {str(otu_id): idx for idx, otu_id in enumerate(otu_ids)}
        logger.info(f'Loaded {len(self.otu_id_to_idx)} OTU embeddings')

    @staticmethod
    def _decode_h5_strings(values) -> List[str]:
        decoded = []
        for value in values:
            if isinstance(value, (bytes, bytearray, np.bytes_)):
                decoded.append(value.decode('utf-8', errors='replace'))
            else:
                decoded.append(str(value))
        return decoded

    @staticmethod
    def _clean_taxon_name(value: str) -> str:
        value = str(value).strip()
        if not value or value.lower() in {'unclassified', 'unknown', 'none', 'nan'}:
            return ''
        return value

    @classmethod
    def _parse_taxonomy_label(cls, label: str) -> Dict[str, str]:
        """Parse plain or rank-prefixed taxonomy labels."""
        if '__' not in str(label):
            parts = [cls._clean_taxon_name(part) for part in str(label).split(';')]
            parts = [part for part in parts if part]
            return {rank: value for rank, value in zip(cls.TAXON_RANKS, parts)}

        rank_to_name: Dict[str, str] = {}
        for token in str(label).split(';'):
            token = token.strip()
            if '__' not in token:
                continue
            prefix, name = token.split('__', 1)
            rank = cls.TAXON_PREFIX_TO_RANK.get(prefix.strip().lower())
            name = cls._clean_taxon_name(name)
            if rank is not None and name:
                rank_to_name[rank] = name
        return rank_to_name

    @classmethod
    def _build_taxon_keys(cls, label: str) -> List[Tuple[str, str]]:
        rank_to_name = cls._parse_taxonomy_label(label)
        keys = []
        path = []
        for rank in cls.TAXON_RANKS:
            name = rank_to_name.get(rank)
            if not name:
                continue
            path.append(name)
            keys.append((rank, f"{rank}|{';'.join(path)}"))
        return keys

    @staticmethod
    def _normalize_taxon_alias(name: str) -> str:
        return ' '.join(str(name).strip().replace('_', ' ').split()).casefold()

    @classmethod
    def _terminal_alias_candidates(cls, label: str) -> List[Tuple[str, str]]:
        label = str(label).strip()
        if ';' not in label and '__' not in label:
            return [(rank, label) for rank in reversed(cls.TAXON_RANKS)]

        rank_to_name = cls._parse_taxonomy_label(label)
        return [
            (rank, rank_to_name[rank])
            for rank in reversed(cls.TAXON_RANKS)
            if rank in rank_to_name
        ]

    @classmethod
    def _lookup_terminal_alias(
        cls,
        label: str,
        terminal_aliases: Dict[str, Dict[str, str]],
    ) -> Optional[Tuple[str, str]]:
        for rank, name in cls._terminal_alias_candidates(label):
            alias = cls._normalize_taxon_alias(name)
            matched_key = terminal_aliases.get(rank, {}).get(alias)
            if matched_key is not None:
                return rank, matched_key
        return None

    def _load_fallback_embeddings(self):
        if not self.fallback_embedding_path:
            return {}, None, {}
        with h5py.File(self.fallback_embedding_path, 'r') as f:
            taxon_keys = self._decode_h5_strings(f['taxon_keys'][:])
            taxon_embeddings = f['taxon_embeddings'][:].astype(np.float32)
            global_embedding = f['global_embedding'][:].astype(np.float32) if 'global_embedding' in f else None
            taxon_ranks = (
                self._decode_h5_strings(f['taxon_ranks'][:])
                if 'taxon_ranks' in f
                else [key.partition('|')[0] for key in taxon_keys]
            )
            taxon_paths = (
                self._decode_h5_strings(f['taxon_paths'][:])
                if 'taxon_paths' in f
                else [key.partition('|')[2] for key in taxon_keys]
            )

        if len(taxon_embeddings) != len(taxon_keys):
            raise ValueError('Fallback taxon_embeddings and taxon_keys must have the same length')
        if len(taxon_ranks) != len(taxon_keys) or len(taxon_paths) != len(taxon_keys):
            raise ValueError('Fallback taxonomy metadata must align with taxon_keys')

        fallback_map = {key: taxon_embeddings[idx] for idx, key in enumerate(taxon_keys)}
        alias_candidates: Dict[str, Dict[str, set]] = {}
        for key, rank, path in zip(taxon_keys, taxon_ranks, taxon_paths):
            rank = str(rank).strip().lower()
            terminal_name = self._clean_taxon_name(str(path).rsplit(';', 1)[-1])
            if rank not in self.TAXON_RANKS or not terminal_name:
                continue
            alias = self._normalize_taxon_alias(terminal_name)
            alias_candidates.setdefault(rank, {}).setdefault(alias, set()).add(key)

        terminal_aliases = {
            rank: {
                alias: next(iter(keys))
                for alias, keys in rank_aliases.items()
                if len(keys) == 1
            }
            for rank, rank_aliases in alias_candidates.items()
        }
        return fallback_map, global_embedding, terminal_aliases

    def _default_resolved_embedding_path(self, output_path: str) -> str:
        return str(Path(output_path).with_suffix('.resolved_embeddings.h5'))

    def _prepare_feature_index(self, feature_ids: Sequence[str], output_path: str) -> None:
        """
        Resolve table features to an embedding matrix.

        Without fallback embeddings this keeps the historical behavior: input_ids
        are NTv3 indices from the original embedding file. With fallback enabled,
        it writes a compact resolved embedding HDF5 containing one row per table
        feature that can be resolved by direct NTv3 lookup or taxonomy fallback.
        """
        if not self.fallback_embedding_path:
            self.embedding_resolution = {
                'mode': 'ntv3_only',
                'source_counts': {'ntv3': sum(str(feature_id) in self.otu_id_to_idx for feature_id in feature_ids)},
            }
            return

        fallback_map, global_embedding, terminal_aliases = self._load_fallback_embeddings()
        resolved_embeddings = []
        resolved_ids = []
        resolved_sources = []
        resolved_keys = []

        with h5py.File(self.ntv3_embedding_path, 'r') as f:
            ntv3_embeddings = f['embeddings']
            for feature_id in feature_ids:
                feature_id = str(feature_id)
                if feature_id in self.otu_id_to_idx:
                    embedding = np.asarray(ntv3_embeddings[self.otu_id_to_idx[feature_id]], dtype=np.float32)
                    source = 'ntv3'
                    matched_key = ''
                else:
                    embedding = None
                    source = ''
                    matched_key = ''
                    for rank, key in reversed(self._build_taxon_keys(feature_id)):
                        if key in fallback_map:
                            embedding = fallback_map[key]
                            source = f'fallback:{rank}'
                            matched_key = key
                            break
                    if embedding is None:
                        alias_match = self._lookup_terminal_alias(feature_id, terminal_aliases)
                        if alias_match is not None:
                            rank, matched_key = alias_match
                            embedding = fallback_map[matched_key]
                            source = f'fallback:{rank}_name'
                    if embedding is None and global_embedding is not None:
                        embedding = global_embedding
                        source = 'fallback:global'
                    if embedding is None:
                        continue

                resolved_ids.append(feature_id)
                resolved_embeddings.append(embedding)
                resolved_sources.append(source)
                resolved_keys.append(matched_key)

        if not resolved_embeddings:
            raise ValueError('No features could be resolved by NTv3 or taxonomy fallback embeddings')

        resolved_path = self.resolved_embedding_path or self._default_resolved_embedding_path(output_path)
        Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding='utf-8')
        with h5py.File(resolved_path, 'w') as f:
            f.create_dataset('embeddings', data=np.stack(resolved_embeddings).astype(np.float32), compression='gzip')
            f.create_dataset('otu_ids', data=np.asarray(resolved_ids, dtype=object), dtype=string_dtype)
            f.create_dataset('embedding_source', data=np.asarray(resolved_sources, dtype=object), dtype=string_dtype)
            f.create_dataset('matched_taxon_key', data=np.asarray(resolved_keys, dtype=object), dtype=string_dtype)
            f.attrs['source_ntv3_embedding'] = str(self.ntv3_embedding_path)
            f.attrs['source_fallback_embedding'] = str(self.fallback_embedding_path)

        source_counts: Dict[str, int] = {}
        for source in resolved_sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        self.ntv3_embedding_path = resolved_path
        self.otu_id_to_idx = {feature_id: idx for idx, feature_id in enumerate(resolved_ids)}
        self.embedding_resolution = {
            'mode': 'ntv3_with_taxonomy_fallback',
            'resolved_embedding_path': resolved_path,
            'source_counts': source_counts,
        }

    def _keep_top_k_by_relative_abundance(self, otu_ids: np.ndarray, abundances: np.ndarray):
        """Optionally keep only the top-K non-zero OTUs by relative abundance."""
        if self.top_k_otus <= 0 or len(abundances) <= self.top_k_otus:
            return otu_ids, abundances

        top_indices = np.argpartition(abundances, -self.top_k_otus)[-self.top_k_otus:]
        top_indices = top_indices[np.argsort(abundances[top_indices])[::-1]]
        return otu_ids[top_indices], abundances[top_indices]

    def build_sample(self, otu_ids: List[str], abundances: np.ndarray) -> Dict[str, np.ndarray]:
        """Build a single sample, keeping only non-zero OTUs and raw relative abundance values."""
        non_zero_mask = abundances > 0
        non_zero_otu_ids = np.array(otu_ids)[non_zero_mask]
        non_zero_abundances = abundances[non_zero_mask].astype(np.float32)
        non_zero_otu_ids, non_zero_abundances = self._keep_top_k_by_relative_abundance(
            non_zero_otu_ids,
            non_zero_abundances,
        )

        input_ids = [-1]
        abundance_values = [0.0]

        for otu_id, abundance in zip(non_zero_otu_ids, non_zero_abundances):
            if otu_id in self.otu_id_to_idx:
                input_ids.append(self.otu_id_to_idx[otu_id])
                abundance_values.append(float(abundance))

        input_ids = np.array(input_ids, dtype=np.int64)
        abundance_values = np.array(abundance_values, dtype=np.float32)
        attention_mask = np.ones(len(input_ids), dtype=np.int64)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'abundance_values': abundance_values,
        }

    def process_otu_table(
        self,
        otu_table_path: str,
        output_path: str,
        input_format: str = 'csv',
        h5_key: str = 'genus',
        h5_load_all: bool = False,
    ):
        """Process an OTU table and build an MGM2 pretraining dataset."""
        logger.info(f'Processing OTU table from {otu_table_path} (format: {input_format})')
        if input_format == 'biom':
            self._process_biom_table(otu_table_path, output_path)
        elif input_format == 'h5':
            self._process_h5_table(otu_table_path, output_path, h5_key, h5_load_all=h5_load_all)
        else:
            sep = '\t' if input_format == 'tsv' else ','
            self._process_delimited_table(otu_table_path, output_path, sep)

    def _build_dataset_payload(
        self,
        all_samples: List[str],
        otu_data: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict:
        return {
            'samples': all_samples,
            'otu_data': otu_data,
            'otu_id_to_idx': self.otu_id_to_idx,
            'ntv3_embedding_path': self.ntv3_embedding_path,
            'format_version': 'mgm2_v1',
            'preprocessing': {'top_k_otus': self.top_k_otus},
            'embedding_resolution': self.embedding_resolution,
        }

    def _save_dataset(self, dataset: Dict, output_path: str, all_samples: List[str], otu_data: Dict[str, Dict[str, np.ndarray]]):
        total_tokens = sum(len(sample['input_ids']) for sample in otu_data.values())
        logger.info(f'Processed {len(all_samples)} valid samples')
        logger.info(f'Total tokens stored: {total_tokens:,} (avg {total_tokens / len(all_samples):.1f} per sample)')
        logger.info(f'Estimated memory without embeddings: ~{total_tokens * 8 / 1024 / 1024:.2f} MB')

        logger.info(f'Saving dataset to {output_path}')
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            pickle.dump(dataset, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.flush()
            file_size_mb = tmp.tell() / 1024 / 1024
            logger.info(f'Dataset file size: {file_size_mb:.2f} MB')

        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f'Dataset built successfully with {len(all_samples)} samples')

    def _process_delimited_table(
        self,
        otu_table_path: str,
        output_path: str,
        sep: str,
    ):
        otu_df = pd.read_csv(otu_table_path, sep=sep, index_col=0)
        logger.info(f'Loaded OTU table with {otu_df.shape[0]} OTUs and {otu_df.shape[1]} samples')

        otu_ids = otu_df.index.tolist()
        logger.info(f'Number of unique OTUs: {len(otu_ids)}')
        self._prepare_feature_index(otu_ids, output_path)

        otu_df_t = otu_df.T
        logger.info('Calculating relative abundance...')
        row_sums = otu_df_t.sum(axis=1).values
        row_sums[row_sums == 0] = 1.0
        rel_abun = otu_df_t.div(row_sums, axis=0).values

        all_samples: List[str] = []
        otu_data: Dict[str, Dict[str, np.ndarray]] = {}
        for row_idx, (sample_id, _) in enumerate(tqdm(otu_df_t.iterrows(), total=len(otu_df_t))):
            abundances = rel_abun[row_idx]
            sample_data = self.build_sample(otu_ids, abundances)
            if len(sample_data['input_ids']) > 1:
                all_samples.append(sample_id)
                otu_data[sample_id] = sample_data

        self._validate_non_empty_samples(all_samples, output_path)
        dataset = self._build_dataset_payload(all_samples, otu_data)
        self._save_dataset(dataset, output_path, all_samples, otu_data)

    def _process_h5_table(
        self,
        otu_table_path: str,
        output_path: str,
        h5_key: str,
        h5_load_all: bool = False,
    ):
        with h5py.File(otu_table_path, 'r') as f:
            group = f[h5_key]
            feature_ids = self._decode_h5_strings(group['axis1'][:])
            sample_ids = self._decode_h5_strings(group['axis0'][:])
            values_dataset = group['block0_values']
            if values_dataset.shape != (len(feature_ids), len(sample_ids)):
                raise ValueError(
                    f'Expected {h5_key}/block0_values shape '
                    f'({len(feature_ids)}, {len(sample_ids)}), got {values_dataset.shape}'
                )

            self._prepare_feature_index(feature_ids, output_path)
            if h5_load_all:
                logger.info(
                    'Loading full HDF5 abundance matrix into memory: '
                    f'{values_dataset.shape[0]:,} features x {values_dataset.shape[1]:,} samples'
                )
                values = np.asarray(values_dataset[:], dtype=np.float32)
                logger.info(f'Loaded HDF5 matrix into memory: {values.nbytes / 1024 / 1024 / 1024:.2f} GB')
                col_sums = values.sum(axis=0, dtype=np.float64)
            else:
                values = values_dataset
                col_sums = None

            all_samples: List[str] = []
            otu_data: Dict[str, Dict[str, np.ndarray]] = {}
            for sample_idx, sample_id in enumerate(tqdm(sample_ids, total=len(sample_ids))):
                abundances = np.asarray(values[:, sample_idx], dtype=np.float32)
                total = float(col_sums[sample_idx]) if col_sums is not None else float(abundances.sum())
                if total > 0:
                    abundances = abundances / total
                sample_data = self.build_sample(feature_ids, abundances)
                if len(sample_data['input_ids']) > 1:
                    all_samples.append(sample_id)
                    otu_data[sample_id] = sample_data

        self._validate_non_empty_samples(all_samples, output_path)
        dataset = self._build_dataset_payload(all_samples, otu_data)
        dataset['preprocessing']['h5_key'] = h5_key
        dataset['preprocessing']['h5_load_all'] = h5_load_all
        self._save_dataset(dataset, output_path, all_samples, otu_data)

    def _process_biom_table(
        self,
        otu_table_path: str,
        output_path: str,
    ):
        if not BIOM_AVAILABLE:
            raise ImportError('BIOM library is required to process BIOM files. Install with: pip install biom-format')

        logger.info(f'Loading BIOM table from {otu_table_path}')
        if otu_table_path.endswith('.h5') or otu_table_path.endswith('.biom'):
            try:
                logger.info('Using h5py for HDF5 BIOM file')
                with h5py.File(otu_table_path, 'r'):
                    pass
                otu_table = load_table(otu_table_path)
            except Exception as exc:
                logger.warning(f'Failed to use optimized loading: {exc}')
                otu_table = load_table(otu_table_path)
        else:
            otu_table = load_table(otu_table_path)

        sparse_matrix = otu_table.matrix_data.tocsc()
        logger.info(f'Loaded BIOM table with shape {sparse_matrix.shape} (samples x OTUs)')

        all_otu_ids = np.array(otu_table.ids(axis='observation'))
        unmap_mask = all_otu_ids == 'Unmapped'
        if np.any(unmap_mask):
            logger.info("Found 'Unmapped' OTU, removing it")
            keep_mask = ~unmap_mask
            sparse_matrix = sparse_matrix[keep_mask, :].tocsc()
            all_otu_ids = all_otu_ids[keep_mask]

        otu_ids = all_otu_ids.tolist()
        logger.info(f'Number of OTUs: {len(otu_ids)}')
        self._prepare_feature_index(otu_ids, output_path)

        logger.info('Calculating relative abundance...')
        col_sums = sparse_matrix.sum(axis=0).A1
        col_sums[col_sums == 0] = 1.0
        rel_abun = sparse_matrix.dot(sp.diags(1.0 / col_sums)).tocsc()

        sample_ids = np.array(otu_table.ids(axis='sample')).tolist()
        logger.info(f'Number of samples: {len(sample_ids)}')

        otu_to_ntv3_idx = np.full(len(otu_ids), -1, dtype=np.int64)
        for i, otu_id in enumerate(otu_ids):
            if otu_id in self.otu_id_to_idx:
                otu_to_ntv3_idx[i] = self.otu_id_to_idx[otu_id]
        valid_otu_mask = otu_to_ntv3_idx != -1

        indptr = rel_abun.indptr
        indices = rel_abun.indices
        data = rel_abun.data

        all_samples: List[str] = []
        otu_data: Dict[str, Dict[str, np.ndarray]] = {}
        for sample_idx, sample_id in enumerate(tqdm(sample_ids, total=len(sample_ids))):
            start_idx = indptr[sample_idx]
            end_idx = indptr[sample_idx + 1]
            if start_idx == end_idx:
                continue

            col_indices = indices[start_idx:end_idx]
            abundances = data[start_idx:end_idx]
            valid_mask = valid_otu_mask[col_indices] & (abundances > 0)
            valid_cols = col_indices[valid_mask]
            valid_abundances = abundances[valid_mask].astype(np.float32)
            if len(valid_cols) == 0:
                continue

            ntv3_indices = otu_to_ntv3_idx[valid_cols]
            ntv3_indices, valid_abundances = self._keep_top_k_by_relative_abundance(
                ntv3_indices,
                valid_abundances,
            )
            otu_data[sample_id] = {
                'input_ids': np.concatenate([[-1], ntv3_indices]).astype(np.int64),
                'attention_mask': np.ones(len(ntv3_indices) + 1, dtype=np.int64),
                'abundance_values': np.concatenate([[0.0], valid_abundances]).astype(np.float32),
            }
            all_samples.append(sample_id)

        self._validate_non_empty_samples(all_samples, output_path)
        dataset = self._build_dataset_payload(all_samples, otu_data)
        self._save_dataset(dataset, output_path, all_samples, otu_data)


def main():
    parser = argparse.ArgumentParser(
        description='Build the MGM2 preprocessed dataset from an OTU table',
        epilog=(
            'Example:\n'
            '  python src/data/build_dataset.py --otu_table data/otu_table.csv --format csv '
            '--ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5 --output data/pretrain_dataset.pkl\n'
            '  python src/data/build_dataset.py --otu_table data/otu_table.biom --format biom '
            '--ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5 --output data/pretrain_dataset.pkl'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--otu_table', type=str, required=True, help='Path to community table (CSV, TSV, HDF5, or BIOM format)')
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'tsv', 'h5', 'biom'],
        default='csv',
        help='Input format. HDF5 expects <key>/axis0, <key>/axis1, and <key>/block0_values.',
    )
    parser.add_argument('--h5_key', type=str, default='genus', help='HDF5 group key when --format h5')
    parser.add_argument(
        '--h5_load_all',
        action='store_true',
        help='Load the whole HDF5 block0_values matrix into memory before iterating samples; faster but memory intensive.',
    )
    parser.add_argument(
        '--ntv3_embedding',
        type=str,
        default='ntv3_embeddings/otus_99_embeddings.h5',
        help='Path to NTv3 embedding file',
    )
    parser.add_argument('--output', type=str, required=True, help='Output path for the preprocessed dataset')
    parser.add_argument('--embedding_dim', type=int, default=1536, help='Dimension of NTv3 embeddings')
    parser.add_argument(
        '--fallback_embedding',
        type=str,
        default=None,
        help='Optional taxonomy fallback embedding HDF5 built by scripts/build_taxonomy_ntv3_fallback.py',
    )
    parser.add_argument(
        '--resolved_embedding',
        type=str,
        default=None,
        help='Optional output path for compact resolved embeddings when fallback is enabled',
    )
    parser.add_argument(
        '--top_k_otus',
        type=int,
        default=0,
        help='Keep only the top-K OTUs by relative abundance per sample; 0 keeps all non-zero OTUs',
    )
    args = parser.parse_args()

    if args.otu_table.endswith(('.h5', '.hdf5')) and args.format == 'csv':
        input_format = 'h5'
        logger.info('File extension indicates HDF5 format')
    elif args.otu_table.endswith(('.tsv', '.txt')) and args.format == 'csv':
        input_format = 'tsv'
        logger.info('File extension indicates TSV format')
    elif args.format == 'csv' and args.otu_table.endswith('.biom'):
        input_format = 'biom'
        logger.info('File extension indicates BIOM format')
    else:
        input_format = args.format

    builder = MGM2DatasetBuilder(
        args.ntv3_embedding,
        args.embedding_dim,
        top_k_otus=args.top_k_otus,
        fallback_embedding_path=args.fallback_embedding,
        resolved_embedding_path=args.resolved_embedding,
    )
    builder.load_ntv3_embeddings()
    builder.process_otu_table(
        args.otu_table,
        args.output,
        input_format=input_format,
        h5_key=args.h5_key,
        h5_load_all=args.h5_load_all,
    )


if __name__ == '__main__':
    main()
