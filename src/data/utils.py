"""
MGM2.0 Data Utilities

Helper functions for data loading and processing
"""

import os
import pickle
from typing import Dict, Any

import h5py
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_ntv3_embeddings(
    embedding_path: str,
    use_memory_mapping: bool = True
) -> np.ndarray:
    """
    Load NTv3 embeddings from HDF5 file.

    The previous implementation returned an h5py dataset handle from inside a
    context manager, which left callers with an invalid dataset identifier.
    MGM2 keeps the simpler and safer behavior: always materialize the array.
    """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    with h5py.File(embedding_path, 'r') as f:
        embeddings = f['embeddings'][:]
        otu_ids = f['otu_ids'][:].astype(str)

    logger.info(f"Loaded embeddings: {embeddings.shape[0]} OTUs, {embeddings.shape[1]} dimensions")
    logger.info(f"OTU IDs range: {otu_ids[:5]}...{otu_ids[-5:]}")
    if use_memory_mapping and embeddings.shape[0] > 50000:
        logger.info("Large embedding file detected; loading to memory to avoid invalid closed HDF5 handles")
    else:
        logger.info("Loading embeddings to memory")

    return embeddings


def load_preprocessed_data(
    data_path: str,
    load_in_memory: bool = True
) -> Dict[str, Any]:
    """
    Load preprocessed data from pickle file.

    Args:
        data_path: Path to preprocessed data file
        load_in_memory: Whether to load all data to memory

    Returns:
        Dictionary containing preprocessed data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading preprocessed data from {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    otu_count = 0
    if 'otu_ids' in data:
        otu_count = len(data['otu_ids'])
    elif 'otu_id_to_idx' in data:
        otu_count = len(data['otu_id_to_idx'])

    logger.info(f"Loaded data with {len(data.get('samples', []))} samples and {otu_count} OTUs")
    logger.info(f"Data keys: {list(data.keys())}")

    return data
