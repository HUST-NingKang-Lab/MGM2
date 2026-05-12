"""
MGM2 data module.

Contains dataset classes, preprocessing utilities, and data loading logic
for the current dual-task masked abundance model.
"""

from .Preprocessing import AbundancePreprocessor, BinManager
from .Dataset import MicrobiomeDatasetMGM2, create_dataloaders_mgm2, BatchCollatorMGM2
from .CommunityClassificationDataset import (
    CommunityClassificationDataset,
    create_community_classification_dataloaders,
    create_community_classification_test_loader,
    stratified_split_sample_ids,
)
from .DrugPerturbationDataset import DrugPerturbationCollator, DrugPerturbationDataset
from .utils import load_ntv3_embeddings, load_preprocessed_data
from .build_dataset import MGM2DatasetBuilder

__all__ = [
    'AbundancePreprocessor',
    'BinManager',
    'MicrobiomeDatasetMGM2',
    'CommunityClassificationDataset',
    'DrugPerturbationDataset',
    'DrugPerturbationCollator',
    'create_dataloaders_mgm2',
    'create_community_classification_dataloaders',
    'create_community_classification_test_loader',
    'stratified_split_sample_ids',
    'BatchCollatorMGM2',
    'load_ntv3_embeddings',
    'load_preprocessed_data',
    'MGM2DatasetBuilder'
]
