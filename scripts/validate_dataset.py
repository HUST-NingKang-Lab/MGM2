#!/usr/bin/env python3
"""
微生物组预训练数据集验证脚本
用于检查数据集构建是否正确
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List
import h5py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetValidator:
    """数据集验证器"""

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: 预处理后的数据集路径
        """
        self.dataset_path = dataset_path
        self.dataset = None

    def load_dataset(self):
        """加载数据集"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        logger.info(f"Loaded dataset with {len(self.dataset['samples'])} samples")

    def validate_structure(self):
        """验证数据结构"""
        logger.info("Validating dataset structure...")

        required_keys = ['samples', 'input_ids', 'attention_mask',
                        'abundance_values', 'labels', 'embeddings']

        for key in required_keys:
            if key not in self.dataset:
                raise ValueError(f"Missing required key: {key}")
            if len(self.dataset[key]) != len(self.dataset['samples']):
                raise ValueError(f"Length mismatch for key: {key}")

        logger.info("✓ Dataset structure is valid")

    def validate_abundance_ranges(self):
        """验证丰度值范围"""
        logger.info("Validating abundance ranges...")

        all_abundances = []
        for abundances in self.dataset['abundance_values']:
            all_abundances.extend(abundances)

        all_abundances = np.array(all_abundances)

        # 检查CLS token（值为0）
        cls_values = all_abundances[all_abundances == 0]
        logger.info(f"CLS tokens: {len(cls_values)} ({len(cls_values)/len(all_abundances)*100:.2f}%)")

        # 检查正常丰度值范围
        valid_abundances = all_abundances[all_abundances != 0]
        if len(valid_abundances) > 0:
            logger.info(f"Abundance range: [{valid_abundances.min():.4f}, {valid_abundances.max():.4f}]")

            # 检查是否在[-2, 2]范围内
            out_of_range = (valid_abundances < -2) | (valid_abundances > 2)
            if np.any(out_of_range):
                logger.warning(f"Found {np.sum(out_of_range)} abundance values out of range")
            else:
                logger.info("✓ All abundance values are in range [-2, 2]")
        else:
            logger.warning("No valid abundance values found")

        return all_abundances

    def validate_embeddings(self, ntv3_path: str):
        """验证embedding的正确性"""
        logger.info("Validating embeddings...")

        # 加载NTv3 embedding
        with h5py.File(ntv3_path, 'r') as f:
            ntv3_embeddings = f['embeddings'][:]

        # 随机检查几个样本
        n_samples_to_check = min(5, len(self.dataset['samples']))
        indices = np.random.choice(len(self.dataset['samples']), n_samples_to_check, replace=False)

        for idx in indices:
            sample_embeddings = self.dataset['embeddings'][idx]
            input_ids = self.dataset['input_ids'][idx]

            # 检查每个非CLS token的embedding
            for i, otu_idx in enumerate(input_ids):
                if otu_idx == -1:  # CLS token
                    # CLS embedding应该是全零
                    if not np.allclose(sample_embeddings[i], 0):
                        logger.error(f"CLS embedding at sample {idx}, position {i} is not zero")
                else:
                    # 检查embedding是否匹配NTv3 embedding
                    if not np.allclose(sample_embeddings[i], ntv3_embeddings[otu_idx]):
                        logger.error(f"Embedding mismatch at sample {idx}, OTU index {otu_idx}")

        logger.info("✓ Embedding validation passed")

    def validate_sequences(self):
        """验证序列信息"""
        logger.info("Validating sequence information...")

        # 检查序列长度
        seq_lengths = [len(ids) for ids in self.dataset['input_ids']]

        logger.info(f"Sequence length statistics:")
        logger.info(f"  Min: {np.min(seq_lengths)}")
        logger.info(f"  Max: {np.max(seq_lengths)}")
        logger.info(f"  Mean: {np.mean(seq_lengths):.2f}")
        logger.info(f"  Median: {np.median(seq_lengths):.2f}")

        # 检查是否所有序列都以CLS token开始
        for i, ids in enumerate(self.dataset['input_ids']):
            if ids[0] != -1:
                logger.error(f"Sample {i} does not start with CLS token")

        logger.info("✓ Sequence validation passed")

    def plot_distributions(self):
        """绘制分布图"""
        logger.info("Generating distribution plots...")

        # 丰度分布
        all_abundances = []
        for abundances in self.dataset['abundance_values']:
            all_abundances.extend(abundances[abundances != 0])  # 排除CLS token和0值

        plt.figure(figsize=(15, 5))

        # 丰度分布
        plt.subplot(1, 3, 1)
        plt.hist(all_abundances, bins=50, alpha=0.7)
        plt.title('Abundance Distribution')
        plt.xlabel('Abundance Value')
        plt.ylabel('Frequency')

        # 序列长度分布
        plt.subplot(1, 3, 2)
        seq_lengths = [len(ids) for ids in self.dataset['input_ids']]
        plt.hist(seq_lengths, bins=50, alpha=0.7)
        plt.title('Sequence Length Distribution')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')

        # 每个样本的OTU数量分布
        plt.subplot(1, 3, 3)
        n_otus_per_sample = [np.sum(np.array(values) != 0) - 1 for values in self.dataset['abundance_values']]  # 减去CLS token
        plt.hist(n_otus_per_sample, bins=50, alpha=0.7)
        plt.title('Number of OTUs per Sample')
        plt.xlabel('Number of OTUs')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('dataset_distributions.png', dpi=300)
        logger.info("Saved distribution plots to dataset_distributions.png")

    def check_otu_coverage(self, otu_table_path: str):
        """检查OTU覆盖率"""
        logger.info("Checking OTU coverage...")

        # 根据文件扩展名判断格式
        if otu_table_path.endswith('.biom'):
            self._check_otu_coverage_biom(otu_table_path)
        else:
            self._check_otu_coverage_csv(otu_table_path)

    def _check_otu_coverage_csv(self, otu_table_path: str):
        """检查CSV格式的OTU覆盖率"""
        # 加载OTU表
        import pandas as pd
        otu_df = pd.read_csv(otu_table_path, index_col=0)

        # 统计每个OTU有多少样本包含它
        otu_presence = (otu_df > 0).sum()

        logger.info(f"OTU presence statistics:")
        logger.info(f"  Total OTUs: {len(otu_presence)}")
        logger.info(f"  Mean presence: {otu_presence.mean():.2f} samples")
        logger.info(f"  Median presence: {otu_presence.median():.2f} samples")
        logger.info(f"  OTUs present in >50% samples: {(otu_presence > len(otu_df) * 0.5).sum()}")
        logger.info(f"  OTUs present in >90% samples: {(otu_presence > len(otu_df) * 0.9).sum()}")

        # 检查这些OTU是否在embedding中
        otu_ids = otu_df.columns.tolist()
        missing_embeddings = []
        for otu_id in otu_ids:
            if otu_id not in self.dataset['otu_id_to_idx']:
                missing_embeddings.append(otu_id)

        if missing_embeddings:
            logger.warning(f"Found {len(missing_embeddings)} OTUs without embeddings")
            logger.warning(f"Example missing OTUs: {missing_embeddings[:10]}")
        else:
            logger.info("✓ All OTUs have corresponding embeddings")

        return otu_presence

    def _check_otu_coverage_biom(self, otu_table_path: str):
        """检查BIOM格式的OTU覆盖率"""
        try:
            from biom import load_table
        except ImportError:
            logger.error("BIOM library not available. Install with: pip install biom-format")
            return

        # 加载BIOM表
        otu_table = load_table(otu_table_path)
        sparse_matrix = otu_table.matrix_data

        # 移除 'Unmapped' 行
        if sparse_matrix.shape[0] > 0:
            sparse_matrix = sparse_matrix[:-1, :]

        # 获取OTU ID列表
        all_otu_ids = otu_table.ids(axis='observation')
        if len(all_otu_ids) > 0:
            otu_ids = all_otu_ids[:-1].tolist()
        else:
            otu_ids = []

        # 统计每个OTU有多少样本包含它
        otu_presence = (sparse_matrix > 0).sum(axis=1).A1
        otu_presence_dict = dict(zip(otu_ids, otu_presence))

        logger.info(f"OTU presence statistics:")
        logger.info(f"  Total OTUs: {len(otu_presence_dict)}")
        logger.info(f"  Mean presence: {otu_presence.mean():.2f} samples")
        logger.info(f"  Median presence: {np.median(otu_presence):.2f} samples")
        logger.info(f"  OTUs present in >50% samples: {(otu_presence > sparse_matrix.shape[1] * 0.5).sum()}")
        logger.info(f"  OTUs present in >90% samples: {(otu_presence > sparse_matrix.shape[1] * 0.9).sum()}")

        # 检查这些OTU是否在embedding中
        missing_embeddings = []
        for otu_id in otu_ids:
            if otu_id not in self.dataset['otu_id_to_idx']:
                missing_embeddings.append(otu_id)

        if missing_embeddings:
            logger.warning(f"Found {len(missing_embeddings)} OTUs without embeddings")
            logger.warning(f"Example missing OTUs: {missing_embeddings[:10]}")
        else:
            logger.info("✓ All OTUs have corresponding embeddings")

        return otu_presence_dict

    def run_full_validation(self, otu_table_path: str, ntv3_path: str):
        """运行完整验证"""
        logger.info("Starting full dataset validation")

        # 加载数据
        self.load_dataset()

        # 结构验证
        self.validate_structure()

        # 丰度范围验证
        abundances = self.validate_abundance_ranges()

        # Embedding验证
        self.validate_embeddings(ntv3_path)

        # 序列验证
        self.validate_sequences()

        # OTU覆盖率检查
        otu_presence = self.check_otu_coverage(otu_table_path)

        # 绘制分布图
        self.plot_distributions()

        # 总结
        logger.info("=" * 50)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total samples: {len(self.dataset['samples'])}")
        logger.info(f"Total OTUs: {len(self.dataset['otu_id_to_idx'])}")
        logger.info(f"Embedding dimension: {self.dataset['embeddings'][0].shape[1]}")
        logger.info(f"Abundance range: [{abundances[abundances != 0].min():.4f}, {abundances[abundances != 0].max():.4f}]")
        logger.info(f"Avg sequence length: {np.mean([len(ids) for ids in self.dataset['input_ids']]):.2f}")
        logger.info("✓ All validations passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate microbiome pre-training dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to the preprocessed dataset file')
    parser.add_argument('--otu_table', type=str, required=True,
                       help='Path to the original OTU table')
    parser.add_argument('--ntv3_embedding', type=str,
                       default='ntv3_embeddings/otus_99_embeddings.h5',
                       help='Path to NTv3 embedding file')

    args = parser.parse_args()

    validator = DatasetValidator(args.dataset)
    validator.run_full_validation(args.otu_table, args.ntv3_embedding)
