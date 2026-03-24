import torch

from src.configs.Config import MGM2Config
from src.training.TrainerModule import MGM2LightningModule
from src.data.Dataset import create_dataloaders_mgm2, MicrobiomeDatasetMGM2
from src.data.Preprocessing import AbundancePreprocessor, BinManager
from pickle import load

dataset = load(open('data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.pkl', 'rb'))
dataset = MicrobiomeDatasetMGM2(dataset, 
                                AbundancePreprocessor(), 
                                BinManager(),
                                qwen3_embedding_path='qwen3_embeddings/microbeatlas_keywords_emb.h5',
                                qwen3_projection_dim=4096,
                                enable_qwen3_alignment=True)