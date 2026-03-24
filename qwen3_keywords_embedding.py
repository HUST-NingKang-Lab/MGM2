#!/usr/bin/env python3
"""
Qwen3 Embedding for MicrobeAtlas Keywords

Multi-GPU parallel encoding of sample keywords from microbeatlass metadata.
Output: h5 file with embeddings where each entry is keyed by sample ID.

Usage:
python qwen3_keywords_embedding.py \
  --metadata_path /data9/zhanghaohong/databases/microbeatals/samples.env.info.tsv \
  --model_path /data7/zhanghaohong/transformers_model/Qwen/Qwen3-Embedding-8B \
  --output_path embeddings/microbeatlas_keywords_emb.h5 \
  --gpu_ids 0,1,2,3 \
  --batch_size 32 \
  --max_length 512
"""

import argparse
import os
import h5py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple


def last_token_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """Last token pooling for Qwen3 embeddings"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format instruction for Qwen3 instruct"""
    return f'Instruct: {task_description}\nQuery:{query}'


def process_keywords(keywords_str: str) -> str:
    """Process keywords string from microbeatlass metadata"""
    # Replace semicolon with commas or spaces
    keywords_str = keywords_str.replace(';', ', ').strip()
    if not keywords_str or pd.isna(keywords_str):
        keywords_str = 'Unknown'
    # Add descriptive prompt
    return f'This microbiome sample is associated with the following environmental keywords: {keywords_str}'


def embed_batch(samples: List[str], model, tokenizer, device, max_length: int):
    """Embed a batch of text samples"""
    # Tokenize
    batch_dict = tokenizer(
        samples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    batch_dict = batch_dict.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def worker(gpu_id: int, sample_chunk: List[Tuple[str, str]], args):
    """Worker process for a single GPU"""
    device = f"cuda:{gpu_id}"

    # Load tokenizer and model
    print(f"GPU {gpu_id}: Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left')
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16
    )
    model.to(device)
    model.eval()

    embeddings_list = []
    sample_ids_list = []

    batch_size = args.batch_size
    num_batches = math.ceil(len(sample_chunk) / batch_size)

    for i in tqdm(range(num_batches), desc=f"GPU {gpu_id}", position=gpu_id):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sample_chunk))
        batch_samples = sample_chunk[start_idx:end_idx]

        # Extract texts
        sample_ids = [sid for sid, text in batch_samples]
        texts = [text for sid, text in batch_samples]

        # Embed batch
        embeddings = embed_batch(texts, model, tokenizer, device, args.max_length)

        embeddings_list.extend(embeddings)
        sample_ids_list.extend(sample_ids)

    # Save temporary file
    temp_h5_path = f"{args.output_path}_tmp_{gpu_id}.h5"
    with h5py.File(temp_h5_path, "w") as f:
        f.create_dataset("embeddings", data=np.stack(embeddings_list))
        # Store sample_ids as variable-length UTF-8 strings
        dt = h5py.special_dtype(vlen=str)
        sample_ids_arr = np.array(sample_ids_list, dtype=object)
        f.create_dataset("sample_ids", data=sample_ids_arr, dtype=dt)

    return temp_h5_path


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU embedding of microbeatlass sample keywords using Qwen3")
    parser.add_argument("-i", "--metadata_path", required=True, type=str,
                        help="Path to microbeatlass metadata tsv file (samples.env.info.tsv)")
    parser.add_argument("-m", "--model_path", required=True, type=str,
                        help="Path to Qwen3 embedding model directory")
    parser.add_argument("-o", "--output_path", required=True, type=str,
                        help="Path for output h5 file")
    parser.add_argument("-g", "--gpu_ids", type=str, default="0",
                        help="Comma-separated list of GPU IDs to use (e.g. '0,1,2,3')")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("-l", "--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--keyword_column", type=str, default="Keywords",
                        help="Column name for keywords in metadata")
    parser.add_argument("--sampleid_column", type=str, default="MAP_SID",
                        help="Column name for sample ID in metadata")

    args = parser.parse_args()

    # Read metadata
    print(f"Reading metadata from {args.metadata_path}...")
    df = pd.read_csv(args.metadata_path, sep='\t')
    print(f"Loaded {len(df)} samples")

    # Process all samples
    samples = []
    for _, row in tqdm(df.iterrows()):
        sample_id = str(row[args.sampleid_column])
        keywords = str(row[args.keyword_column])
        processed_text = process_keywords(keywords)
        samples.append((sample_id, processed_text))

    print(f"Processed {len(samples)} samples")

    # Make output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Split into chunks for each GPU
    gpu_list = [int(gid) for gid in args.gpu_ids.split(",")]
    print(f"Using GPUs: {gpu_list}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    num_gpus = len(gpu_list)

    chunk_size = math.ceil(len(samples) / num_gpus)
    chunks = [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]
    print(f"Split into {len(chunks)} chunks (avg {chunk_size} samples per GPU)")

    # Set multiprocessing start method to 'spawn' for safe CUDA usage
    mp.set_start_method('spawn', force=True)

    pool = mp.Pool(processes=num_gpus)
    results = []

    for i in range(min(num_gpus, len(chunks))):
        results.append(pool.apply_async(worker, args=(i, chunks[i], args)))

    pool.close()
    pool.join()

    temp_files = [res.get() for res in results]

    print(f"Merging temporary files into {args.output_path}...")

    # Merge all results
    all_embeddings = []
    all_sample_ids = []

    for temp_h5 in temp_files:
        with h5py.File(temp_h5, "r") as in_f:
            all_embeddings.append(in_f["embeddings"][:])
            # Read variable-length strings
            sample_ids_chunk = [str(sid) for sid in in_f["sample_ids"][:]]
            all_sample_ids.extend(sample_ids_chunk)
        os.remove(temp_h5)

    # Get embedding dimension
    emb_dim = np.concatenate(all_embeddings, axis=0).shape[1]
    print(f"Final embeddings: {len(all_sample_ids)} samples, dimension={emb_dim}")

    # Save final output
    with h5py.File(args.output_path, "w") as out_f:
        # Store embeddings as (n_samples, embedding_dim)
        out_f.create_dataset("embeddings", data=np.concatenate(all_embeddings, axis=0))
        # Store sample_ids as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        sample_ids_arr = np.array(all_sample_ids, dtype=object)
        out_f.create_dataset("sample_ids", data=sample_ids_arr, dtype=dt)

    print(f"Done! Saved to {args.output_path}")
    print(f"File contains:")
    print(f"  - 'embeddings': shape ({len(all_sample_ids)}, {emb_dim})")
    print(f"  - 'sample_ids': array of sample ID strings")


if __name__ == "__main__":
    main()
