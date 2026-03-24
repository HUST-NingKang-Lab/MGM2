import argparse
import os
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from Bio import SeqIO
import h5py
import torch
import numpy as np
from tqdm import tqdm
import math
import multiprocessing as mp

'''
python ntv3_embedding.py \
  --fasta_path data/microbeatals/otus.99.fasta \
  --model_path /data7/zhanghaohong/transformers_model/InstaDeepAI/NTv3_650M_pre \
  --output_path ntv3_embeddings/otus_99_embeddings.h5 \
  --gpu_ids 0,1,2,3,4,7
'''

def embed_seq(seq, model, tokenizer, device):
    # pad seq to nearest multiple of 128 with "N"
    seq_len = len(seq)
    pad_len = (128 - (seq_len % 128)) % 128
    padded_seq = seq + "N" * pad_len
    # tokenize and convert to input ids
    inputs = tokenizer(padded_seq, return_tensors="pt")
    input_ids = inputs["input_ids"]
    input_ids = input_ids.to(device)
    # get hidden states from model
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states[-1].squeeze(0).cpu()  # (seq_len + pad_len, hidden_size)
    # mean pool the hidden states to get a single embedding vector
    embedding = hidden_states[:seq_len].mean(dim=0)  # (hidden_size,)
    return embedding

def worker(gpu_id, seq_chunk, args):
    device = f"cuda:{gpu_id}"
    
    # Load model and tokenizer for this specific GPU
    config = AutoConfig.from_pretrained(args.model_path, 
                                        trust_remote_code=True, 
                                        local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, 
                                              trust_remote_code=True, 
                                              local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(
                    args.model_path, 
                    trust_remote_code=True,
                    config=config,
                    local_files_only=True
                )
    
    model.to(device)
    model.eval()

    embeddings_list = []
    otu_ids_list = []

    for seq_id, seq in tqdm(seq_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
        embedding = embed_seq(seq, model, tokenizer, device)
        embeddings_list.append(embedding.numpy())
        
        otu_ids_list.append(seq_id.encode('utf-8'))
            
    temp_h5_path = f"{args.output_path}_tmp_{gpu_id}.h5"
    with h5py.File(temp_h5_path, "w") as f:
        f.create_dataset("embeddings", data=np.stack(embeddings_list))
        f.create_dataset("otu_ids", data=np.array(otu_ids_list))
            
    return temp_h5_path

def main():
    parser = argparse.ArgumentParser(description="Extract NTv3 embeddings from a FASTA file.")
    parser.add_argument("-i", "--fasta_path", required=True, type=str, help="Path to the input fasta file")
    parser.add_argument("-m", "--model_path", required=True, type=str, help="Path to the NT model directory")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="Path for the output h5 file")
    parser.add_argument("-g", "--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g. '0,1,2')")
    
    args = parser.parse_args()

    # Read sequences from fasta file
    print(f"Reading sequences from {args.fasta_path}...")
    seq_list = []
    for record in SeqIO.parse(args.fasta_path, "fasta"):
        seq_list.append((record.id, str(record.seq)))

    # Make output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    gpu_list = [int(gid) for gid in args.gpu_ids.split(",")]
    print(f"Using GPUs: {gpu_list}")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    num_gpus = len(gpu_list)
    
    # Split sequences into chunks for each GPU
    chunk_size = math.ceil(len(seq_list) / num_gpus)
    chunks = [seq_list[i:i + chunk_size] for i in range(0, len(seq_list), chunk_size)]

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
    
    all_embeddings = []
    all_otu_ids = []

    for temp_h5 in temp_files:
        with h5py.File(temp_h5, "r") as in_f:
            all_embeddings.append(in_f["embeddings"][:])
            all_otu_ids.append(in_f["otu_ids"][:])
        os.remove(temp_h5)
            
    with h5py.File(args.output_path, "w") as out_f:
        out_f.create_dataset("embeddings", data=np.concatenate(all_embeddings, axis=0))
        out_f.create_dataset("otu_ids", data=np.concatenate(all_otu_ids, axis=0))


if __name__ == "__main__":
    main()