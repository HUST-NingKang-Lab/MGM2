python src/training/Train.py \
    --data_file data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.pkl \
    --ntv3_embedding ntv3_embeddings/otus_99_embeddings.h5 \
    --qwen3_embedding_path qwen3_embeddings/microbeatlas_keywords_emb.h5 \
    --enable_qwen3_alignment \
    --max_epochs 50 \
    --batch_size 32 \
    --devices 6 7