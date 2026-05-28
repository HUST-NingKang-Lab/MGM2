python scripts/evaluate_embeddings.py \
    --checkpoint outputs_grid/stage1/20260329_085632/run_001_lr1e-4_wd0.01_mask0.15_reg0.3_qw0.1/checkpoints/mgm2-epoch=18-val/total_loss=0.1695.ckpt \
    --data_file data/microbeatlas/otus.99.mapped.metag.minfilter.refilt.dict.mgm2v1.pkl \
    --ntv3_embedding_path ntv3_embeddings/otus_99_embeddings.h5 \
    --metadata_file /data9/zhanghaohong/databases/microbeatals/samples.env.info.tsv \
    --sample_id_column MAP_SID \
    --label_column Environments \
    --embedding_strategy cls \
    --output_dir outputs_eval/stage1_run001_cls \
    --device cpu \
    --config outputs_grid/stage1/20260329_085632/run_001_lr1e-4_wd0.01_mask0.15_reg0.3_qw0.1/config.json

# UMAP visualization for Qwen3 keyword embeddings
python scripts/qwen3_umap_viz.py \
    --emb_path qwen3_embeddings/microbeatlas_keywords_emb.h5 \
    --metadata_path /data9/zhanghaohong/databases/microbeatals/samples.env.info.tsv \
    --n_samples 50000 \
    --n_clusters 20 \
    --output_dir figures/umap_qwen3 \
    --seed 0

# NTV3 embedding validation

python scripts/validate_ntv3_embedding.py \
    --emb_path ntv3_embeddings/otus_99_embeddings.h5 \
    --fasta_path data/microbeatlas/otus.99.fasta \
    --tax_path /data9/zhanghaohong/databases/microbeatals/otus.info.tsv \
    --n_sample_seq 10000 \
    --n_pairs 50000 \
    --n_sample_nn 10000 \
    --k_max 10 \
    --seed 0 \
    --k_mer 6