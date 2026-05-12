python scripts/downstreams/drug_perturbation/train_mlp_baseline.py \
    --split_modes pair \
    --top_k_otus 1024 \
    --max_folds 20 \
    --max_epochs 50 \
    --patience 8 \
    --output_dir outputs_downstreams/drug_perturbation/mlp_pair_top1024_test \
    --device cuda:7

python scripts/downstreams/drug_perturbation/train_mlp_baseline.py \
    --split_modes pair \
    --top_k_otus 1024 \
    --max_folds 0 \
    --output_dir outputs_downstreams/drug_perturbation/mlp_pair_top1024_full