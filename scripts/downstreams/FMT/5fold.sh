DEVICES='7' SEEDS='42' sh scripts/downstreams/FMT/train_fmt_community_5folds_all_scales.sh --ntv3_embedding_path data/downstreams/FMT_5folds/community_all.resolved_embeddings.h5
DEVICES='7' SEEDS='42' sh scripts/downstreams/FMT/train_fmt_classification_5folds_all_scales.sh --ntv3_embedding_path data/downstreams/FMT_5folds/community_all.resolved_embeddings.h5
SEEDS='42' sh scripts/downstreams/FMT/train_mlp_community_5folds.sh
SEEDS='42' sh scripts/downstreams/FMT/train_mlp_classification_5folds.sh