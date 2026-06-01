OUTPUT=data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl \
    TOP_K_ASVS=200 \
    bash scripts/downstreams/wastewater/build_wastewater_windows.sh --add_other_bucket

DATASET=data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl \
    SCALE=scale_small \
    DEVICE=cuda:5 \
    BATCH_SIZE=8 \
    OUTPUT=data/downstreams/wastewater/cache/mgm2_scale_small_top200_cls.pkl \
    bash scripts/downstreams/wastewater/extract_mgm2_cls_cache.sh

DATASET=data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl \
    CACHE=data/downstreams/wastewater/cache/mgm2_scale_small_top200_cls.pkl \
    SCALE=scale_small \
    DEVICE=cuda:5 \
    OUTPUT_DIR=outputs_downstreams/wastewater/mgm2_temporal/scale_small_top200 \
    bash scripts/downstreams/wastewater/train_mgm2_temporal_forecaster.sh --save_predictions

DATASET=data/downstreams/wastewater/wastewater_windows_top200_ctx10_h10.pkl \
    OUTPUT_DIR=outputs_downstreams/wastewater/baseline/baseline_top200 \
    bash scripts/downstreams/wastewater/run_simple_baselines.sh --save_predictions