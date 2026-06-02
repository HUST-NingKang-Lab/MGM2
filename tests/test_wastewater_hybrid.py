import pandas as pd

from scripts.downstreams.wastewater.evaluate_horizon_hybrid import build_hybrid_predictions, select_cutoff_by_val


def _prediction_frame(model, split, h1_pred, h2_pred):
    rows = []
    for horizon, pred in [(1, h1_pred), (2, h2_pred)]:
        rows.append(
            {
                "model": model,
                "plant": "PlantA",
                "split": split,
                "window_idx": 0,
                "horizon": horizon,
                "sample_id": f"S{horizon}",
                "date": f"2020-01-0{horizon}",
                "asv_id": "ASV1",
                "y_true": 1.0,
                "y_pred": pred,
            }
        )
    return pd.DataFrame(rows)


def test_build_hybrid_predictions_uses_persistence_until_cutoff():
    persistence = _prediction_frame("persistence", "test", h1_pred=0.9, h2_pred=0.8)
    learned = _prediction_frame("vector_gru", "test", h1_pred=0.1, h2_pred=0.7)

    hybrid = build_hybrid_predictions(
        persistence,
        learned,
        learned_model="vector_gru",
        cutoff=1,
        hybrid_model_name="hybrid",
    )

    assert hybrid["model"].unique().tolist() == ["hybrid"]
    assert hybrid.sort_values("horizon")["y_pred"].tolist() == [0.9, 0.7]


def test_select_cutoff_by_val_chooses_lowest_bray_curtis():
    val_metrics = pd.DataFrame(
        [
            {"cutoff": 0, "split": "val", "bray_curtis": 0.3, "top10_overlap": 0.5},
            {"cutoff": 1, "split": "val", "bray_curtis": 0.2, "top10_overlap": 0.4},
            {"cutoff": 2, "split": "val", "bray_curtis": 0.2, "top10_overlap": 0.8},
        ]
    )

    assert select_cutoff_by_val(val_metrics) == 2
