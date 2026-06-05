import numpy as np
import torch

from scripts.downstreams.wastewater_global_forecast.train_global_abundance_transformer import (
    GlobalAbundanceTransformer,
    _combined_loss,
    _split_train_tail_by_plant,
)


def test_global_abundance_transformer_outputs_closed_abundance():
    model = GlobalAbundanceTransformer(num_asvs=7, horizon=3, context_len=5, hidden_size=16, num_heads=4, dropout=0.0)
    context = torch.rand(4, 5, 7)
    output = model(context)

    assert output.shape == (4, 3, 7)
    assert torch.all(output >= 0)
    assert torch.allclose(output.sum(dim=-1), torch.ones(4, 3), atol=1e-6)


def test_combined_loss_adds_mse_auxiliary_term():
    pred = torch.tensor([[[0.6, 0.4]]], dtype=torch.float32)
    target = torch.tensor([[[0.4, 0.6]]], dtype=torch.float32)

    pure = _combined_loss(pred, target, mse_weight=0.0)
    combined = _combined_loss(pred, target, mse_weight=0.5)

    assert combined > pure


def test_split_train_tail_by_plant_keeps_holdout_for_each_plant():
    records = []
    for plant in ["A", "B"]:
        for idx in range(6):
            records.append({"plant": plant, "split": "train", "idx": idx})
        records.append({"plant": plant, "split": "test", "idx": 99})

    train, holdout = _split_train_tail_by_plant(records, val_ratio=0.25)

    assert len(train) == 8
    assert len(holdout) == 4
    assert {record["plant"] for record in train} == {"A", "B"}
    assert {record["plant"] for record in holdout} == {"A", "B"}
    assert [record["idx"] for record in holdout if record["plant"] == "A"] == [4, 5]
    assert [record["idx"] for record in holdout if record["plant"] == "B"] == [4, 5]
