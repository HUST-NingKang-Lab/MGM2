import torch
import pytest

from src.models.WastewaterModel import (
    MGM2TokenCrossAttentionForecaster,
    PlantTemporalForecaster,
    VectorTemporalForecaster,
    bray_curtis_raw_loss_torch,
    bray_curtis_loss_torch,
)
from scripts.downstreams.wastewater.extract_mgm2_cls_cache import _load_taxonomy


def test_plant_temporal_forecaster_outputs_normalized_abundance():
    model = PlantTemporalForecaster(
        input_size=8,
        hidden_size=16,
        horizon=3,
        num_asvs=5,
        num_layers=1,
        dropout=0.0,
    )
    context = torch.randn(4, 6, 8)
    output = model(context)

    assert output.shape == (4, 3, 5)
    assert torch.allclose(output.sum(dim=-1), torch.ones(4, 3), atol=1e-6)
    assert torch.all(output >= 0)


def test_bray_curtis_loss_torch_is_zero_for_identical_inputs():
    y_true = torch.tensor([[[0.7, 0.2, 0.1]]], dtype=torch.float32)
    y_pred = torch.tensor([[[0.7, 0.2, 0.1]]], dtype=torch.float32)
    assert bray_curtis_loss_torch(y_pred, y_true).item() < 1e-6


def test_bray_curtis_loss_torch_penalizes_different_inputs():
    y_true = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    y_pred = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
    assert bray_curtis_loss_torch(y_pred, y_true).item() > 0.99


def test_raw_bray_curtis_loss_does_not_normalize_predictions():
    y_true = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
    y_pred_same_direction = torch.tensor([[[5.0, 5.0]]], dtype=torch.float32)

    assert bray_curtis_loss_torch(y_pred_same_direction, y_true).item() < 1e-6
    assert bray_curtis_raw_loss_torch(y_pred_same_direction, y_true).item() > 0.8


def test_extract_cache_requires_taxonomy_by_default(tmp_path):
    with pytest.raises(FileNotFoundError, match="no taxonomy files found"):
        _load_taxonomy(tmp_path, "MissingPlant")


@pytest.mark.parametrize("model_type", ["gru", "tcn"])
def test_vector_temporal_forecaster_outputs_normalized_abundance(model_type):
    model = VectorTemporalForecaster(
        num_asvs=7,
        horizon=4,
        hidden_size=16,
        model_type=model_type,
        num_layers=2,
        dropout=0.0,
    )
    context = torch.rand(3, 5, 7)
    context = context / context.sum(dim=-1, keepdim=True)
    output = model(context)

    assert output.shape == (3, 4, 7)
    assert torch.allclose(output.sum(dim=-1), torch.ones(3, 4), atol=1e-6)
    assert torch.all(output >= 0)


def test_mgm2_token_cross_attention_forecaster_outputs_normalized_abundance():
    model = MGM2TokenCrossAttentionForecaster(
        hidden_size=12,
        horizon=3,
        num_asvs=6,
        num_heads=3,
        dropout=0.0,
    )
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    context_abundance = context_abundance / context_abundance.sum(dim=-1, keepdim=True)
    output = model(context_states, context_abundance)

    assert output.shape == (2, 3, 6)
    assert torch.allclose(output.sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.all(output >= 0)


def test_mgm2_token_cross_attention_forecaster_supports_temporal_attention_and_logratio_head():
    model = MGM2TokenCrossAttentionForecaster(
        hidden_size=12,
        horizon=3,
        num_asvs=6,
        num_heads=3,
        dropout=0.0,
        query_mode="temporal_attention",
        abundance_head="logratio_residual",
        residual_scale=1.0,
    )
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    context_abundance = context_abundance / context_abundance.sum(dim=-1, keepdim=True)

    output = model(context_states, context_abundance)

    assert output.shape == (2, 3, 6)
    assert torch.allclose(output.sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.all(output >= 0)


def test_mgm2_token_cross_attention_forecaster_supports_paper_relu_head():
    model = MGM2TokenCrossAttentionForecaster(
        hidden_size=12,
        horizon=3,
        num_asvs=6,
        num_heads=3,
        dropout=0.0,
        query_mode="temporal_attention",
        abundance_head="paper_relu",
    )
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    output = model(context_states, context_abundance)

    assert output.shape == (2, 3, 6)
    assert torch.all(output >= 0)
    assert not torch.allclose(output.sum(dim=-1), torch.ones(2, 3), atol=1e-4)
