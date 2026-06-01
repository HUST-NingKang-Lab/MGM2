import torch

from src.models.WastewaterWorldModel import MGM2DeltaMAEWorldModel, MGM2JEPAWorldModel, MGM2MAEWorldModel
from scripts.downstreams.wastewater.train_mgm2_world_model import _jepa_latent_diagnostics


def test_mgm2_mae_world_model_outputs_normalized_abundance():
    model = MGM2MAEWorldModel(hidden_size=12, horizon=3, num_asvs=6, num_heads=3, dropout=0.0)
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    context_abundance = context_abundance / context_abundance.sum(dim=-1, keepdim=True)

    outputs = model(context_states, context_abundance)

    assert outputs["abundance"].shape == (2, 3, 6)
    assert torch.allclose(outputs["abundance"].sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.all(outputs["abundance"] >= 0)


def test_mgm2_mae_world_model_defaults_to_main_logratio_attention_config():
    model = MGM2MAEWorldModel(hidden_size=12, horizon=3, num_asvs=6, num_heads=3, dropout=0.0)

    assert model.predictor.query_mode == "temporal_attention"
    assert model.abundance_head.abundance_head == "logratio_residual"
    assert model.abundance_head.residual_scale == 1.0


def test_mgm2_mae_world_model_supports_logratio_abundance_head():
    model = MGM2MAEWorldModel(
        hidden_size=12,
        horizon=3,
        num_asvs=6,
        num_heads=3,
        dropout=0.0,
        abundance_head="logratio_residual",
        residual_scale=1.0,
    )
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    context_abundance = context_abundance / context_abundance.sum(dim=-1, keepdim=True)

    outputs = model(context_states, context_abundance)

    assert outputs["abundance"].shape == (2, 3, 6)
    assert torch.allclose(outputs["abundance"].sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.all(outputs["abundance"] >= 0)


def test_mgm2_mae_world_model_supports_temporal_attention_queries():
    model = MGM2MAEWorldModel(
        hidden_size=12,
        horizon=3,
        num_asvs=6,
        num_heads=3,
        dropout=0.0,
        query_mode="temporal_attention",
    )
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    context_abundance = context_abundance / context_abundance.sum(dim=-1, keepdim=True)

    outputs = model(context_states, context_abundance)

    assert outputs["abundance"].shape == (2, 3, 6)
    assert torch.allclose(outputs["abundance"].sum(dim=-1), torch.ones(2, 3), atol=1e-6)


def test_mgm2_mae_world_model_supports_paper_relu_raw_bray_loss():
    model = MGM2MAEWorldModel(
        hidden_size=12,
        horizon=3,
        num_asvs=6,
        num_heads=3,
        dropout=0.0,
        abundance_head="paper_relu",
        query_mode="temporal_attention",
    )
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    target_abundance = torch.rand(2, 3, 6)

    outputs = model(context_states, context_abundance)
    loss_dict = model.training_loss(
        context_states,
        context_abundance,
        target_abundance,
        loss_mse_weight=0.0,
        bc_loss_mode="raw",
    )

    assert outputs["abundance"].shape == (2, 3, 6)
    assert torch.all(outputs["abundance"] >= 0)
    assert not torch.allclose(outputs["abundance"].sum(dim=-1), torch.ones(2, 3), atol=1e-4)
    assert set(loss_dict) == {"loss", "bray_loss", "mse_loss"}
    assert loss_dict["loss"].ndim == 0


def test_mgm2_delta_mae_world_model_supervises_abundance_change():
    model = MGM2DeltaMAEWorldModel(hidden_size=12, horizon=3, num_asvs=6, num_heads=3, dropout=0.0)
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    context_abundance = context_abundance / context_abundance.sum(dim=-1, keepdim=True)
    target_abundance = torch.rand(2, 3, 6)
    target_abundance = target_abundance / target_abundance.sum(dim=-1, keepdim=True)

    outputs = model(context_states, context_abundance)
    loss_dict = model.training_loss(context_states, context_abundance, target_abundance, loss_mse_weight=0.1, delta_loss_weight=0.5)
    expected_delta = outputs["abundance"] - context_abundance[:, -1:, :].repeat(1, 3, 1)

    assert outputs["abundance"].shape == (2, 3, 6)
    assert outputs["delta"].shape == (2, 3, 6)
    assert torch.allclose(outputs["abundance"].sum(dim=-1), torch.ones(2, 3), atol=1e-6)
    assert torch.allclose(outputs["delta"], expected_delta, atol=1e-6)
    assert set(loss_dict) == {"loss", "bray_loss", "mse_loss", "delta_loss"}
    assert loss_dict["loss"].ndim == 0


def test_mgm2_jepa_world_model_outputs_latent_and_abundance():
    model = MGM2JEPAWorldModel(hidden_size=12, horizon=3, num_asvs=6, num_heads=3, dropout=0.0)
    context_states = torch.randn(2, 5, 6, 12)
    context_abundance = torch.rand(2, 5, 6)
    context_abundance = context_abundance / context_abundance.sum(dim=-1, keepdim=True)

    outputs = model(context_states, context_abundance)

    assert outputs["predicted_states"].shape == (2, 3, 6, 12)
    assert "abundance" not in outputs

    probe_outputs = model(context_states, context_abundance, return_abundance=True)
    assert probe_outputs["abundance"].shape == (2, 3, 6)
    assert torch.allclose(probe_outputs["abundance"].sum(dim=-1), torch.ones(2, 3), atol=1e-6)


def test_mgm2_jepa_world_model_uses_ema_target_encoder():
    model = MGM2JEPAWorldModel(hidden_size=12, horizon=3, num_asvs=6, num_heads=3, dropout=0.0)
    context_states = torch.randn(2, 5, 6, 12)
    target_states = torch.randn(2, 3, 6, 12)

    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    loss_dict = model.training_loss(context_states, target_states)

    assert set(loss_dict) == {"loss", "latent_loss"}
    assert loss_dict["loss"].ndim == 0
    assert "abundance_head" in dict(model.named_modules())

    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.online_encoder.parameters()).add_(1.0)
    model.update_target_encoder(momentum=0.5)
    after = next(model.target_encoder.parameters()).detach()

    assert not torch.allclose(before, after)


def test_jepa_latent_diagnostics_returns_finite_values():
    class Args:
        mode = "jepa"
        batch_size = 2

    model = MGM2JEPAWorldModel(hidden_size=12, horizon=3, num_asvs=6, num_heads=3, dropout=0.0)
    records = [
        {
            "plant": "plant",
            "context_sample_ids": ["c0", "c1"],
            "target_sample_ids": ["t0", "t1", "t2"],
            "context_abundance": torch.rand(2, 6).numpy(),
            "target_abundance": torch.rand(3, 6).numpy(),
        }
    ]
    cache = {
        "sample_to_idx": {
            ("plant", "c0"): 0,
            ("plant", "c1"): 1,
            ("plant", "t0"): 2,
            ("plant", "t1"): 3,
            ("plant", "t2"): 4,
        },
        "token_states": torch.randn(5, 6, 12).numpy(),
    }

    diagnostics = _jepa_latent_diagnostics(model, records, cache, Args(), torch.device("cpu"))

    assert diagnostics["latent_pred_std"] == diagnostics["latent_pred_std"]
    assert diagnostics["latent_target_std"] == diagnostics["latent_target_std"]
    assert diagnostics["latent_pred_target_cosine"] == diagnostics["latent_pred_target_cosine"]
