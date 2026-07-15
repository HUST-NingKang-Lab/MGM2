import types
import unittest

import torch

from scripts.downstreams.wastewater_v2.train_token_asv_delta_transformer import (
    TokenASVDeltaTransformer,
    combined_loss,
)


def _args(**overrides):
    base = {
        "pseudocount": 1e-4,
        "loss_mse_weight": 0.1,
        "loss_clr_delta_weight": 0.05,
        "loss_corr_weight": 0.05,
        "loss_direction_weight": 0.02,
        "loss_weighted_bray_weight": 0.0,
        "loss_weighted_mse_weight": 0.0,
        "loss_asv_corr_weight": 0.0,
        "loss_delta_l2_weight": 0.0,
        "loss_weight_abundance_factor": 1.0,
        "loss_weight_delta_factor": 2.0,
        "loss_weight_power": 0.5,
        "loss_weight_max": 8.0,
        "asv_corr_min_std": 1e-5,
        "direction_delta_threshold": 1e-4,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


class WastewaterV2TokenCalibrationTests(unittest.TestCase):
    def test_token_model_supports_bottleneck_projection(self):
        torch.manual_seed(7)
        model = TokenASVDeltaTransformer(
            num_asvs=5,
            context_len=4,
            horizon=3,
            token_dim=64,
            num_plants=2,
            hidden_size=32,
            num_heads=4,
            temporal_layers=1,
            asv_layers=1,
            dropout=0.0,
            token_bottleneck_dim=16,
        )
        context = torch.rand(2, 4, 5)
        context = context / context.sum(dim=-1, keepdim=True)
        tokens = torch.randn(2, 4, 5, 64)
        plant_id = torch.tensor([0, 1])

        outputs = model(context, tokens, plant_id)

        self.assertEqual(outputs["abundance"].shape, (2, 3, 5))
        self.assertEqual(outputs["pred_delta_clr"].shape, (2, 3, 5))
        self.assertTrue(torch.allclose(outputs["abundance"].sum(dim=-1), torch.ones(2, 3), atol=1e-6))
        self.assertEqual(model.token_projection[1].out_features, 16)

    def test_delta_l2_weight_contributes_to_combined_loss(self):
        torch.manual_seed(11)
        context = torch.rand(2, 4, 5)
        context = context / context.sum(dim=-1, keepdim=True)
        target = torch.rand(2, 3, 5)
        target = target / target.sum(dim=-1, keepdim=True)
        abundance = torch.rand(2, 3, 5)
        abundance = abundance / abundance.sum(dim=-1, keepdim=True)
        outputs = {
            "abundance": abundance,
            "pred_delta_clr": torch.full((2, 3, 5), 0.25),
        }

        loss_without_l2 = combined_loss(outputs, context, target, _args(loss_delta_l2_weight=0.0))
        loss_with_l2 = combined_loss(outputs, context, target, _args(loss_delta_l2_weight=0.5))

        self.assertIn("delta_l2_loss", loss_with_l2)
        self.assertGreater(loss_with_l2["delta_l2_loss"].item(), 0)
        self.assertGreater(loss_with_l2["loss"].item(), loss_without_l2["loss"].item())

    def test_pred_delta_scale_shrinks_predicted_clr_delta(self):
        torch.manual_seed(19)
        kwargs = dict(
            num_asvs=5,
            context_len=4,
            horizon=3,
            token_dim=64,
            num_plants=2,
            hidden_size=32,
            num_heads=4,
            temporal_layers=1,
            asv_layers=1,
            dropout=0.0,
            token_bottleneck_dim=16,
        )
        model_full = TokenASVDeltaTransformer(**kwargs, pred_delta_scale=1.0)
        model_half = TokenASVDeltaTransformer(**kwargs, pred_delta_scale=0.5)
        model_half.load_state_dict(model_full.state_dict())
        context = torch.rand(2, 4, 5)
        context = context / context.sum(dim=-1, keepdim=True)
        tokens = torch.randn(2, 4, 5, 64)
        plant_id = torch.tensor([0, 1])

        full = model_full(context, tokens, plant_id)["pred_delta_clr"]
        half = model_half(context, tokens, plant_id)["pred_delta_clr"]

        self.assertTrue(torch.allclose(half, full * 0.5, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
