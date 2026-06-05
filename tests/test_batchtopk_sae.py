import h5py
import numpy as np
import torch

from src.models.BatchTopKSAE import BatchTopKSAE
from scripts.sae.train_batchtopk_sae import H5ActivationStore


def test_batchtopk_sae_outputs_reconstruction_and_sparse_features():
    model = BatchTopKSAE(input_dim=8, dict_size=16, top_k=3, input_unit_norm=True)
    x = torch.randn(4, 8)

    out = model(x)

    assert out.sae_out.shape == x.shape
    assert out.feature_acts.shape == (4, 16)
    assert out.num_active == 12
    assert out.loss.ndim == 0


def test_batchtopk_sae_keeps_decoder_rows_unit_norm():
    model = BatchTopKSAE(input_dim=8, dict_size=16, top_k=2)
    norms = model.W_dec.norm(dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    x = torch.randn(4, 8)
    out = model(x)
    out.loss.backward()
    model.make_decoder_weights_and_grad_unit_norm()
    model.normalize_decoder_weights()

    norms = model.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_batchtopk_sae_can_count_dead_features_by_forwards():
    model = BatchTopKSAE(input_dim=8, dict_size=16, top_k=2, dead_feature_threshold=1, inactive_counter_mode="forwards")
    model.train()
    x = torch.randn(4, 8)

    model(x)
    assert model.num_dead_features().item() == 0

    model(x)
    assert model.num_dead_features().item() >= 0
    assert int(model.num_tokens_since_fired.max().item()) <= 2


def test_h5_activation_store_preloads_by_sequential_scan(tmp_path):
    path = tmp_path / "activations.h5"
    activations = np.arange(12 * 4, dtype=np.float16).reshape(12, 4)
    split = np.array([0, 1, 0, 2, 0, 1, 0, 0, 1, 2, 0, 1], dtype=np.int8)
    token_type = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=np.int8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("activations", data=activations, chunks=(3, 4))
        handle.create_dataset("split", data=split)
        handle.create_dataset("token_type", data=token_type)

    store = H5ActivationStore(
        path,
        token_type="otu",
        seed=1,
        preload_to_memory=True,
        preload_splits=["train", "val"],
        preload_dtype="float16",
        preload_chunk_rows=4,
    )

    try:
        assert store.counts()["train"] == 4
        assert store.counts()["val"] == 3
        assert store.preloaded["train"].dtype == np.float16
        assert store.preloaded["train"].shape == (4, 4)
        np.testing.assert_array_equal(store.preloaded["train"], activations[[0, 4, 6, 7]])
        np.testing.assert_array_equal(store.preloaded["val"], activations[[1, 8, 11]])
    finally:
        store.close()
