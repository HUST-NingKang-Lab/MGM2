import h5py
import numpy as np
import torch

from src.models.BatchTopKSAE import BatchTopKSAE
from scripts.sae import build_pretrain_sae_sampling_manifest as manifest
from scripts.sae import extract_mgm2_sae_activations as extract
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


def test_sae_manifest_reads_mgm2v1_otu_data_and_transforms_abundance():
    data = {
        "samples": ["sample-a"],
        "otu_data": {
            "sample-a": {
                "input_ids": np.array([-1, 10, 20, 30], dtype=np.int64),
                "attention_mask": np.array([1, 1, 1, 0], dtype=np.int64),
                "abundance_values": np.array([0.0, 0.2, 1e-6, 0.5], dtype=np.float32),
            }
        },
    }

    positions, otu_indices, processed, raw = manifest.valid_token_arrays(data, 0)

    np.testing.assert_array_equal(positions, np.array([1, 2]))
    np.testing.assert_array_equal(otu_indices, np.array([10, 20]))
    np.testing.assert_allclose(raw, np.array([0.2, 1e-6], dtype=np.float32))
    np.testing.assert_allclose(processed, np.array([np.log10(0.2) + 2.0, -2.0], dtype=np.float32), rtol=1e-5)


def test_sae_extract_builds_batches_from_mgm2v1_otu_data_with_model_abundance():
    data = {
        "samples": ["sample-a"],
        "otu_data": {
            "sample-a": {
                "input_ids": np.array([-1, 10, 20], dtype=np.int64),
                "attention_mask": np.array([1, 1, 1], dtype=np.int64),
                "abundance_values": np.array([0.0, 0.2, 1e-6], dtype=np.float32),
            }
        },
    }
    ntv3 = np.arange(40, dtype=np.float32).reshape(40, 1)

    input_ids, attention_mask, abundance, embeddings = extract._build_batch(
        data,
        [{"sample_index": 0}],
        ntv3,
    )

    np.testing.assert_array_equal(input_ids, np.array([[-1, 10, 20]], dtype=np.int64))
    np.testing.assert_array_equal(attention_mask, np.array([[1, 1, 1]], dtype=np.int64))
    np.testing.assert_allclose(abundance, np.array([[0.0, np.log10(0.2) + 2.0, -2.0]], dtype=np.float32), rtol=1e-5)
    np.testing.assert_array_equal(embeddings[:, 0, :], np.zeros((1, 1), dtype=np.float32))
    np.testing.assert_array_equal(embeddings[:, 1:, 0], np.array([[10.0, 20.0]], dtype=np.float32))
