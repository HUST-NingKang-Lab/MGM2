import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from mgm2 import EmbeddingBank, prepare_sample, collate_samples
from mgm2._research_model import corrupt_abundance, reconstruction_sums, alignment_loss
from mgm2._sae import BatchTopKSAE


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "bank.h5"
        with h5py.File(self.path, "w") as f:
            f["otu_ids"] = np.asarray([b"a", b"b", b"c", b"d", b"e"])
            f["embeddings"] = np.arange(20, dtype=np.float32).reshape(5, 4)
        self.bank = EmbeddingBank(self.path)

    def tearDown(self):
        self.bank.close()
        self.temp.cleanup()

    def test_floor_is_not_renormalized(self):
        s = prepare_sample(["a", "b", "c"], [1e-7, .0099999, .99], self.bank)
        self.assertEqual(s["abundance_values"][0], -2.)
        np.testing.assert_allclose(s["abundance_values"][2], np.log10(.99) + 2, atol=1e-6)

    def test_zeros_omitted_and_bin0_retained(self):
        s = prepare_sample(["a", "b", "c", "d", "e"], [0, .001, .01, .1, .889], self.bank)
        self.assertNotIn("a", s["feature_ids"])
        self.assertEqual(s["bins"].min(), 0)
        batch = collate_samples([s, prepare_sample(["a"], [1], self.bank)])
        self.assertEqual(int(batch["bins"][1, 1]), 0)
        self.assertEqual(int(batch["bins"][1, 2]), -100)
        self.assertEqual(int(batch["attention_mask"][1, 2]), 0)

    def test_ties_stay_in_same_lower_bin(self):
        s = prepare_sample(["a", "b", "c"], [1, 1, 1], self.bank)
        np.testing.assert_array_equal(s["bins"], [0, 0, 0])

    def test_topk_before_floor(self):
        s = prepare_sample(["a", "b", "c"], [1, 2, 7], self.bank, top_k=2)
        self.assertEqual(s["feature_ids"], ["c", "b"])
        np.testing.assert_allclose(s["abundance_values"], np.log10([.7, .2]) + 2, atol=1e-6)

    def test_invalid_and_unknown_inputs_raise(self):
        for x in ([0, 0], [-1, 1], [float("nan"), 1], [float("inf"), 1]):
            with self.assertRaises(ValueError):
                prepare_sample(["a", "b"], x, self.bank)
        with self.assertRaises(KeyError):
            prepare_sample(["unknown"], [1], self.bank)
        with self.assertRaises(ValueError):
            prepare_sample(["a", "a"], [1, 1], self.bank)
        with self.assertRaises(ValueError):
            collate_samples([])

    def test_opt_in_taxonomy_and_global_fallback(self):
        path = Path(self.temp.name) / "fallback.h5"
        with h5py.File(path, "w") as f:
            f["taxon_keys"] = np.asarray([b"genus|Bacteroides"])
            f["taxon_ranks"] = np.asarray([b"genus"])
            f["taxon_paths"] = np.asarray([b"Bacteroides"])
            f["taxon_embeddings"] = np.ones((1, 4), dtype=np.float32)
            f["global_embedding"] = np.full(4, 2, dtype=np.float32)
        with EmbeddingBank(self.path, path) as bank:
            vectors, source = bank.resolve(["a", "g__Bacteroides", "unknown"])
            self.assertEqual(source, ["ntv3", "fallback:genus", "fallback:global"])
            np.testing.assert_array_equal(vectors[1:], [[1]*4, [2]*4])


class ObjectiveTests(unittest.TestCase):
    def test_all_bins_get_gradient(self):
        logits = torch.zeros(5, 5, requires_grad=True)
        prediction = torch.zeros(5, requires_grad=True)
        loss, _ = reconstruction_sums(logits, prediction, torch.arange(5), torch.ones(5))
        loss.backward()
        self.assertTrue((logits.grad.abs().sum(1) > 0).all())
        self.assertEqual(prediction.grad[:3].abs().sum().item(), 0)
        self.assertTrue((prediction.grad[3:].abs() > 0).all())

    def test_mask_only_valid_otus_and_reproducible(self):
        y = torch.tensor([[0., -2., -.5, 0.]])
        valid = torch.tensor([[False, True, True, False]])
        a, sel = corrupt_abundance(y, valid, seed=7)
        b, other = corrupt_abundance(y, valid, seed=7)
        self.assertTrue(torch.equal(a, b) and torch.equal(sel, other))
        self.assertEqual(int(sel.sum()), 1)
        self.assertFalse((sel & ~valid).any())

    def test_multi_positive_and_missing_text(self):
        x = torch.eye(3, requires_grad=True)
        loss, n = alignment_loss(x, x.detach(), torch.zeros(3, dtype=torch.long))
        self.assertEqual(n, 3)
        self.assertEqual(loss.item(), 0)
        loss, n = alignment_loss(x, x.detach(), torch.full((3,), -1))
        self.assertEqual(n, 0)
        self.assertEqual(loss.item(), 0)

    def test_sae_fixed_threshold_batch_invariance(self):
        torch.manual_seed(4)
        model = BatchTopKSAE(4, 16, 3).eval()
        with self.assertRaises(RuntimeError):
            model.encode(torch.ones(1, 4))
        model.threshold.fill_(.2)
        x = torch.randn(5, 4)
        torch.testing.assert_close(model.encode(x), torch.cat([model.encode(v[None]) for v in x]))
        torch.testing.assert_close(model.denormalize(model.normalize(x)), x)


if __name__ == "__main__":
    unittest.main()
