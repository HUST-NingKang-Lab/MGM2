"""End-to-end table inference using a tiny local checkpoint, without downloads."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch
from safetensors.torch import save_file

from mgm2 import MGM2, EmbeddingBank, collate_samples, prepare_sample
from mgm2.cli import main
from mgm2.data.tables import read_abundance_table
from mgm2.inference import embed_csv, embed_table, save_embeddings
from mgm2.models.config import CONFIG


@unittest.skipUnless(importlib.util.find_spec("pandas"), "requires the tables extra")
class TableInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bank_path = self.root / "bank.h5"
        with h5py.File(self.bank_path, "w") as handle:
            handle["otu_ids"] = np.asarray([b"001", b"NA", b"c"])
            handle["embeddings"] = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
        self.csv = self.root / "table.csv"
        self.csv.write_text("otu,sample_b,sample_a\n001,1,0\nNA,2,4\nc,3,1\n")
        self.checkpoint = self.root / "model"
        self.checkpoint.mkdir()
        config = dict(
            CONFIG,
            embedding_dim=4,
            hidden_size=8,
            intermediate_size=16,
            heads=2,
            layers=1,
            qwen_dim=6,
        )
        torch.manual_seed(17)
        model = MGM2(config)
        save_file(model.state_dict(), self.checkpoint / "model.safetensors")
        (self.checkpoint / "config.json").write_text(
            json.dumps(
                dict(
                    format_version="mgm2-five-positive-bin-v1",
                    num_bins=5,
                    absence_class=False,
                    architecture=config,
                    scale="fixture",
                    source_checkpoint_sha256="fixture-sha",
                )
            )
        )
        self.model = MGM2.from_pretrained(self.checkpoint)
        self.output = self.root / "output.npz"

    def test_identifiers_preserve_lexical_values(self):
        table = read_abundance_table(self.csv)
        self.assertEqual(list(table.index), ["001", "NA", "c"])
        self.assertEqual(list(table.columns), ["sample_b", "sample_a"])

    def test_batched_api_matches_direct_forward(self):
        table = read_abundance_table(self.csv)
        with EmbeddingBank(self.bank_path) as bank:
            one = embed_table(table, self.model, bank, batch_size=1)
            two = embed_table(table, self.model, bank, batch_size=2)
            samples = [prepare_sample(table.index, table[col], bank) for col in table]
            with torch.inference_mode():
                direct = self.model(collate_samples(samples))
        for key in ("cls", "mean_pool", "qwen"):
            np.testing.assert_allclose(one[key], two[key], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(two[key], direct[key].numpy(), atol=1e-6, rtol=1e-5)
        self.assertEqual(one["sample_ids"].tolist(), ["sample_b", "sample_a"])
        self.assertEqual(one["resolution_counts"].tolist(), [[3, 0, 0], [2, 0, 0]])
        self.assertEqual(one["source_checkpoint_sha256"].item(), "fixture-sha")

    def test_sample_can_share_feature_column_header(self):
        self.csv.write_text("otu,otu\n001,1\nNA,2\n")
        table = read_abundance_table(self.csv)
        self.assertEqual(list(table.columns), ["otu"])
        self.assertEqual(list(table.index), ["001", "NA"])

    def test_csv_and_cli_use_same_pipeline(self):
        count = embed_csv(self.checkpoint, self.csv, self.bank_path, self.output, batch_size=1)
        self.assertEqual(count, 2)
        cli_output = self.root / "cli.npz"
        main(
            [
                "--model",
                str(self.checkpoint),
                "--table",
                str(self.csv),
                "--embeddings",
                str(self.bank_path),
                "--output",
                str(cli_output),
                "--batch-size",
                "1",
            ]
        )
        with (
            np.load(self.output, allow_pickle=False) as expected,
            np.load(cli_output, allow_pickle=False) as actual,
        ):
            self.assertEqual(set(expected.files), set(actual.files))
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_requires_eval_without_changing_model_mode(self):
        table = read_abundance_table(self.csv)
        self.model.train()
        with (
            EmbeddingBank(self.bank_path) as bank,
            self.assertRaisesRegex(ValueError, "evaluation mode"),
        ):
            embed_table(table, self.model, bank)
        self.assertTrue(self.model.training)

    def test_rejects_invalid_batches_before_loading(self):
        for size in (0, -1, True, 1.5):
            with patch("mgm2.inference.tables.MGM2.from_pretrained") as load:
                with self.assertRaises(ValueError):
                    embed_csv(
                        self.checkpoint, self.csv, self.bank_path, self.output, batch_size=size
                    )
                load.assert_not_called()

    def test_invalid_csv_fails_before_model_loading(self):
        invalid = [
            "otu,a,a\n001,1,2\n",
            "otu,a,\n001,1,2\n",
            "otu,a\n001,1\n001,2\n",
            "otu,a\n,1\n",
            "otu,a\n001,-1\n",
            "otu,a\n001,NaN\n",
            "otu,a\n001,inf\n",
            "otu,a\n001,hello\n",
            "otu,a\n001,0\n",
            "otu,a\n",
        ]
        for contents in invalid:
            with self.subTest(csv=contents):
                self.csv.write_text(contents)
                with patch("mgm2.inference.tables.MGM2.from_pretrained") as load:
                    with self.assertRaises(ValueError):
                        embed_csv(self.checkpoint, self.csv, self.bank_path, self.output)
                    load.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_unknown_feature_does_not_create_output(self):
        self.csv.write_text("otu,a\nmissing,1\n")
        with self.assertRaises(KeyError):
            embed_csv(self.checkpoint, self.csv, self.bank_path, self.output)
        self.assertFalse(self.output.exists())

    def test_refuses_overwrite_and_pickle_arrays(self):
        save_embeddings({"cls": np.ones((2, 3))}, self.output)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            embed_csv(self.checkpoint, self.csv, self.bank_path, self.output)
        with self.assertRaises(FileExistsError):
            save_embeddings({"cls": np.zeros((2, 3))}, self.output)
        self.assertEqual(before, self.output.read_bytes())
        with self.assertRaises(ValueError):
            save_embeddings({"bad": np.asarray([object()])}, self.root / "unsafe.npz")
        self.assertFalse((self.root / "unsafe.npz").exists())

    def test_stringified_identifier_collisions_rejected(self):
        import pandas as pd

        table = pd.DataFrame([[1], [2]], index=[1, "1"], columns=["a"])
        with EmbeddingBank(self.bank_path) as bank, self.assertRaisesRegex(ValueError, "unique"):
            embed_table(table, self.model, bank)

    def test_cli_reports_input_errors_without_traceback(self):
        with patch("sys.stderr") as stderr, self.assertRaises(SystemExit) as error:
            main(
                [
                    "--model",
                    str(self.checkpoint),
                    "--table",
                    str(self.csv),
                    "--embeddings",
                    str(self.bank_path),
                    "--output",
                    str(self.output),
                    "--batch-size",
                    "0",
                ]
            )
        self.assertEqual(error.exception.code, 2)
        self.assertIn("positive integer", "".join(str(c) for c in stderr.write.call_args_list))
