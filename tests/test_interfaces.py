"""Distribution-facing checks requiring no model downloads."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import mgm2


class InterfaceTests(unittest.TestCase):
    def test_legacy_imports_are_identical(self):
        from mgm2._research_model import PilotModel, reconstruction_sums
        from mgm2.inputs import EmbeddingBank, collate_samples, prepare_sample
        from mgm2.model import MGM2
        from mgm2.models.core import PilotModel as Core
        from mgm2.sae import load_sae
        from mgm2.training.objectives import reconstruction_sums as loss

        for name, value in dict(
            MGM2=MGM2,
            EmbeddingBank=EmbeddingBank,
            prepare_sample=prepare_sample,
            collate_samples=collate_samples,
            load_sae=load_sae,
        ).items():
            self.assertIs(value, getattr(mgm2, name))
        self.assertIs(PilotModel, Core)
        self.assertIs(reconstruction_sums, loss)

    def test_core_import_does_not_require_optional_extras(self):
        code = """
import sys
class BlockOptional:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'huggingface_hub', 'matplotlib', 'IPython'}:
            raise ImportError('Optional dependency must not load: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import mgm2
import mgm2.inference
assert 'mgm2.training.objectives' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_public_exports(self):
        for name in ["MGM2", "EmbeddingBank", "prepare_sample", "collate_samples", "load_sae"]:
            self.assertTrue(callable(getattr(mgm2, name)))

    def test_cli_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "mgm2.cli", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn(mgm2.__version__, result.stdout)

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "mgm2.cli", "--help"], capture_output=True, text=True, check=True
        )
        for flag in ["--model", "--embeddings", "--table", "--fallback", "--device"]:
            self.assertIn(flag, result.stdout)

    def test_clean_tutorial(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/01_getting_started.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertGreaterEqual(len(code), 6)
        for cell in code:
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            compile("".join(cell["source"]), str(path), "exec")
