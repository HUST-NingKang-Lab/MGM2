"""Distribution-facing checks requiring no model downloads."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import mgm2


class InterfaceTests(unittest.TestCase):
    def test_public_exports(self):
        for name in ["MGM2", "EmbeddingBank", "prepare_sample", "collate_samples", "load_sae"]:
            self.assertTrue(callable(getattr(mgm2, name)))

    def test_cli_version(self):
        result = subprocess.run([sys.executable, "-m", "mgm2.cli", "--version"],
                                capture_output=True, text=True, check=True)
        self.assertIn(mgm2.__version__, result.stdout)

    def test_cli_help(self):
        result = subprocess.run([sys.executable, "-m", "mgm2.cli", "--help"],
                                capture_output=True, text=True, check=True)
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
