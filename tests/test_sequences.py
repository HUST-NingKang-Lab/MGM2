"""Offline contract tests; no NTv3 weights or network access required."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from mgm2 import EmbeddingBank
from mgm2.sequences import (
    EMBEDDING_DIM,
    LEGACY_CODE_REVISION,
    LEGACY_MODEL_ID,
    LEGACY_MODEL_REVISION,
    _encode_batch,
    _load_encoder,
    encode_fasta,
    iter_fasta,
)


class Tokenizer:
    def __call__(self, sequences, return_tensors, add_special_tokens=False):
        assert not add_special_tokens
        if isinstance(sequences, str):
            sequences = [sequences]
        return {"input_ids": torch.tensor([["ACGTN".index(c) for c in s] for s in sequences])}


class Encoder(torch.nn.Module):
    def forward(self, ids, output_hidden_states):
        assert output_hidden_states
        assert not self.training
        assert not torch.is_grad_enabled()
        assert ids.shape[1] % 128 == 0
        # Depend on total padding length as well as each nucleotide. This detects
        # incorrect mixed-length padding and pooling over the appended Ns.
        states = (ids.float() + ids.shape[1]).unsqueeze(-1).expand(-1, -1, EMBEDDING_DIM)
        return SimpleNamespace(hidden_states=(states,))


@pytest.fixture
def encoder_loader():
    with patch(
        "mgm2.sequences._load_encoder", return_value=(Encoder().eval(), Tokenizer(), {})
    ) as mock:
        yield mock


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_archive_pooling_and_bank_contract(tmp_path, encoder_loader, batch_size):
    fasta = tmp_path / "sequences.fa"
    sequences = ["AC", "G" * 129, "T" * 128, "N" * 257, "ACGT" * 40]
    fasta.write_text("".join(f">asv{i} description\n{s}\n" for i, s in enumerate(sequences)))
    output = tmp_path / "bank.h5"
    encode_fasta(fasta, output, batch_size=batch_size)
    expected = []
    model, tokenizer = Encoder().eval(), Tokenizer()
    # Archived implementation, including its default tokenizer invocation.
    for sequence in sequences:
        padded = sequence + "N" * ((128 - len(sequence) % 128) % 128)
        ids = tokenizer(padded, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            hidden = model(ids, output_hidden_states=True).hidden_states[-1].squeeze(0).cpu()
        expected.append(hidden[: len(sequence)].mean(dim=0).numpy())
    with EmbeddingBank(output) as bank:
        assert bank.ids == [f"asv{i}" for i in range(len(sequences))]
        values, sources = bank.resolve(bank.ids)
        np.testing.assert_array_equal(values, expected)
        assert sources == ["ntv3"] * len(sequences)
    with h5py.File(output) as bank:
        metadata = json.loads(bank.attrs["provenance_json"])
        assert metadata["sequence_count"] == 5
        assert metadata["dtype"] == "float32"
        assert metadata["special_tokens"] is False


@pytest.mark.parametrize(
    "text", ["", "ACGT\n", ">\nAC\n", ">a\n", ">a\nAC\n>a\nG\n", ">a\nAU\n", ">a\nAX\n"]
)
def test_invalid_fasta_rejected_before_loading(tmp_path, encoder_loader, text):
    fasta = tmp_path / "invalid.fa"
    fasta.write_text(text)
    with pytest.raises(ValueError):
        encode_fasta(fasta, tmp_path / "bank.h5")
    encoder_loader.assert_not_called()


def test_fasta_lowercase_multiline_and_limit(tmp_path):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">id description\nac\ngtn\n")
    assert list(iter_fasta(fasta)) == [("id", "ACGTN")]
    with pytest.raises(ValueError, match="no truncation"):
        list(iter_fasta(fasta, max_length=4))


def test_existing_output_and_invalid_batch(tmp_path, encoder_loader):
    output = tmp_path / "output.h5"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        encode_fasta("missing.fa", output)
    with pytest.raises(ValueError, match="batch_size"):
        encode_fasta("missing.fa", output, batch_size=0)
    assert output.read_bytes() == b"existing"
    encoder_loader.assert_not_called()


def test_failed_encoding_removes_partial_output(tmp_path, encoder_loader):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">a\nACGT\n")
    with patch("mgm2.sequences._encode_batch", side_effect=RuntimeError("encoder failure")):
        with pytest.raises(RuntimeError):
            encode_fasta(fasta, tmp_path / "output.h5")
    assert not (tmp_path / "output.h5").exists()
    assert not list(tmp_path.glob(".ntv3-*"))


def test_changed_input_is_not_published(tmp_path, encoder_loader):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">a\nACGT\n")

    def load(*args):
        fasta.write_text(">a\nTGCA\n")
        return Encoder().eval(), Tokenizer(), {}

    encoder_loader.side_effect = load
    with pytest.raises(ValueError, match="changed"):
        encode_fasta(fasta, tmp_path / "output.h5")
    assert not (tmp_path / "output.h5").exists()


def test_batch_shape_guards():
    with pytest.raises(ValueError, match="same N-padded"):
        _encode_batch(["AC", "A" * 129], Encoder().eval(), Tokenizer(), "cpu")

    class WrongTokenizer(Tokenizer):
        def __call__(self, *args, **kwargs):
            return {"input_ids": torch.ones(1, 129, dtype=torch.long)}

    with pytest.raises(ValueError, match="one token"):
        _encode_batch(["AC"], Encoder().eval(), WrongTokenizer(), "cpu")


def test_loader_pins_weights_and_external_code():
    # Fake the optional dependency so base-only test installations also work.
    from unittest.mock import MagicMock

    transformers = MagicMock()
    config = SimpleNamespace(model_type="ntv3", auto_map={})
    transformers.AutoConfig.from_pretrained.return_value = config
    with (
        patch.dict("sys.modules", {"transformers": transformers}),
        patch("mgm2.sequences.version", return_value="4.57.3"),
    ):
        _, _, metadata = _load_encoder(None, "cpu", True, True)
    for factory in (
        transformers.AutoConfig,
        transformers.AutoTokenizer,
        transformers.AutoModelForMaskedLM,
    ):
        args, kwargs = factory.from_pretrained.call_args
        assert args == (LEGACY_MODEL_ID,)
        assert kwargs["revision"] == LEGACY_MODEL_REVISION
        assert kwargs["code_revision"] == LEGACY_CODE_REVISION
        assert kwargs["local_files_only"]
        assert kwargs["trust_remote_code"]
    transformers.AutoModelForMaskedLM.from_pretrained.return_value.to.return_value.eval.assert_called_once()
    assert metadata["model_revision"] == LEGACY_MODEL_REVISION


def test_custom_code_requires_opt_in():
    with pytest.raises(ValueError, match="explicitly"):
        _load_encoder(None, "cpu", False, False)
