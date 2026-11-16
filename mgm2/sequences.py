"""User FASTA → legacy NTv3 embeddings for MGM2.

Transformers is optional and imported only when loading the sequence encoder.
"""

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from importlib.metadata import version
from itertools import islice
from pathlib import Path

import h5py
import numpy as np
import torch

LEGACY_MODEL_ID = "InstaDeepAI/NTv3_650M_pre"
LEGACY_MODEL_REVISION = "7506167a84fafca2a691a4bc6e7d821279edebc9"
LEGACY_CODE_ID = "InstaDeepAI/ntv3_base_model"
LEGACY_CODE_REVISION = "27682ed2f537a4c4fcc356b6b8201e7dfceca602"
EMBEDDING_DIM = 1536


def iter_fasta(path, *, max_length=16384):
    """Yield unique first-word IDs and uppercase DNA, rejecting invalid records."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    seen = set()
    name, pieces, length = None, [], 0

    def finish():
        if not pieces:
            raise ValueError(f"Empty FASTA sequence: {name}")
        return name, "".join(pieces)

    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield finish()
                fields = line[1:].split()
                if not fields:
                    raise ValueError(f"Missing FASTA ID at line {line_number}")
                name = fields[0]
                if "\x00" in name or name in seen:
                    raise ValueError(f"Invalid or duplicate FASTA ID: {name!r}")
                seen.add(name)
                pieces, length = [], 0
            else:
                if name is None:
                    raise ValueError(f"Sequence before FASTA header at line {line_number}")
                sequence = line.upper()
                if set(sequence) - set("ACGTN"):
                    raise ValueError(f"Only A/C/G/T/N are accepted: {name}, line {line_number}")
                length += len(sequence)
                if length > max_length:
                    raise ValueError(
                        f"{name} exceeds max_length={max_length}; no truncation applied"
                    )
                pieces.append(sequence)
    if name is None:
        raise ValueError("FASTA contains no records")
    yield finish()


def _update_digest(digest, name, sequence):
    digest.update((name + "\0" + sequence + "\n").encode("utf-8"))


def _load_encoder(model_path, device, local_files_only, trust_remote_code):
    if not trust_remote_code:
        raise ValueError("NTv3 requires custom code; explicitly set trust_remote_code=True")
    try:
        from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Install the sequence extra: pip install '.[ntv3]'") from error

    source = str(model_path) if model_path is not None else LEGACY_MODEL_ID
    local = model_path is not None
    if local and not Path(model_path).is_dir():
        raise ValueError("model_path must be a local legacy NTv3 directory")
    options = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "code_revision": LEGACY_CODE_REVISION,
    }
    if not local:
        options["revision"] = LEGACY_MODEL_REVISION
    config = AutoConfig.from_pretrained(source, **options)
    if config.model_type != "ntv3":
        raise ValueError("Expected the legacy NTv3 model configuration")
    for reference in getattr(config, "auto_map", {}).values():
        references = reference if isinstance(reference, (list, tuple)) else [reference]
        for entry in references:
            if isinstance(entry, str) and "--" in entry:
                if entry.split("--", 1)[0] != LEGACY_CODE_ID:
                    raise ValueError("Unexpected custom-code repository in NTv3 configuration")
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    model = AutoModelForMaskedLM.from_pretrained(
        source, config=config, torch_dtype=torch.float32, **options
    ).to(device)
    model.eval()
    provenance = {
        "model_id": LEGACY_MODEL_ID,
        "source": str(Path(source).resolve()) if local else source,
        "model_revision": "local-user-supplied" if local else LEGACY_MODEL_REVISION,
        "code_repository": LEGACY_CODE_ID,
        "remote_code_revision": LEGACY_CODE_REVISION,
        "transformers_version": version("transformers"),
    }
    if local:
        # Local copies can include older bundled code: identify it separately.
        files = sorted(Path(source).glob("*.py")) + sorted(Path(source).glob("*.json"))
        provenance["local_metadata_sha256"] = {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files
        }
    return model, tokenizer, provenance


def _encode_batch(sequences, model, tokenizer, device):
    """Match archived per-sequence N-padding; batch only identical padded lengths."""
    padded = [sequence + "N" * (-len(sequence) % 128) for sequence in sequences]
    width = len(padded[0])
    if any(len(sequence) != width for sequence in padded):
        raise ValueError("Batch sequences must have the same N-padded length")
    tokens = tokenizer(padded, return_tensors="pt", add_special_tokens=False)
    ids = tokens["input_ids"]
    if tuple(ids.shape) != (len(sequences), width):
        raise ValueError("Expected one token per nucleotide and no special tokens")
    with torch.inference_mode():
        output = model(ids.to(device), output_hidden_states=True)
        hidden = output.hidden_states[-1]
        if tuple(hidden.shape) != (len(sequences), width, EMBEDDING_DIM):
            raise ValueError("Expected final nucleotide-resolution NTv3 states with width 1536")
        # The archived extractor moves states to CPU before taking the mean.
        hidden = hidden.float().cpu()
        vectors = torch.stack(
            [hidden[i, : len(sequence)].mean(dim=0) for i, sequence in enumerate(sequences)]
        ).numpy()
    if not np.isfinite(vectors).all():
        raise ValueError("NTv3 produced non-finite embeddings")
    return vectors


def encode_fasta(
    fasta_path,
    output_path,
    *,
    model_path=None,
    device="cpu",
    batch_size=1,
    max_length=16384,
    local_files_only=False,
    trust_remote_code=False,
):
    """Write a compatible HDF5 bank without changing an existing output.

    Remote loading pins the legacy weight and code repositories separately.
    A local model directory is an explicit user-provided legacy checkpoint.
    Batch size 1 reproduces the archived execution pattern; larger batches
    combine only sequences with the same padded length. All inference is FP32.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    expected_digest = hashlib.sha256()
    count = 0
    # Validate all IDs and sequences before loading the large encoder.
    for name, sequence in iter_fasta(fasta_path, max_length=max_length):
        _update_digest(expected_digest, name, sequence)
        count += 1
    model, tokenizer, metadata = _load_encoder(
        model_path, device, local_files_only, trust_remote_code
    )
    metadata.update(
        format="mgm2-ntv3-bank-v1",
        pooling="final_deconv_mean_over_original_bases",
        padding="N_to_multiple_of_128",
        special_tokens=False,
        normalization="none",
        dtype="float32",
        model_eval=True,
        batch_size=batch_size,
        max_length=max_length,
        sequence_count=count,
        canonical_fasta_sha256=expected_digest.hexdigest(),
        torch_version=str(torch.__version__),
        device=str(device),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".ntv3-", suffix=".h5", dir=output_path.parent)
    os.close(fd)
    try:
        actual_digest = hashlib.sha256()
        with h5py.File(temporary, "w") as bank:
            ids = bank.create_dataset("otu_ids", (count,), dtype=h5py.string_dtype("utf-8"))
            vectors = bank.create_dataset("embeddings", (count, EMBEDDING_DIM), dtype="float32")
            records = iter(iter_fasta(fasta_path, max_length=max_length))
            offset = 0
            while chunk := list(islice(records, max(64, batch_size))):
                groups = defaultdict(list)
                for index, (name, sequence) in enumerate(chunk):
                    _update_digest(actual_digest, name, sequence)
                    ids[offset + index] = name
                    groups[(len(sequence) + 127) // 128].append((offset + index, sequence))
                for group in groups.values():
                    for start in range(0, len(group), batch_size):
                        batch = group[start : start + batch_size]
                        rows, sequences = zip(*batch)
                        vectors[list(rows)] = _encode_batch(sequences, model, tokenizer, device)
                offset += len(chunk)
            if offset != count or actual_digest.digest() != expected_digest.digest():
                raise ValueError("FASTA changed during encoding; output discarded")
            bank.attrs["provenance_json"] = json.dumps(metadata, sort_keys=True)
            bank.flush()
        # Atomic publication with no overwrite, including concurrent writers.
        os.link(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output_path
