# Encoding representative sequences

MGM2 accepts two complementary input routes: sequence embeddings from a FASTA
file, or identifier/taxonomy resolution against an existing embedding bank.
Use sequence encoding when representative ASV or OTU sequences are available.

## Legacy encoder

The sequence route uses [InstaDeepAI/NTv3_650M_pre](https://huggingface.co/InstaDeepAI/NTv3_650M_pre),
the original NTv3 release, rather than `NTv3_pretrained` from the revised release.
The loader fixes both repositories to immutable commits:

| Resource | Revision |
|---|---|
| `InstaDeepAI/NTv3_650M_pre` | `7506167a84fafca2a691a4bc6e7d821279edebc9` |
| `InstaDeepAI/ntv3_base_model` (custom code) | `27682ed2f537a4c4fcc356b6b8201e7dfceca602` |

These are the legacy repositories' revisions checked on 16 November 2026.
They identify this encoder entry point, not a recovered revision identifier
for the historical precomputed OTU bank.

Review and accept the access terms for both repositories on Hugging Face,
then authenticate using `hf auth login`. NTv3 has its own model licence;
weights are downloaded from InstaDeepAI, not redistributed in this package.
The explicit `--trust-remote-code` flag permits its custom Python implementation.

## FASTA to community embeddings

```bash
python -m pip install '.[ntv3,tables]'
mgm2-encode-fasta --fasta representative_sequences.fasta \
  --output outputs/asv_ntv3.h5 --device cuda:0 --trust-remote-code

mgm2-embed --model "$MODEL_ROOT/models/small" \
  --table communities.csv --embeddings outputs/asv_ntv3.h5 \
  --output outputs/communities.npz --device cuda:0
```

The first whitespace-delimited word in each FASTA header is the feature ID.
IDs must be unique and match the rows of the abundance table. Sequences may
span multiple lines; lowercase DNA is uppercased. Accepted bases are A/C/G/T/N.
Resolve other ambiguity codes explicitly before encoding. Empty records,
duplicate IDs and unsupported bases produce errors before model loading.
The default length guard is 16,384 bases, adjustable with `--max-length`;
longer sequences are rejected rather than truncated.

For an existing local legacy checkpoint:

```bash
mgm2-encode-fasta --fasta representative_sequences.fasta \
  --model-path /path/to/NTv3_650M_pre \
  --output outputs/asv_ntv3.h5 --device cuda:0 \
  --local-files-only --trust-remote-code
```

Local checkpoints must be copies of the legacy encoder. The loader records local
configuration and code hashes, but does not certify local weights against the
Hub checkpoint. If a local configuration references external custom code, the
pinned code must also be cached for `--local-files-only` to work. Without that
flag, referenced code can be fetched after authentication.

## Representation and execution

The extraction follows the archived MGM2 sequence pipeline:

1. Append N bases to make each sequence length a multiple of 128.
2. Tokenize at nucleotide resolution without added special tokens.
3. Run the masked-language model in evaluation mode with gradients disabled,
   using FP32 weights and no autocast.
4. Read the final post-deconvolution hidden state (`hidden_states[-1]`).
5. Move states to CPU and average only the original sequence positions,
   excluding appended N bases. Save the 1,536-dimensional vector without L2
   normalization.

The default `--batch-size 1` follows the archived per-sequence execution pattern.
Larger batches group sequences with the same N-padded length; they never add
extra padding solely to combine different length groups. Output remains in
FASTA order. FP32 batch execution can still introduce small floating-point
differences relative to single-sequence execution. Reduce batch size if GPU
memory is limited. Sequence vectors are written incrementally to disk.

The HDF5 output contains `otu_ids` and `embeddings`, ready for `EmbeddingBank`.
Its `provenance_json` attribute stores encoder revisions, pooling, padding,
precision, library versions, batching and a digest of the canonical FASTA
records. Completed files appear atomically; existing outputs are never
overwritten. Failed runs remove their temporary output.

## Python interface

```python
from mgm2.sequences import encode_fasta

encode_fasta(
    "representative_sequences.fasta",
    "outputs/asv_ntv3.h5",
    device="cuda:0",
    trust_remote_code=True,
)
```

## Verification

`tests/test_sequences.py` tests FASTA validation, legacy padding and pooling,
same-length batching, HDF5 interoperability, pinned loading and output cleanup
with a deterministic mock encoder. For a real-checkpoint smoke test, encode a
small FASTA using the archived script and this entry point with batch size 1,
the same local checkpoint, FP32, library versions and device, then compare
vectors after aligning IDs. This is separate from offline unit tests.
