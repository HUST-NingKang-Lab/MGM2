"""Command-line entry point for the legacy NTv3 sequence route."""

import argparse

from . import __version__
from .sequences import encode_fasta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Encode FASTA with legacy NTv3_650M_pre")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--fasta", required=True, help="DNA FASTA; first header word is the feature ID"
    )
    parser.add_argument("--output", required=True, help="New HDF5 embedding bank")
    parser.add_argument(
        "--model-path", help="Local legacy NTv3 directory; otherwise use pinned Hub model"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-length", type=int, default=16384, help="Reject longer sequences; no truncation"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow NTv3 custom Python code"
    )
    args = parser.parse_args(argv)
    try:
        output = encode_fasta(
            args.fasta,
            args.output,
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
