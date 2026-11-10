"""Embed a feature-by-sample CSV with MGM2."""

import argparse
from pathlib import Path

from . import __version__
from .inference import embed_csv


def build_parser():
    """Construct the CLI parser without parsing or loading model resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--fallback", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv=None):
    """Parse arguments and delegate to the reusable inference pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        count = embed_csv(
            args.model,
            args.table,
            args.embeddings,
            args.output,
            fallback_path=args.fallback,
            batch_size=args.batch_size,
            device=args.device,
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Saved {count} samples to {args.output}")


if __name__ == "__main__":
    main()
