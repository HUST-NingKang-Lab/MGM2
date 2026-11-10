"""Embed a constructed community using three entries from the release vocabulary."""
import argparse
import json

import torch

from mgm2 import MGM2, EmbeddingBank, collate_samples, prepare_sample


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--scale", choices=["small", "medium", "large", "xlarge"], default="small")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = MGM2.from_pretrained(args.model_root / "models" / args.scale, device=args.device)
    with EmbeddingBank(args.model_root / "embeddings/otus_99_embeddings.h5") as bank:
        sample = prepare_sample(bank.ids[:3], [20, 5, 1], bank)
    with torch.inference_mode():
        outputs = model(collate_samples([sample], device=args.device))
    print(json.dumps({name: list(value.shape) for name, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
