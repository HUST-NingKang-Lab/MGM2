"""CLS insertion, padding and device placement."""

import torch


def collate_samples(samples, device="cpu"):
    """Prepend CLS and right-pad, without putting padding into bin 0."""
    if not samples:
        raise ValueError("An empty batch is not supported")
    lengths = [len(s["abundance_values"]) for s in samples]
    if min(lengths) < 1:
        raise ValueError("Empty communities are not supported")
    n, width, dim = len(samples), max(lengths) + 1, samples[0]["embeddings"].shape[1]
    ids = torch.full((n, width), -1, dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.long)
    abundance = torch.zeros((n, width), dtype=torch.float32)
    vectors = torch.zeros((n, width, dim), dtype=torch.float32)
    bins = torch.full((n, width), -100, dtype=torch.long)
    for i, (sample, length) in enumerate(zip(samples, lengths)):
        ids[i, 1 : length + 1] = torch.arange(length)
        mask[i, : length + 1] = 1
        abundance[i, 1 : length + 1] = torch.as_tensor(sample["abundance_values"])
        vectors[i, 1 : length + 1] = torch.as_tensor(sample["embeddings"])
        bins[i, 1 : length + 1] = torch.as_tensor(sample["bins"])
    return {
        k: v.to(device)
        for k, v in dict(
            input_ids=ids,
            embeddings=vectors,
            abundance_values=abundance,
            attention_mask=mask,
            bins=bins,
        ).items()
    }
