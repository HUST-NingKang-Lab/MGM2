"""Masking, reconstruction and semantic-alignment objectives."""

import numpy as np
import torch
from torch.nn import functional as F

from ..models.config import CONFIG, VARIANTS


def corrupt_abundance(original, valid, seed, mask_prob=0.15, ratios=(0.8, 0.1, 0.1)):
    if not 0 <= mask_prob <= 1 or len(ratios) != 3 or not np.isclose(sum(ratios), 1):
        raise ValueError("Invalid masking config")
    generator = torch.Generator(device=original.device).manual_seed(int(seed))
    counts = valid.sum(1)
    k = (counts.float() * mask_prob).long()
    if mask_prob > 0:
        k = torch.where(counts > 0, k.clamp_min(1), k)
    scores = torch.rand(original.shape, device=original.device, generator=generator).masked_fill(
        ~valid, 2.0
    )
    ranks = scores.argsort(dim=1).argsort(dim=1)
    selected = (ranks < k[:, None]) & valid
    branches = torch.rand(original.shape, device=original.device, generator=generator)
    positions = torch.arange(original.shape[1], device=original.device)[None, :].expand_as(original)
    choices = positions.masked_fill(~valid, original.shape[1]).sort(dim=1).values
    pick = (
        torch.rand(original.shape, device=original.device, generator=generator) * counts[:, None]
    ).long()
    donor = choices.gather(1, pick).clamp_max(original.shape[1] - 1)
    output = torch.where(selected & (branches < ratios[0]), -3.0, original)
    random_branch = selected & (branches >= ratios[0]) & (branches < ratios[0] + ratios[1])
    output = torch.where(random_branch, original.gather(1, donor), output)
    return output, selected


def alignment_loss(projection, target, groups, temperature=0.07):
    valid = groups >= 0
    projection = projection[valid].float()
    target = target[valid].float()
    groups = groups[valid]
    n = len(groups)
    if n < 2:
        return projection.sum() * 0, n
    with torch.autocast(device_type=projection.device.type, enabled=False):
        logits = F.normalize(projection, dim=-1) @ F.normalize(target, dim=-1).T / temperature
    positives = groups[:, None] == groups[None, :]
    positive_logits = logits.masked_fill(~positives, float("-inf"))
    loss = 0.5 * (
        (logits.logsumexp(1) - positive_logits.logsumexp(1)).mean()
        + (logits.logsumexp(0) - positive_logits.logsumexp(0)).mean()
    )
    return loss, n


def reconstruction_sums(logits, prediction, bins, target):
    logits = logits.float()
    prediction = prediction.float()
    target = target.float()
    n = bins.numel()
    reg = bins >= 3
    nr = reg.sum()
    if n:
        ce = F.cross_entropy(logits, bins.long(), reduction="none")
        focal = (0.25 * (1 - torch.exp(-ce)).square() * ce).sum()
    else:
        focal = logits.sum() * 0
    residual = prediction[reg] - target[reg]
    mse = residual.square().sum()
    mae = residual.abs().sum()
    loss = focal / max(n, 1) + CONFIG["regression_weight"] * mse / nr.clamp_min(1)
    # Sufficient statistics, not means of batch means/correlations.
    r = prediction[reg]
    y = target[reg]
    stats = torch.stack(
        [
            focal.detach(),
            torch.as_tensor(n, device=logits.device),
            ((logits.argmax(-1) == bins).sum()).float(),
            mse.detach(),
            mae.detach(),
            nr.float(),
            r.detach().sum(),
            y.sum(),
            r.detach().square().sum(),
            y.square().sum(),
            (r.detach() * y).sum(),
        ]
    ).double()
    return loss, stats


def objective(outputs, batch, variant):
    selected = batch["selected"]
    bins = batch["bins"][selected]
    target = batch["target_abundance"][selected]
    loss = outputs["cls"].sum() * 0
    stats = {}
    for key, weight in zip(["token", "cls_reconstruction"], VARIANTS[variant]):
        if weight:
            term, stats[key] = reconstruction_sums(*outputs[key], bins, target)
            loss = loss + weight * term
    align, n = alignment_loss(
        outputs["qwen"], batch["qwen"], batch["text_groups"], CONFIG["temperature"]
    )
    loss = loss + CONFIG["qwen_weight"] * align
    stats["qwen"] = torch.stack(
        [align.detach().double() * n, torch.tensor(n, device=align.device, dtype=torch.float64)]
    )
    return loss, stats


def summarize_statistics(sums, variant):
    result = {}
    total = 0.0
    for key, weight in zip(["token", "cls_reconstruction"], VARIANTS[variant]):
        if not weight:
            continue
        s = np.asarray(sums[key])
        focal = s[0] / max(s[1], 1)
        mse = s[3] / max(s[5], 1)
        denom = np.sqrt(max(s[5] * s[8] - s[6] ** 2, 0) * max(s[5] * s[9] - s[7] ** 2, 0))
        corr = (s[5] * s[10] - s[6] * s[7]) / denom if denom > 0 else None
        result[key] = dict(
            focal=focal,
            accuracy=s[2] / max(s[1], 1),
            mse=mse,
            mae=s[4] / max(s[5], 1),
            correlation=corr,
            masked_tokens=int(s[1]),
            regression_tokens=int(s[5]),
        )
        total += weight * (focal + CONFIG["regression_weight"] * mse)
    q = np.asarray(sums["qwen"])
    result["qwen_loss"] = float(q[0] / max(q[1], 1))
    result["qwen_anchors"] = int(q[1])
    result["total_loss"] = float(total + CONFIG["qwen_weight"] * result["qwen_loss"])
    return result
