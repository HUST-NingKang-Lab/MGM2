"""Formal BatchTopK SAE with calibrated inference threshold."""
import torch
from torch import nn
from torch.nn import functional as F

class BatchTopKSAE(nn.Module):
    """Batch budget in training; a frozen scalar threshold in all inference.

    Unlike the historical SAE, the encoder has its own bias. A discovery-only
    affine normalizer is stored in the checkpoint, preserving relative norms.
    """
    def __init__(self, dimension, width, k):
        super().__init__()
        self.dimension, self.width, self.k = dimension, width, k
        self.decoder = nn.Parameter(F.normalize(torch.randn(width, dimension), dim=1))
        self.encoder = nn.Parameter(self.decoder.detach().T.clone())
        self.encoder_bias = nn.Parameter(torch.zeros(width))
        self.decoder_bias = nn.Parameter(torch.zeros(dimension))
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("threshold", torch.tensor(float("inf")))
        self.register_buffer("inactive_tokens", torch.zeros(width, dtype=torch.long))

    def normalize(self, x):
        return (x - self.mean) / self.scale

    def denormalize(self, x):
        return x * self.scale + self.mean

    def dense(self, normalized):
        return F.relu((normalized - self.decoder_bias) @ self.encoder + self.encoder_bias)

    def batch_gate(self, dense):
        values, indices = dense.flatten().topk(min(self.k * len(dense), dense.numel()), sorted=False)
        return torch.zeros_like(dense).flatten().scatter(0, indices, values).view_as(dense)

    def encode(self, x, normalized=False):
        if not torch.isfinite(self.threshold):
            raise RuntimeError("Calibrate inference threshold on discovery data before encoding")
        acts = self.dense(x if normalized else self.normalize(x))
        return acts * (acts > self.threshold)

    def decode(self, acts):
        return acts @ self.decoder + self.decoder_bias

    def training_loss(self, x, auxiliary_weight, dead_after):
        dense = self.dense(x)
        acts = self.batch_gate(dense)
        recon = self.decode(acts)
        residual = x - recon
        mse = residual.square().mean()
        with torch.no_grad():
            self.inactive_tokens += len(x)
            self.inactive_tokens[(acts > 0).any(0)] = 0
        dead = self.inactive_tokens > dead_after
        aux = mse.new_zeros(())
        if dead.any() and auxiliary_weight:
            candidates = dense[:, dead]
            # Per-token residual auxiliary reconstruction, with fixed bounded K.
            v, ix = candidates.topk(min(32, candidates.shape[1]), dim=1)
            sparse = torch.zeros_like(candidates).scatter(1, ix, v)
            aux = (sparse @ self.decoder[dead] - residual.detach()).square().mean()
        return mse + auxiliary_weight * aux, mse.detach(), acts.detach()

    @torch.no_grad()
    def constrain_decoder(self, project_gradient=False):
        self.decoder.copy_(F.normalize(self.decoder, dim=1))
        if project_gradient and self.decoder.grad is not None:
            self.decoder.grad -= (self.decoder.grad * self.decoder).sum(1, keepdim=True) * self.decoder

    @torch.no_grad()
    def calibrate(self, normalized_data, batches, batch_size, seed):
        generator = torch.Generator(device=normalized_data.device).manual_seed(seed)
        cutoffs = []
        for _ in range(batches):
            ix = torch.randint(len(normalized_data), (batch_size,), generator=generator, device=normalized_data.device)
            dense = self.dense(normalized_data[ix])
            selected = dense.flatten().topk(min(self.k * batch_size, dense.numel()), sorted=False).values
            positive = selected[selected > 0]
            cutoffs.append(positive.min() if positive.numel() else dense.new_zeros(()))
        self.threshold.copy_(torch.stack(cutoffs).mean())
