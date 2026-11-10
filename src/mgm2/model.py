"""Portable interface for the five-positive-bin S1 checkpoints."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from ._research_model import PilotModel


class MGM2(PilotModel):
    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        """Load a local exported directory, with dropout disabled by default.

        Hub downloading is separate: this method never executes remote code.
        """
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text())
        if config.get("format_version") != "mgm2-five-positive-bin-v1":
            raise ValueError("Expected a five-positive-bin release, not a legacy Lightning checkpoint")
        if config.get("num_bins") != 5 or config.get("absence_class") is not False:
            raise ValueError("Expected five positive abundance bins (bin 0 is not absence)")
        model = cls(config["architecture"])
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        # S1 trained the CLS decoder, not these retained compatibility heads.
        model.token_bin.requires_grad_(False)
        model.token_regression.requires_grad_(False)
        model.config = config
        return model.to(device).eval()

    def forward(self, batch):
        """Return CLS, contextual OTUs, mean-pooled OTUs and aligned text projection.

        `selected`, when supplied, selects masked OTUs for the trained CLS
        reconstruction head. Mean pooling excludes both CLS and padding.
        """
        fields = ("input_ids", "embeddings", "abundance_values", "attention_mask")
        hidden = self.backbone(**{k: batch[k] for k in fields})["last_hidden_state"]
        valid = (batch["input_ids"] >= 0) & batch["attention_mask"].bool()
        mean = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        out = dict(hidden=hidden, cls=hidden[:, 0], mean_pool=mean,
                   qwen=self.qwen_head(hidden[:, 0]))
        if "selected" in batch:
            selected = batch["selected"]
            if selected.dtype != torch.bool or selected.shape != valid.shape:
                raise ValueError("selected must be a boolean token mask")
            if (selected & ~valid).any():
                raise ValueError("CLS and padding cannot be reconstruction targets")
            out["cls_reconstruction"] = self.cls_decoder(
                out["cls"], batch["embeddings"][selected], selected.nonzero(as_tuple=True)[0])
        return out
