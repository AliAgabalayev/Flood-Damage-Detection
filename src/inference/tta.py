from __future__ import annotations

import torch
from torch import Tensor, nn

FLIP_DIMS: list[list[int]] = [[], [-1], [-2], [-1, -2]]


def tta_predict(model: nn.Module, batch: Tensor) -> Tensor:
    probs = []
    for dims in FLIP_DIMS:
        aug = torch.flip(batch, dims) if dims else batch
        logits = model(aug)
        prob = torch.sigmoid(logits)
        probs.append(torch.flip(prob, dims) if dims else prob)
    return torch.stack(probs).mean(dim=0)
