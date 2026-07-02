import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

from utils.config import Config

LOSSES = {"dice", "bce", "dice_bce", "focal"}


class MaskedLoss(nn.Module):
    def __init__(self, mode: str, pos_weight: Tensor | None = None):
        super().__init__()
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="binary")

    def forward(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if self.mode == "bce":
            return self._bce(logits, target, mask)
        if self.mode == "dice":
            return self._dice(logits, target, mask)
        if self.mode == "focal":
            return self._focal(logits, target, mask)
        return self._bce(logits, target, mask) + self._dice(logits, target, mask)

    def _bce(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = self.bce(logits, target)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _dice(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return self.dice(logits.masked_fill(mask == 0, -30.0), target * mask)

    def _focal(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return self.focal(logits.masked_fill(mask == 0, -30.0), target * mask)


def build_loss(cfg: Config) -> MaskedLoss:
    name = cfg.training.loss
    if name not in LOSSES:
        raise ValueError(name)
    w = None
    if cfg.training.pos_weight is not None:
        w = torch.tensor([cfg.training.pos_weight])
    return MaskedLoss(name, w)
