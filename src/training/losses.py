import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

from utils.config import Config

LOSSES = {
    "dice", "bce", "dice_bce", "focal",
    "tversky", "focal_tversky", "lovasz", "dice_focal",
}


class MaskedLoss(nn.Module):
    #Segmentation loss that ignores invalid pixels via a per-pixel mask

    def __init__(
        self,
        mode: str,
        pos_weight: Tensor | None = None,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_alpha: float | None = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp.losses.FocalLoss(
            mode="binary", alpha=focal_alpha, gamma=focal_gamma
        )
        self.tversky = smp.losses.TverskyLoss(
            mode="binary", from_logits=True,
            alpha=tversky_alpha, beta=tversky_beta, gamma=tversky_gamma,
        )
        self.lovasz = smp.losses.LovaszLoss(mode="binary", from_logits=True)

    def forward(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if self.mode == "bce":
            return self._bce(logits, target, mask)
        if self.mode == "dice":
            return self._dice(logits, target, mask)
        if self.mode == "focal":
            return self._focal(logits, target, mask)
        if self.mode in ("tversky", "focal_tversky"):
            return self._masked(self.tversky, logits, target, mask)
        if self.mode == "lovasz":
            return self._masked(self.lovasz, logits, target, mask)
        if self.mode == "dice_focal":
            return (
                self.dice_weight * self._dice(logits, target, mask)
                + self._focal(logits, target, mask)
            )
        return (
            self.bce_weight * self._bce(logits, target, mask)
            + self.dice_weight * self._dice(logits, target, mask)
        )

    def _bce(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss = self.bce(logits, target)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _dice(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return self._masked(self.dice, logits, target, mask)

    def _focal(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return self._masked(self.focal, logits, target, mask)

    @staticmethod
    def _masked(loss_fn: nn.Module, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return loss_fn(logits.masked_fill(mask == 0, -30.0), target * mask)


def build_loss(cfg: Config) -> MaskedLoss:
    name = cfg.training.loss
    if name not in LOSSES:
        raise ValueError(name)
    w = None
    if cfg.training.pos_weight is not None:
        w = torch.tensor([cfg.training.pos_weight])
    t = cfg.training
    return MaskedLoss(
        name,
        w,
        bce_weight=t.bce_weight,
        dice_weight=t.dice_weight,
        focal_alpha=t.focal_alpha,
        focal_gamma=t.focal_gamma,
        tversky_alpha=t.tversky_alpha,
        tversky_beta=t.tversky_beta,
        tversky_gamma=t.tversky_gamma,
    )
