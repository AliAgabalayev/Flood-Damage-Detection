import segmentation_models_pytorch as smp
import torch
from torch import Tensor, nn

from utils.config import Config

LOSSES = {"dice", "bce", "dice_bce", "focal"}


class MaskedLoss(nn.Module):
    """Segmentation loss that ignores invalid pixels via a per-pixel mask.

    Modes:
      * ``bce``      — pixel-wise BCE-with-logits, optional ``pos_weight`` to
                       up-weight the rare flood class.
      * ``dice``     — soft Dice (region overlap); intrinsically imbalance-robust.
      * ``focal``    — focal loss; ``alpha`` balances classes, ``gamma`` focuses
                       learning on hard/rare pixels.
      * ``dice_bce`` — weighted sum ``bce_weight * BCE + dice_weight * Dice``.

    All modes respect ``mask`` (1 = valid, 0 = ignore). For Dice/Focal the
    invalid logits are pushed to a large negative value and the target zeroed so
    those pixels contribute as confident true-negatives (zero overlap error).
    """

    def __init__(
        self,
        mode: str,
        pos_weight: Tensor | None = None,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_alpha: float | None = 0.25,
        focal_gamma: float = 2.0,
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

    def forward(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if self.mode == "bce":
            return self._bce(logits, target, mask)
        if self.mode == "dice":
            return self._dice(logits, target, mask)
        if self.mode == "focal":
            return self._focal(logits, target, mask)
        return (
            self.bce_weight * self._bce(logits, target, mask)
            + self.dice_weight * self._dice(logits, target, mask)
        )

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
    return MaskedLoss(
        name,
        w,
        bce_weight=cfg.training.bce_weight,
        dice_weight=cfg.training.dice_weight,
        focal_alpha=cfg.training.focal_alpha,
        focal_gamma=cfg.training.focal_gamma,
    )
