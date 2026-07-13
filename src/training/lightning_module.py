import pytorch_lightning as pl
import torch
from torch import Tensor

from inference.evaluate import build_scorer, masked_select
from inference.tta import predict_prob
from models.build import build_model
from training.losses import build_loss
from utils.config import Config


class FloodModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.net = build_model(cfg)
        self.loss = build_loss(cfg)
        self.val_metrics = build_scorer()

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def _prep(self, batch: tuple[Tensor, Tensor, Tensor, list[str]]) -> tuple[Tensor, Tensor, Tensor]:
        img, lbl, mask, _paths = batch
        return img, lbl.unsqueeze(1).float(), mask.unsqueeze(1).float()

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor, list[str]], idx: int) -> Tensor:
        img, target, valid = self._prep(batch)
        loss = self.loss(self.net(img), target, valid)
        self.log("train_loss", loss, prog_bar=True, batch_size=img.size(0))
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor, list[str]], idx: int) -> None:
        img, target, valid = self._prep(batch)
        logits = self.net(img)
        loss = self.loss(logits, target, valid)
        prob = predict_prob(self.net, img, tta=self.cfg.inference.tta, logits=logits)
        p, truth = masked_select(prob, target, valid)
        self.val_metrics.update(p, truth)
        self.log("val_loss", loss, prog_bar=True, batch_size=img.size(0))

    def on_validation_epoch_end(self) -> None:
        scores = self.val_metrics.compute()
        self.log("val_iou", scores["iou"], prog_bar=True)
        self.log("val_f1", scores["f1"], prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        lr = self.cfg.training.lr
        if self.cfg.training.optimizer == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr)
        if self.cfg.training.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        return torch.optim.Adam(self.parameters(), lr=lr)
