import argparse

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from inference.tta import tta_predict
from utils.config import Config, load_config


def build_scorer() -> MetricCollection:
    return MetricCollection({"iou": BinaryJaccardIndex(), "f1": BinaryF1Score()})


def masked_select(prob: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    valid = mask.bool()
    return prob[valid], target[valid].int()


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str, tta: bool = False) -> dict:
    model.eval().to(device)
    scorer = build_scorer().to(device)
    with torch.no_grad():
        for img, lbl, mask in loader:
            img, lbl, mask = img.to(device), lbl.to(device), mask.to(device)
            prob = tta_predict(model, img) if tta else torch.sigmoid(model(img))
            target = lbl.unsqueeze(1).float()
            valid = mask.unsqueeze(1).float()
            p, truth = masked_select(prob, target, valid)
            scorer.update(p, truth)
    return {k: float(v) for k, v in scorer.compute().items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tta", action="store_true", help="flip-based test-time augmentation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from data.datamodule import FloodDataModule
    from training.lightning_module import FloodModel

    dm = FloodDataModule(cfg)
    dm.setup("test" if args.split == "test" else "validate")
    model = FloodModel(cfg)
    ckpt = torch.load(cfg.inference.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    device = "cuda" if cfg.training.device == "cuda" and torch.cuda.is_available() else "cpu"
    loader = dm.test_dataloader() if args.split == "test" else dm.val_dataloader()
    print(evaluate(model, loader, device, tta=args.tta or cfg.inference.tta))


if __name__ == "__main__":
    main()
