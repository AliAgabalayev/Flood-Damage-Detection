import argparse
import gc
import os

os.environ.setdefault("GDAL_CACHEMAX", "128")

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from inference.postprocess import postprocess
from inference.stitching import binarize
from inference.tta import predict_prob
from utils.config import Config, load_config


def build_scorer() -> MetricCollection:
    return MetricCollection({"iou": BinaryJaccardIndex(), "f1": BinaryF1Score()})


def masked_select(prob: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    valid = mask.bool()
    return prob[valid], target[valid].int()


def _has_postprocessing(cfg: Config) -> bool:
    return cfg.inference.permanent_water is not None or cfg.inference.layover_shadow is not None


def _fused_batch(prob: Tensor, paths: list[str], cfg: Config) -> Tensor:
    binary = binarize(prob.detach().cpu().numpy(), threshold=cfg.inference.threshold)
    results = []
    for i in range(binary.shape[0]):
        results.append(postprocess(binary[i, 0], paths[i], cfg)[0])
        gc.collect()
    fused = np.stack(results)[:, None, :, :]
    return torch.from_numpy(fused).float().to(prob.device)


def evaluate(model: torch.nn.Module, loader: DataLoader, device: str, tta: bool = False, cfg: Config | None = None) -> dict:
    model.eval().to(device)
    scorer = build_scorer().to(device)
    fused_scorer = build_scorer().to(device) if cfg is not None and _has_postprocessing(cfg) else None

    with torch.no_grad():
        for img, lbl, mask, paths in loader:
            img, lbl, mask = img.to(device), lbl.to(device), mask.to(device)
            prob = predict_prob(model, img, tta=tta)
            target = lbl.unsqueeze(1).float()
            valid = mask.unsqueeze(1).float()
            p, truth = masked_select(prob, target, valid)
            scorer.update(p, truth)

            if fused_scorer is not None:
                fused_prob = _fused_batch(prob, paths, cfg)
                fp, ftruth = masked_select(fused_prob, target, valid)
                fused_scorer.update(fp, ftruth)

    result = {k: float(v) for k, v in scorer.compute().items()}
    if fused_scorer is not None:
        result["fused"] = {k: float(v) for k, v in fused_scorer.compute().items()}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None, help="flip-based test-time augmentation (overrides inference.tta in config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from data.datamodule import FloodDataModule
    from training.lightning_module import FloodModel

    # This is a one-off sequential scoring run, not training -- persistent
    # DataLoader workers just add idle process overhead here (each one imports
    # torch/rasterio fresh), and the fused-metrics path is already CPU/IO-bound
    # per sample. Skip the worker pool entirely.
    cfg.data.num_workers = 0

    dm = FloodDataModule(cfg)
    dm.setup("test" if args.split == "test" else "validate")
    model = FloodModel(cfg)
    ckpt = torch.load(cfg.inference.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    device = "cuda" if cfg.training.device == "cuda" and torch.cuda.is_available() else "cpu"
    loader = dm.test_dataloader() if args.split == "test" else dm.val_dataloader()
    tta = cfg.inference.tta if args.tta is None else args.tta
    print(evaluate(model, loader, device, tta=tta, cfg=cfg))


if __name__ == "__main__":
    main()
