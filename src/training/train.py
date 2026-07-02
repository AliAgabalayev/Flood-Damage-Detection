from __future__ import annotations

import argparse
import time
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from data.datamodule import FloodDataModule
from training.lightning_module import FloodModel
from utils.config import Config, load_config
from utils.mlflow_utils import git_sha


class BestMetricsTracker(pl.Callback):
    """Record the (val_iou, val_f1) pair from the epoch with the best val_iou.

    ``ModelCheckpoint`` already keeps the best *checkpoint*, but it does not
    expose the companion F1 at that epoch. This callback captures both so the
    loss study can report the metrics that belong to the selected model.
    """

    def __init__(self, monitor: str = "val_iou") -> None:
        self.monitor = monitor
        self.best_iou: float = float("-inf")
        self.best_f1: float = float("nan")
        self.best_epoch: int = -1

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Runs after the LightningModule's on_validation_epoch_end has logged
        # val_iou/val_f1 into callback_metrics (same hook ModelCheckpoint uses).
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        iou = metrics.get("val_iou")
        if iou is None:
            return
        iou = float(iou)
        if iou > self.best_iou:
            f1 = metrics.get("val_f1")
            self.best_iou = iou
            self.best_f1 = float(f1) if f1 is not None else float("nan")
            self.best_epoch = int(trainer.current_epoch)


def build_trainer(
    cfg: Config,
    *,
    run_name: Optional[str] = None,
    tracker: Optional[BestMetricsTracker] = None,
    extra_trainer_kwargs: Optional[dict[str, Any]] = None,
) -> pl.Trainer:
    gpu = cfg.training.device == "cuda" and torch.cuda.is_available()
    accel = "gpu" if gpu else "cpu"
    precision = cfg.training.precision if gpu else "32-true"

    logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=run_name,
        log_model=True,
    )
    logger.experiment.set_tag(logger.run_id, "git_sha", git_sha())
    logger.log_hyperparams(cfg.model_dump())

    ckpt = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="best",
        monitor="val_iou",
        mode="max",
        save_last=True,
    )
    callbacks: list[pl.Callback] = [ckpt]
    if tracker is not None:
        callbacks.append(tracker)
    if cfg.training.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_iou",
                mode="max",
                patience=cfg.training.early_stopping_patience,
            )
        )

    kwargs: dict[str, Any] = dict(
        accelerator=accel,
        devices=1,
        precision=precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        max_epochs=cfg.training.epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    if extra_trainer_kwargs:
        kwargs.update(extra_trainer_kwargs)
    return pl.Trainer(**kwargs)


def run_training(
    cfg: Config,
    *,
    run_name: Optional[str] = None,
    resume: Optional[str] = None,
    extra_trainer_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train one model and return the best validation metrics.

    Returns a dict with best ``val_iou``/``val_f1``, the epoch they occurred at,
    and wall-clock training time — enough for the loss study to tabulate results
    without re-querying MLflow.
    """
    pl.seed_everything(cfg.seed, workers=True)

    dm = FloodDataModule(cfg)
    model = FloodModel(cfg)
    tracker = BestMetricsTracker()
    trainer = build_trainer(
        cfg,
        run_name=run_name,
        tracker=tracker,
        extra_trainer_kwargs=extra_trainer_kwargs,
    )

    start = time.perf_counter()
    trainer.fit(model, datamodule=dm, ckpt_path=resume)
    elapsed = time.perf_counter() - start

    return {
        "run_name": run_name,
        "loss": cfg.training.loss,
        "pos_weight": cfg.training.pos_weight,
        "focal_alpha": cfg.training.focal_alpha if cfg.training.loss == "focal" else None,
        "best_val_iou": tracker.best_iou if tracker.best_epoch >= 0 else float("nan"),
        "best_val_f1": tracker.best_f1,
        "best_epoch": tracker.best_epoch,
        "epochs_run": int(trainer.current_epoch),
        "train_seconds": round(elapsed, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = run_training(cfg, run_name=args.run_name, resume=args.resume)
    print("Best validation metrics:", result)


if __name__ == "__main__":
    main()
