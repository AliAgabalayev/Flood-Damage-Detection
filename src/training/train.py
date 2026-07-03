from __future__ import annotations

import argparse
import time
from typing import Any, Optional

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from data.datamodule import FloodDataModule
from inference.evaluate import evaluate
from training.lightning_module import FloodModel
from utils.config import Config, load_config
from utils.mlflow_utils import dvc_data_hash, git_sha


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
    logger.experiment.set_tag(logger.run_id, "dvc_data_hash", dvc_data_hash(f"{cfg.data.root}.dvc"))
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
    init_weights: Optional[str] = None,
    extra_trainer_kwargs: Optional[dict[str, Any]] = None,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    """Train one model and return the best validation (and optional test) metrics.

    Every run is logged to MLflow: params, git SHA, DVC data hash, per-epoch
    curves, the checkpoint artifact, and (added here) the summary metrics below
    — enough to compare runs without a side CSV. ``evaluate_test`` is opt-in and
    should be used for exactly one run (the chosen production model): the test
    split is touched once project-wide.

    ``resume`` restores full trainer state (optimizer, epoch, LR) to continue an
    interrupted run. ``init_weights`` only seeds the model weights from a prior
    checkpoint before a fresh training run — the right one for fine-tuning a
    pretrained model on a new dataset.
    """
    pl.seed_everything(cfg.seed, workers=True)

    dm = FloodDataModule(cfg)
    model = FloodModel(cfg)
    if init_weights is not None:
        state = torch.load(init_weights, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
    tracker = BestMetricsTracker()
    trainer = build_trainer(
        cfg,
        run_name=run_name,
        tracker=tracker,
        extra_trainer_kwargs=extra_trainer_kwargs,
    )
    run_id = trainer.logger.run_id
    tracking_uri = cfg.mlflow.tracking_uri

    start = time.perf_counter()
    trainer.fit(model, datamodule=dm, ckpt_path=resume)
    elapsed = time.perf_counter() - start

    result: dict[str, Any] = {
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

    if evaluate_test:
        dm.setup("test")
        best_path = trainer.checkpoint_callback.best_model_path
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        device = "cuda" if cfg.training.device == "cuda" and torch.cuda.is_available() else "cpu"
        test_metrics = evaluate(model, dm.test_dataloader(), device)
        result["test_iou"] = test_metrics["iou"]
        result["test_f1"] = test_metrics["f1"]

    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "best_val_iou": result["best_val_iou"],
            "best_val_f1": result["best_val_f1"],
            "best_epoch": float(result["best_epoch"]),
            "epochs_run": float(result["epochs_run"]),
            "train_seconds": result["train_seconds"],
            **({"test_iou": result["test_iou"], "test_f1": result["test_f1"]} if evaluate_test else {}),
        })

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-weights", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = run_training(
        cfg, run_name=args.run_name, resume=args.resume,
        init_weights=args.init_weights, evaluate_test=args.evaluate_test,
    )
    print("Best validation metrics:", result)


if __name__ == "__main__":
    main()
