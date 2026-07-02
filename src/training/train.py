import argparse

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from data.datamodule import FloodDataModule
from training.lightning_module import FloodModel
from utils.config import Config, load_config
from utils.mlflow_utils import git_sha


def build_trainer(cfg: Config) -> pl.Trainer:
    gpu = cfg.training.device == "cuda" and torch.cuda.is_available()
    accel = "gpu" if gpu else "cpu"
    precision = cfg.training.precision if gpu else "32-true"

    logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment,
        tracking_uri=cfg.mlflow.tracking_uri,
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
    return pl.Trainer(
        accelerator=accel,
        devices=1,
        precision=precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        max_epochs=cfg.training.epochs,
        logger=logger,
        callbacks=[ckpt],
        log_every_n_steps=10,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg.seed, workers=True)

    dm = FloodDataModule(cfg)
    model = FloodModel(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
