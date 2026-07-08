from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data.datamodule import FloodDataModule  # noqa: E402
from inference.evaluate import evaluate  # noqa: E402
from training.lightning_module import FloodModel  # noqa: E402
from training.train import run_training  # noqa: E402
from utils.config import Config  # noqa: E402

COUNTRIES = [
    "Ghana", "India", "Mekong", "Nigeria", "Pakistan",
    "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA",
]


def load_raw(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_cfg(raw: dict, split_dir: str, checkpoint_dir: str, experiment: str) -> Config:
    raw = dict(raw)
    raw["data"] = {**raw["data"], "split_dir": split_dir}
    raw["training"] = {**raw["training"], "checkpoint_dir": checkpoint_dir}
    raw["mlflow"] = {**raw["mlflow"], "experiment": experiment}
    return Config.model_validate(raw)


def evaluate_heldout(finetune_cfg: Config, country: str) -> dict:
    heldout_cfg = build_cfg(
        load_raw("config/experiments/weak_finetune.yaml"),
        split_dir=f"data/splits/loco/{country}/heldout",
        checkpoint_dir=finetune_cfg.training.checkpoint_dir,
        experiment=finetune_cfg.mlflow.experiment,
    )
    dm = FloodDataModule(heldout_cfg)
    dm.setup("test")
    model = FloodModel(heldout_cfg)
    ckpt = torch.load(f"{finetune_cfg.training.checkpoint_dir}/best.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return evaluate(model, dm.test_dataloader(), device)


def run_fold(country: str, pretrain_raw: dict, finetune_raw: dict, skip_pretrain: bool) -> dict:
    print(f"\n===== LOCO fold: held-out={country} =====")

    finetune_cfg = build_cfg(
        finetune_raw,
        split_dir=f"data/splits/loco/{country}/train9",
        checkpoint_dir=f"models/loco/{country}/finetune",
        experiment=f"flood-water-seg/loco/{country}",
    )

    init_weights = None
    if not skip_pretrain:
        pretrain_cfg = build_cfg(
            pretrain_raw,
            split_dir=f"data/splits/loco/{country}/weak9",
            checkpoint_dir=f"models/loco/{country}/pretrain",
            experiment=f"flood-water-seg/loco/{country}",
        )
        run_training(pretrain_cfg, run_name=f"pretrain_{country}")
        init_weights = f"{pretrain_cfg.training.checkpoint_dir}/best.ckpt"

    finetune_result = run_training(
        finetune_cfg, run_name=f"finetune_{country}", init_weights=init_weights,
    )

    heldout_metrics = evaluate_heldout(finetune_cfg, country)
    print(f"----- {country} HELD-OUT: iou={heldout_metrics['iou']:.4f}  f1={heldout_metrics['f1']:.4f} -----")

    log_heldout_metrics(
        finetune_cfg.mlflow.tracking_uri, finetune_cfg.mlflow.experiment,
        country, heldout_metrics, finetune_result["best_val_iou"],
    )

    return heldout_metrics


def log_heldout_metrics(tracking_uri, experiment_name, country, heldout_metrics, finetune_val_iou):
    # Uses MlflowClient directly (not the fluent mlflow.* API) so this never
    # touches the process-global "active experiment" state -- mlflow.set_experiment()
    # is sticky across calls in a long-lived process and breaks the next
    # country's run_training() (its own fluent mlflow.start_run(run_id=...)
    # call in train.py checks that state and raises if it doesn't match).
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment is not None else client.create_experiment(experiment_name)
    run = client.create_run(experiment_id=experiment_id, run_name=f"heldout_{country}")
    client.log_metric(run.info.run_id, "heldout_iou", heldout_metrics["iou"])
    client.log_metric(run.info.run_id, "heldout_f1", heldout_metrics["f1"])
    client.log_param(run.info.run_id, "held_out_country", country)
    client.log_param(run.info.run_id, "finetune_val_iou", finetune_val_iou)
    client.set_terminated(run.info.run_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="held-out countries to run (default: all 10)")
    ap.add_argument("--skip-pretrain", action="store_true", help="cheaper variant: hand-label-only training, no weak-pretrain stage")
    args = ap.parse_args()

    pretrain_raw = load_raw("config/experiments/weak_pretrain.yaml")
    finetune_raw = load_raw("config/experiments/weak_finetune.yaml")
    countries = args.only or COUNTRIES

    results = []
    for country in countries:
        metrics = run_fold(country, pretrain_raw, finetune_raw, args.skip_pretrain)
        results.append((country, metrics))

    print("\n===== LOCO summary (held-out performance per country) =====")
    ious, f1s = [], []
    for country, m in results:
        print(f"{country:10s}  iou={m['iou']:.4f}  f1={m['f1']:.4f}")
        ious.append(m["iou"])
        f1s.append(m["f1"])
    if ious:
        print(f"\nMean held-out IoU: {sum(ious) / len(ious):.4f}")
        print(f"Mean held-out F1:  {sum(f1s) / len(f1s):.4f}")


if __name__ == "__main__":
    main()
