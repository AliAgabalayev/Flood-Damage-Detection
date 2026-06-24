from __future__ import annotations

import argparse

from utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Train flood water segmentation baseline")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    eff_batch = cfg.training.batch_size * cfg.training.accumulate_grad_batches
    mode = "SMOKE" if args.smoke else "FULL"

    print(f"[train:{mode}] config OK")
    print(f"  arch       : {cfg.model.arch} / {cfg.model.backbone} (pretrained={cfg.model.pretrained})")
    print(f"  in/out     : {cfg.model.in_channels} ch -> {cfg.model.out_classes} logit")
    print(f"  batch      : {cfg.training.batch_size} x accum {cfg.training.accumulate_grad_batches} = {eff_batch}")
    print(f"  precision  : {cfg.training.precision}")
    print(f"  loss / lr  : {cfg.training.loss} / {cfg.training.lr}")
    print(f"  mlflow     : {cfg.mlflow.tracking_uri} ({cfg.mlflow.experiment})")
    print("[train] harness not implemented (task C3)")


if __name__ == "__main__":
    main()
