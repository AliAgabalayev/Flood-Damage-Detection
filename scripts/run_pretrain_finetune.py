"""Two-stage transfer learning: pretrain on Sen1Floods11 weak labels, fine-tune on hand labels.

Stage 1 trains on the ~4384-chip weakly-labeled set (S1Weak input, S2IndexLabelWeak
pseudo-labels) to learn from far more examples than the 252 official hand-labeled
chips allow. Stage 2 loads stage 1's WEIGHTS ONLY (fresh optimizer/LR, not a
resumed training state) and continues training on the official hand-labeled train
split, model-selecting on the official val split. Both stages log to the same
MLflow experiment (flood-water-seg/weak_pretrain_finetune) under distinct run
names.

Usage
-----
    python scripts/run_pretrain_finetune.py
    python scripts/run_pretrain_finetune.py --skip-pretrain --pretrain-ckpt models/weak_pretrain_finetune/pretrain/best.ckpt
    python scripts/run_pretrain_finetune.py --evaluate-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from training.train import run_training  # noqa: E402
from utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-config", default="config/experiments/weak_pretrain.yaml")
    ap.add_argument("--finetune-config", default="config/experiments/weak_finetune.yaml")
    ap.add_argument("--skip-pretrain", action="store_true")
    ap.add_argument("--pretrain-ckpt", default=None, help="required with --skip-pretrain")
    ap.add_argument("--evaluate-test", action="store_true")
    args = ap.parse_args()

    if args.skip_pretrain:
        if args.pretrain_ckpt is None:
            raise ValueError("--pretrain-ckpt is required with --skip-pretrain")
        pretrain_ckpt = args.pretrain_ckpt
    else:
        pretrain_cfg = load_config(args.pretrain_config)
        print("===== Stage 1: pretrain on weak labels =====")
        pretrain_result = run_training(pretrain_cfg, run_name="pretrain_resnet50_dice")
        print("pretrain result:", pretrain_result)
        pretrain_ckpt = f"{pretrain_cfg.training.checkpoint_dir}/best.ckpt"

    finetune_cfg = load_config(args.finetune_config)
    print("\n===== Stage 2: fine-tune on hand labels =====")
    finetune_result = run_training(
        finetune_cfg, run_name="finetune_resnet50_dice",
        init_weights=pretrain_ckpt, evaluate_test=args.evaluate_test,
    )
    print("finetune result:", finetune_result)


if __name__ == "__main__":
    main()
