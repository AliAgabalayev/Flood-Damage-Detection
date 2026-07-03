"""Run a loss-study sweep matrix.

Reads a matrix file (default: config/experiments/loss_study.yaml), trains one
model per entry with model/data/seed held fixed. Every run is logged to MLflow
under its own experiment (derived from the matrix filename, e.g.
"flood-water-seg/loss_study") — that experiment is the single source of truth,
there is no separate results file. Safe to re-run: runs already FINISHED in
MLflow are skipped unless --force is given.

Usage
-----
    python scripts/run_loss_study.py                      # full sweep
    python scripts/run_loss_study.py --only dice focal    # subset by name
    python scripts/run_loss_study.py --smoke              # 1-epoch, 2-batch dry run
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from training.train import run_training  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.mlflow_utils import finished_run_names, study_experiment  # noqa: E402


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_matrix(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_run_config(
    base_raw: dict[str, Any],
    global_overrides: dict[str, Any],
    run_overrides: dict[str, Any],
    *,
    run_name: str,
    checkpoint_root: Path,
) -> Config:
    raw = deep_merge(base_raw, global_overrides)
    raw = deep_merge(raw, run_overrides)
    # Isolate each run's checkpoints so best.ckpt is not clobbered across runs.
    raw = deep_merge(raw, {"training": {"checkpoint_dir": str(checkpoint_root / run_name)}})
    return Config.model_validate(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="config/experiments/loss_study.yaml")
    ap.add_argument("--ckpt-root", default="models/loss_study")
    ap.add_argument("--only", nargs="*", default=None, help="run only these run names")
    ap.add_argument("--force", action="store_true", help="re-run even if already FINISHED in MLflow")
    ap.add_argument("--smoke", action="store_true", help="tiny dry run to validate the pipeline")
    args = ap.parse_args()

    matrix_path = (ROOT / args.matrix) if not Path(args.matrix).is_absolute() else Path(args.matrix)
    matrix = load_matrix(matrix_path)
    study_name = matrix_path.stem + ("-smoke" if args.smoke else "")

    base_path = ROOT / matrix["base_config"]
    with base_path.open("r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)
    base_mlflow = base_raw.get("mlflow", {})
    tracking_uri = base_mlflow.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = study_experiment(base_mlflow.get("experiment", "flood-water-seg"), study_name)

    global_overrides = deep_merge(matrix.get("overrides", {}) or {}, {"mlflow": {"experiment": experiment_name}})
    ckpt_root = (ROOT / args.ckpt_root) if not Path(args.ckpt_root).is_absolute() else Path(args.ckpt_root)

    extra_trainer_kwargs: dict[str, Any] = {}
    if args.smoke:
        global_overrides = deep_merge(
            global_overrides,
            {"training": {"epochs": 1}, "data": {"num_workers": 0, "augment": False}},
        )
        extra_trainer_kwargs = {
            "limit_train_batches": 2,
            "limit_val_batches": 2,
            "num_sanity_val_steps": 0,
        }

    done = set() if args.force else finished_run_names(tracking_uri, experiment_name)
    runs = matrix["runs"]
    if args.only:
        wanted = set(args.only)
        runs = [r for r in runs if r["name"] in wanted]

    print(f"Matrix: {matrix_path}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"Planned runs: {[r['name'] for r in runs]}")
    if done:
        print(f"Already completed (skipping): {sorted(done)}")

    for spec in runs:
        name = spec["name"]
        if name in done:
            continue
        run_overrides = {k: v for k, v in spec.items() if k != "name"}
        cfg = build_run_config(
            base_raw, global_overrides, run_overrides,
            run_name=name, checkpoint_root=ckpt_root,
        )
        print(f"\n===== Running: {name}  (loss={cfg.training.loss}, "
              f"pos_weight={cfg.training.pos_weight}) =====")
        result = run_training(
            cfg, run_name=name, extra_trainer_kwargs=extra_trainer_kwargs or None
        )
        print(f"----- {name}: val_iou={result['best_val_iou']:.4f} "
              f"val_f1={result['best_val_f1']:.4f} "
              f"({result['train_seconds']}s) -----")

    print(f"\nDone. Results in MLflow experiment: {experiment_name}")


if __name__ == "__main__":
    main()
