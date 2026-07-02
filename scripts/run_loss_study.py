"""Run the M1 loss & class-imbalance study sweep.

Reads a matrix file (default: config/experiments/loss_study.yaml), trains one
model per entry with model/data/seed held fixed, and appends the best validation
metrics of each run to a results CSV. Safe to re-run: runs already present in the
CSV are skipped unless --force is given.

Usage
-----
    python scripts/run_loss_study.py                      # full sweep
    python scripts/run_loss_study.py --only dice focal    # subset by name
    python scripts/run_loss_study.py --smoke              # 1-epoch, 2-batch dry run
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.config import Config  # noqa: E402
from training.train import run_training  # noqa: E402

RESULT_FIELDS = [
    "run_name",
    "loss",
    "pos_weight",
    "focal_alpha",
    "best_val_iou",
    "best_val_f1",
    "best_epoch",
    "epochs_run",
    "train_seconds",
]


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


def completed_run_names(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return {row["run_name"] for row in csv.DictReader(f)}


def append_result(csv_path: Path, result: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: result.get(k) for k in RESULT_FIELDS})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="config/experiments/loss_study.yaml")
    ap.add_argument("--out", default="experiments/loss_study/results.csv")
    ap.add_argument("--ckpt-root", default="models/loss_study")
    ap.add_argument("--only", nargs="*", default=None, help="run only these run names")
    ap.add_argument("--force", action="store_true", help="re-run even if already in CSV")
    ap.add_argument("--smoke", action="store_true", help="tiny dry run to validate the pipeline")
    args = ap.parse_args()

    matrix_path = (ROOT / args.matrix) if not Path(args.matrix).is_absolute() else Path(args.matrix)
    matrix = load_matrix(matrix_path)

    base_path = (ROOT / matrix["base_config"])
    with base_path.open("r", encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)
    global_overrides = matrix.get("overrides", {}) or {}

    out_path = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    ckpt_root = (ROOT / args.ckpt_root) if not Path(args.ckpt_root).is_absolute() else Path(args.ckpt_root)

    extra_trainer_kwargs: dict[str, Any] = {}
    if args.smoke:
        # Force a fast, tiny run: CPU-friendly, a couple of batches, one epoch.
        global_overrides = deep_merge(
            global_overrides,
            {"training": {"epochs": 1}, "data": {"num_workers": 0, "augment": False}},
        )
        extra_trainer_kwargs = {
            "limit_train_batches": 2,
            "limit_val_batches": 2,
            "num_sanity_val_steps": 0,
        }
        out_path = out_path.with_name("results_smoke.csv")

    done = set() if args.force else completed_run_names(out_path)
    runs = matrix["runs"]
    if args.only:
        wanted = set(args.only)
        runs = [r for r in runs if r["name"] in wanted]

    print(f"Matrix: {matrix_path}")
    print(f"Results CSV: {out_path}")
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
        append_result(out_path, result)
        print(f"----- {name}: val_iou={result['best_val_iou']:.4f} "
              f"val_f1={result['best_val_f1']:.4f} "
              f"({result['train_seconds']}s) -----")

    print(f"\nDone. Results in {out_path}")


if __name__ == "__main__":
    main()
