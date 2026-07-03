"""Turn a loss-study MLflow experiment into a ranked Markdown table.

Prints the table to stdout and, if --update-report is passed, splices it into
docs/loss_study.md between the RESULTS markers so the report stays in sync with
the latest sweep. MLflow (not a CSV) is the source of truth: the experiment
name is derived from the matrix filename, the same way run_loss_study.py
derives it when logging runs.

Usage
-----
    python scripts/make_results_table.py
    python scripts/make_results_table.py --update-report
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.mlflow_utils import compare_runs, study_experiment  # noqa: E402

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

COLUMNS = [
    ("tags.mlflow.runName", "Run"),
    ("params.training/loss", "Loss"),
    ("params.training/pos_weight", "pos_weight"),
    ("params.training/focal_alpha", "focal_α"),
    ("metrics.best_val_iou", "Val IoU"),
    ("metrics.best_val_f1", "Val F1"),
    ("metrics.best_epoch", "Best epoch"),
    ("metrics.train_seconds", "Time (s)"),
]


def _fmt(key: str, value: object) -> str:
    if value is None or value in ("", "None") or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if key in ("metrics.best_val_iou", "metrics.best_val_f1"):
        return f"{float(value):.4f}"
    if key == "metrics.best_epoch":
        return str(int(float(value)))
    return str(value)


def build_table(df) -> str:
    if df.empty:
        return "_No results yet — run `python scripts/run_loss_study.py` first._"

    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, sep]
    best_iou = df.iloc[0].get("metrics.best_val_iou")
    for _, row in df.iterrows():
        cells = [_fmt(key, row.get(key)) for key, _ in COLUMNS]
        if row.get("params.training/loss") != "focal":
            cells[3] = "—"
        if row.get("metrics.best_val_iou") == best_iou:
            cells[0] = f"**{cells[0]}**"
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_report(report_path: Path, table: str) -> bool:
    if not report_path.exists():
        return False
    text = report_path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    pre = text.split(START)[0]
    post = text.split(END)[1]
    new = f"{pre}{START}\n\n{table}\n\n{END}{post}"
    report_path.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="config/experiments/loss_study.yaml")
    ap.add_argument("--report", default="docs/loss_study.md")
    ap.add_argument("--update-report", action="store_true")
    args = ap.parse_args()

    matrix_path = (ROOT / args.matrix) if not Path(args.matrix).is_absolute() else Path(args.matrix)
    matrix = yaml.safe_load(matrix_path.read_text())
    base_raw = yaml.safe_load((ROOT / matrix["base_config"]).read_text())
    base_mlflow = base_raw.get("mlflow", {})
    tracking_uri = base_mlflow.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = study_experiment(base_mlflow.get("experiment", "flood-water-seg"), matrix_path.stem)

    df = compare_runs(tracking_uri, experiment_name, metric="best_val_iou")
    table = build_table(df)
    print(table)
    print(f"\n(from MLflow experiment: {experiment_name})")

    if args.update_report:
        if df.empty:
            print("\nRefusing to overwrite the report: no MLflow results found for this experiment.")
            return
        report_path = (ROOT / args.report) if not Path(args.report).is_absolute() else Path(args.report)
        ok = update_report(report_path, table)
        print(f"\n{'Updated' if ok else 'Could not update'} {report_path}")


if __name__ == "__main__":
    main()
