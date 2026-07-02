"""Turn the loss-study results CSV into a ranked Markdown table.

Prints the table to stdout and, if --update-report is passed, splices it into
docs/loss_study.md between the RESULTS markers so the report stays in sync with
the latest sweep.

Usage
-----
    python scripts/make_results_table.py
    python scripts/make_results_table.py --update-report
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

COLUMNS = [
    ("run_name", "Run"),
    ("loss", "Loss"),
    ("pos_weight", "pos_weight"),
    ("focal_alpha", "focal_α"),
    ("best_val_iou", "Val IoU"),
    ("best_val_f1", "Val F1"),
    ("best_epoch", "Best epoch"),
    ("train_seconds", "Time (s)"),
]


def _fmt(key: str, value: str) -> str:
    if value is None or value == "" or value == "None":
        return "—"
    if key in ("best_val_iou", "best_val_f1"):
        try:
            return f"{float(value):.4f}"
        except ValueError:
            return value
    return value


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def sort_key(r: dict[str, str]) -> float:
        try:
            v = float(r.get("best_val_iou", "nan"))
        except ValueError:
            v = float("nan")
        return -1.0 if math.isnan(v) else v

    return sorted(rows, key=sort_key, reverse=True)


def build_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No results yet — run `python scripts/run_loss_study.py` first._"

    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, sep]
    best_iou = rows[0].get("best_val_iou")
    for r in rows:
        cells = [_fmt(key, r.get(key, "")) for key, _ in COLUMNS]
        # Bold the winning run.
        if r.get("best_val_iou") == best_iou:
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
    ap.add_argument("--csv", default="experiments/loss_study/results.csv")
    ap.add_argument("--report", default="docs/loss_study.md")
    ap.add_argument("--update-report", action="store_true")
    args = ap.parse_args()

    csv_path = (ROOT / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    if not csv_path.exists():
        print(f"No results CSV at {csv_path}. Run the sweep first.")
        return

    rows = load_rows(csv_path)
    table = build_table(rows)
    print(table)

    if args.update_report:
        report_path = (ROOT / args.report) if not Path(args.report).is_absolute() else Path(args.report)
        ok = update_report(report_path, table)
        print(f"\n{'Updated' if ok else 'Could not update'} {report_path}")


if __name__ == "__main__":
    main()
