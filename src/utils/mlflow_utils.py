from __future__ import annotations

import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import yaml


def git_sha(short: bool = False) -> str:
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        sha = subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ).returncode != 0
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def dvc_data_hash(dvc_file: str | Path) -> str:
    p = Path(dvc_file)
    if not p.exists():
        return "unknown"
    spec = yaml.safe_load(p.read_text())
    outs = spec.get("outs") or []
    return outs[0]["md5"] if outs else "unknown"


def study_experiment(base_experiment: str, study_name: str) -> str:
    return f"{base_experiment}/{study_name}"


def resolve_experiment(tracking_uri: str, experiment_name: str) -> str | None:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp.experiment_id if exp is not None else None


def finished_run_names(tracking_uri: str, experiment_name: str) -> set[str]:
    exp_id = resolve_experiment(tracking_uri, experiment_name)
    if exp_id is None:
        return set()
    df = mlflow.search_runs([exp_id], filter_string="attributes.status = 'FINISHED'")
    if df.empty or "tags.mlflow.runName" not in df.columns:
        return set()
    return set(df["tags.mlflow.runName"].dropna())


def compare_runs(tracking_uri: str, experiment_name: str, metric: str = "best_val_iou", top: int = 50) -> pd.DataFrame:
    exp_id = resolve_experiment(tracking_uri, experiment_name)
    if exp_id is None:
        return pd.DataFrame()
    return mlflow.search_runs(
        [exp_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=top,
    )
