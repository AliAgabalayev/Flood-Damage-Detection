from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import mlflow

from utils.config import Config


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


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def setup_mlflow(cfg: Config) -> str:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment)
    if exp is not None:
        return exp.experiment_id
    artifact_location = Path(cfg.mlflow.artifact_location).resolve().as_uri()
    return mlflow.create_experiment(cfg.mlflow.experiment, artifact_location=artifact_location)


@contextmanager
def mlflow_run(
    cfg: Config,
    run_name: Optional[str] = None,
    extra_params: Optional[dict[str, Any]] = None,
) -> Iterator[mlflow.ActiveRun]:
    setup_mlflow(cfg)
    mlflow.set_experiment(cfg.mlflow.experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten(cfg.model_dump()))
        mlflow.set_tag("git_sha", git_sha())
        if extra_params:
            mlflow.log_params(extra_params)
        yield run


def log_checkpoint(path: str | Path, artifact_path: str = "checkpoints") -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    mlflow.log_artifact(str(p), artifact_path=artifact_path)


def compare_runs(cfg: Config, metric: str = "val_iou", top: int = 20):
    setup_mlflow(cfg)
    exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment)
    if exp is None:
        return None
    order = f"metrics.{metric} DESC"
    return mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=[order], max_results=top)
