from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ClipRange = tuple[float, float]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Base):
    root: str
    s1_subdir: str = "S1Hand"
    label_subdir: str = "LabelHand"
    split_dir: str
    image_size: int = Field(512, gt=0)
    num_workers: int = Field(4, ge=0)
    vv_clip: ClipRange
    vh_clip: ClipRange
    ratio_clip: ClipRange

    @model_validator(mode="after")
    def _check_clips(self) -> "DataConfig":
        for name in ("vv_clip", "vh_clip", "ratio_clip"):
            lo, hi = getattr(self, name)
            if lo >= hi:
                raise ValueError(f"{name}: lo ({lo}) must be < hi ({hi})")
        return self


class ModelConfig(_Base):
    arch: Literal["deeplabv3plus", "deeplabv3", "unet", "unetplusplus", "fpn"]
    backbone: str = "resnet34"
    pretrained: bool = True
    in_channels: int = Field(3, gt=0)
    out_classes: int = Field(1, gt=0)


class TrainingConfig(_Base):
    batch_size: int = Field(2, gt=0)
    accumulate_grad_batches: int = Field(4, gt=0)
    precision: Literal["16-mixed", "bf16-mixed", "32-true", "32"] = "16-mixed"
    epochs: int = Field(50, gt=0)
    lr: float = Field(3e-4, gt=0)
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    loss: Literal["dice", "bce", "dice_bce", "focal"] = "dice_bce"
    pos_weight: Optional[float] = None
    device: Literal["cuda", "cpu"] = "cuda"
    checkpoint_dir: str = "models"


class MLflowConfig(_Base):
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "mlruns"
    experiment: str = "flood-water-seg"


class InferenceConfig(_Base):
    checkpoint: str = "models/best.ckpt"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    tile_size: int = Field(512, gt=0)
    tile_overlap: int = Field(64, ge=0)


class Config(_Base):
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    mlflow: MLflowConfig
    inference: InferenceConfig


def load_config(path: str | Path = "config/default.yaml") -> Config:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml")
    print("Config OK:", cfg.model.arch, "/", cfg.model.backbone)
    print("  effective batch:", cfg.training.batch_size * cfg.training.accumulate_grad_batches)
    print("  clips:", cfg.data.vv_clip, cfg.data.vh_clip, cfg.data.ratio_clip)
