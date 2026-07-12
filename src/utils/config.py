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
    augment: bool = True

    @model_validator(mode="after")
    def _check_clips(self) -> "DataConfig":
        for name in ("vv_clip", "vh_clip", "ratio_clip"):
            lo, hi = getattr(self, name)
            if lo >= hi:
                raise ValueError(f"{name}: lo ({lo}) must be < hi ({hi})")
        return self


class ModelConfig(_Base):
    arch: Literal["deeplabv3plus", "deeplabv3", "unet", "unetplusplus", "fpn", "segformer_b2"]
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
    loss: Literal[
        "dice", "bce", "dice_bce", "focal",
        "tversky", "focal_tversky", "lovasz", "dice_focal",
    ] = "dice_bce"
    pos_weight: Optional[float] = None
    # dice_bce / dice_focal are weighted sums: bce_weight/dice_weight scale the
    # BCE and Dice terms. Defaults (1.0, 1.0) reproduce a plain unweighted sum.
    bce_weight: float = Field(1.0, ge=0.0)
    dice_weight: float = Field(1.0, ge=0.0)
    # Focal loss imbalance controls (used by "focal" and "dice_focal").
    # alpha is the positive-class prior weight; gamma is the focusing strength.
    focal_alpha: Optional[float] = Field(0.25, ge=0.0, le=1.0)
    focal_gamma: float = Field(2.0, ge=0.0)
    # Tversky controls (used by "tversky" and "focal_tversky"). alpha penalises
    # false positives, beta false negatives; beta > alpha boosts flood recall.
    # tversky_gamma > 1 turns it into Focal-Tversky (focuses on hard regions).
    tversky_alpha: float = Field(0.3, ge=0.0, le=1.0)
    tversky_beta: float = Field(0.7, ge=0.0, le=1.0)
    tversky_gamma: float = Field(1.0, ge=0.0)
    # Optional early stopping on val_iou (patience in epochs). None disables it.
    early_stopping_patience: Optional[int] = Field(None, gt=0)
    device: Literal["cuda", "cpu"] = "cuda"
    checkpoint_dir: str = "models"


class MLflowConfig(_Base):
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "mlruns"
    experiment: str = "flood-water-seg"


class PermanentWaterConfig(_Base):
    gsw_dir: str = "data/reference/jrc_gsw/occurrence"
    occurrence_threshold: float = Field(50.0, ge=0.0, le=100.0)


class LayoverShadowConfig(_Base):
    dem_dir: str = "data/reference/dem"
    orbit_pass: Literal["ASCENDING", "DESCENDING"] = "ASCENDING"
    near_incidence_deg: float = Field(29.1, gt=0.0, lt=90.0)
    far_incidence_deg: float = Field(46.0, gt=0.0, lt=90.0)

    @model_validator(mode="after")
    def _check_incidence_range(self) -> "LayoverShadowConfig":
        if self.near_incidence_deg >= self.far_incidence_deg:
            raise ValueError(
                f"near_incidence_deg ({self.near_incidence_deg}) must be < "
                f"far_incidence_deg ({self.far_incidence_deg})"
            )
        return self


class InferenceConfig(_Base):
    checkpoint: str = "models/best.ckpt"
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    tile_size: int = Field(512, gt=0)
    tile_overlap: int = Field(64, ge=0)
    permanent_water: Optional[PermanentWaterConfig] = None
    layover_shadow: Optional[LayoverShadowConfig] = None


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
