from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.preprocessing import Preprocessor, default_preprocessor



def _load_image(path: Union[Path, str]) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with rasterio.open(path) as src:
        image: np.ndarray = src.read().astype(np.float32)   # (C, H, W)

    return image


def _load_label(path: Union[Path, str]) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label not found: {path}")

    with rasterio.open(path) as src:
        label: np.ndarray = src.read(1).astype(np.int16)    # (H, W)

    return label


def _create_valid_mask(label: np.ndarray) -> np.ndarray:
    return (label != -1).astype(bool)


def validate_sample(
    image: Tensor,
    label: Tensor,
    valid_mask: Tensor,
    *,
    source: str = "",
) -> None:
    hint = f" [source: {source}]" if source else ""
    errors: list[str] = []

    # 1. Rank 
    if image.ndim != 3:
        errors.append(
            f"image must be 3-D (C, H, W) but has ndim={image.ndim}, "
            f"shape={tuple(image.shape)}"
        )
    if label.ndim != 2:
        errors.append(
            f"label must be 2-D (H, W) but has ndim={label.ndim}, "
            f"shape={tuple(label.shape)}"
        )
    if valid_mask.ndim != 2:
        errors.append(
            f"valid_mask must be 2-D (H, W) but has ndim={valid_mask.ndim}, "
            f"shape={tuple(valid_mask.shape)}"
        )

    if errors:   # bail before spatial checks — shapes would be misleading
        raise ValueError(
            f"Dataset contract violation{hint}:\n"
            + "\n".join(f"  \u2022 {e}" for e in errors)
        )

    # 2. Spatial alignment 
    img_hw = (image.shape[1], image.shape[2])
    lbl_hw = tuple(label.shape)
    msk_hw = tuple(valid_mask.shape)

    if img_hw != lbl_hw:
        errors.append(
            f"image spatial size {img_hw} != label spatial size {lbl_hw}"
        )
    if msk_hw != lbl_hw:
        errors.append(
            f"valid_mask spatial size {msk_hw} != label spatial size {lbl_hw}"
        )

    # 3. Finiteness of image 
    if not torch.isfinite(image).all():
        n_bad = int((~torch.isfinite(image)).sum().item())
        errors.append(
            f"image contains {n_bad} non-finite value(s) "
            f"(NaN or Inf) out of {image.numel()} total elements"
        )

    if errors:
        raise ValueError(
            f"Dataset contract violation{hint}:\n"
            + "\n".join(f"  \u2022 {e}" for e in errors)
        )



class Sen1FloodDataset(Dataset):
    def __init__(
        self,
        image_paths: List[Union[Path, str]],
        label_paths: List[Union[Path, str]],
        config: object,
        transforms: Optional[Callable] = None,
        preprocessor: Optional[Preprocessor] = None,
    ) -> None:
        if len(image_paths) != len(label_paths):
            raise ValueError(
                f"image_paths and label_paths must have the same length, "
                f"got {len(image_paths)} vs {len(label_paths)}."
            )

        self.image_paths: List[Path] = [Path(p) for p in image_paths]
        self.label_paths: List[Path] = [Path(p) for p in label_paths]
        self.config: object = config
        self.transforms: Optional[Callable] = transforms
        self._preprocessor: Preprocessor = (
            preprocessor if preprocessor is not None
            else default_preprocessor(config)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, str]:
        # I/O
        raw_image: np.ndarray = _load_image(self.image_paths[index])
        raw_label: np.ndarray = _load_label(self.label_paths[index])

        # Preprocessing
        channels: np.ndarray = self._preprocessor(raw_image)

        # Valid mask + label clean-up
        img_finite: np.ndarray = np.isfinite(raw_image).all(axis=0)
        valid_mask_np: np.ndarray = _create_valid_mask(raw_label) & img_finite
        clean_label: np.ndarray = np.where(
            valid_mask_np, raw_label, 0
        ).astype(np.int64)

        # Convert to tensors
        image_tensor: Tensor = torch.from_numpy(channels).float()     # (C, H, W)
        label_tensor: Tensor = torch.from_numpy(clean_label).long()   # (H, W)
        valid_mask:   Tensor = torch.from_numpy(valid_mask_np).bool() # (H, W)

        # Optional transforms
        if self.transforms is not None:
            image_tensor, label_tensor, valid_mask = self.transforms(
                image_tensor, label_tensor, valid_mask
            )

        # Contract validation (post-transform)
        validate_sample(
            image_tensor,
            label_tensor,
            valid_mask,
            source=str(self.image_paths[index]),
        )

        return image_tensor.float(), label_tensor.long(), valid_mask.bool(), str(self.image_paths[index])
