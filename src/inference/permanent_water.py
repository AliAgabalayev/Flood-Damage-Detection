from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from data.reference_raster import aligned_reference


def occurrence_masks(
    scene_path: Path | str,
    gsw_dir: Path | str,
    occurrence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(scene_path) as scene:
        occurrence = aligned_reference(scene, gsw_dir, missing_value=255)
    valid = (occurrence >= 0) & (occurrence <= 100)
    return ((occurrence >= occurrence_threshold) & valid).astype(np.uint8), valid


def permanent_water_mask(
    scene_path: Path | str,
    gsw_dir: Path | str,
    occurrence_threshold: float,
) -> np.ndarray:
    mask, _ = occurrence_masks(scene_path, gsw_dir, occurrence_threshold)
    return mask


def subtract_permanent_water(flood_mask: np.ndarray, permanent_mask: np.ndarray) -> np.ndarray:
    if flood_mask.shape != permanent_mask.shape:
        raise ValueError(f"flood_mask shape {flood_mask.shape} != permanent_mask shape {permanent_mask.shape}")
    return (flood_mask.astype(bool) & ~permanent_mask.astype(bool)).astype(np.uint8)

