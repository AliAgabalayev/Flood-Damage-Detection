from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling

def permanent_water_mask(
        scene_path: Path | str,
        gsw_dir: Path | str,
        occurrence_threshold: float,
):
    scene_path = Path(scene_path)
    gsw_dir = Path(gsw_dir)
    with rasterio.open(scene_path) as scene:
        scene_crs = scene.crs
        scene_transform = scene.transform
        scene_shape = (scene.height, scene.width)
        scene_bounds = scene.bounds

    intersecting_areas = []
    for tif in gsw_dir.glob("*.tif"):
        with rasterio.open(tif) as src:
            b=src.bounds
            if not (b.right < scene_bounds.left or b.left > scene_bounds.right or b.top < scene_bounds.bottom or b.bottom > scene_bounds.top):
                intersecting_areas.append(tif)

    if not intersecting_areas:
        return np.zeros(scene_shape, dtype=np.uint8)

    sources = []

    for t in intersecting_areas:
        src = rasterio.open(t)
        sources.append(src)

    mosaic, mosaic_transformers = merge(sources)

    for src in sources:
        src.close()

    gsw_crs = rasterio.open(intersecting_areas[0]).crs

    dest = np.zeros(scene_shape, dtype=np.float32)
    reproject(
        source=mosaic[0],
        destination=dest,
        src_transform=mosaic_transformers,
        src_crs=gsw_crs,
        dst_transform=scene_transform,
        dst_crs=scene_crs,
        resampling=Resampling.bilinear,
    )

    return (dest >= occurrence_threshold).astype(np.uint8)


def subtract_permanent_water(flood_mask: np.ndarray, permanent_mask: np.ndarray) -> np.ndarray:
    if flood_mask.shape != permanent_mask.shape:
        raise ValueError(f"flood_mask shape {flood_mask.shape} != permanent_mask shape {permanent_mask.shape}")
    return (flood_mask.astype(bool) & ~permanent_mask.astype(bool)).astype(np.uint8)

