from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import rasterio
from PIL import Image

RGBA = Tuple[int, int, int, int]

FLOOD_RGBA: RGBA = (0, 103, 238, 200)      # #0067ee — matches --layer-water
WATER_RGBA: RGBA = (52, 124, 162, 210)     # darker version of the same OSM-water-like blue — matches --layer-permanent
LAYOVER_SHADOW_RGBA: RGBA = (120, 120, 120, 153)


def mask_to_png(mask: np.ndarray, rgba: RGBA) -> Image.Image:
    # Render a binary mask as an RGBA image suitable for a Leaflet imageOverlay
    canvas = np.zeros((*mask.shape, 4), dtype=np.uint8)
    canvas[mask.astype(bool)] = rgba
    return Image.fromarray(canvas, mode="RGBA")


def sar_to_png(scene: Union[Path, str], vv_clip: Tuple[float, float]) -> Image.Image:
    # Grayscale render of the VV band, clipped/scaled the same way as preprocessing
    with rasterio.open(scene) as src:
        vv = src.read(1).astype(np.float32)
    lo, hi = vv_clip
    scaled = np.nan_to_num(np.clip((vv - lo) / (hi - lo), 0.0, 1.0), nan=0.0)
    return Image.fromarray((scaled * 255).astype(np.uint8), mode="L")


def geo_bounds_and_center(scene: Union[Path, str]) -> Tuple[List[List[float]], List[float]]:
    # Real bounds/center from the scene's own CRS, in the [[south, west], [north, east]]
    # shape the frontend expects.
    with rasterio.open(scene) as src:
        if src.crs is not None and src.crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds
            left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        else:
            left, bottom, right, top = src.bounds

    bounds = [[round(bottom, 6), round(left, 6)], [round(top, 6), round(right, 6)]]
    center = [round((bottom + top) / 2, 6), round((left + right) / 2, 6)]
    return bounds, center


def compute_stats(mask: np.ndarray, scene: Union[Path, str]) -> Tuple[float, float]:
    # Return (flooded_area_km2, flooded_pct) for a binary flood mask
    with rasterio.open(scene) as src:
        t = src.transform
        h, w = src.height, src.width
        lat = src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2.0

    px_km2 = (
        abs(t.a) * 111.32 * np.cos(np.radians(lat))
        * abs(t.e) * 111.32
    )
    flood = int(mask.sum())
    return round(flood * px_km2, 2), round(100.0 * flood / (h * w), 2)
