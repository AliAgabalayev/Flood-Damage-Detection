from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling


ChannelBuilder = Callable[[np.ndarray], np.ndarray]

ClipRange = Tuple[float, float]

LOOK_AZIMUTH_DEG = {"ASCENDING": 90.0, "DESCENDING": 270.0}

def build_vv(image: np.ndarray) -> np.ndarray:
    return image[0].astype(np.float32)


def build_vh(image: np.ndarray) -> np.ndarray:
    return image[1].astype(np.float32)


def build_ratio(image: np.ndarray) -> np.ndarray:
    vv = image[0].astype(np.float32)
    vh = image[1].astype(np.float32)
    return vv - vh



class Preprocessor:

    def __init__(self, channel_specs: List[Tuple[ChannelBuilder, ClipRange]]) -> None:
        if not channel_specs:
            raise ValueError(
                "Preprocessor requires at least one channel spec, but got an empty list."
            )
        self._specs: List[Tuple[ChannelBuilder, ClipRange]] = channel_specs



    @property
    def num_channels(self) -> int:
        # Number of output channels produced by this preprocessor
        return len(self._specs)

    def __call__(self, raw_image: np.ndarray) -> np.ndarray:
        planes: List[np.ndarray] = []
        for builder, clip_range in self._specs:
            plane = builder(raw_image)                          # (H, W)
            plane = self._clip_and_scale(plane, clip_range)    # (H, W) in [0, 1]
            planes.append(plane)

        return np.stack(planes, axis=0).astype(np.float32)     # (N, H, W)

   
    @staticmethod
    def _clip_and_scale(plane: np.ndarray, clip_range: ClipRange) -> np.ndarray:
        lo, hi = clip_range
        clipped = np.clip(plane, lo, hi)
        span = hi - lo if (hi - lo) != 0.0 else 1.0
        scaled = np.nan_to_num((clipped - lo) / span, nan=0.0, posinf=1.0, neginf=0.0)
        return scaled.astype(np.float32)




def default_preprocessor(config: object) -> Preprocessor:
    #Build the canonical ``[VV, VH, Ratio]`` preprocessor from a config object.
    data_cfg = config.data
    specs: List[Tuple[ChannelBuilder, ClipRange]] = [
        (build_vv,    tuple(data_cfg.vv_clip)),     # channel 0
        (build_vh,    tuple(data_cfg.vh_clip)),     # channel 1
        (build_ratio, tuple(data_cfg.ratio_clip)),  # channel 2
    ]
    return Preprocessor(specs)


def _mosaic_dem(dem_dir: Path, scene_crs, scene_transform, scene_shape: Tuple[int, int], scene_bounds) -> Tuple[np.ndarray, np.ndarray]:
    intersecting = []
    for tif in dem_dir.glob("*.tif"):
        with rasterio.open(tif) as src:
            b = src.bounds
            if not (b.right < scene_bounds.left or b.left > scene_bounds.right or b.top < scene_bounds.bottom or b.bottom > scene_bounds.top):
                intersecting.append(tif)

    if not intersecting:
        raise FileNotFoundError(f"No DEM tiles in {dem_dir} intersect the scene bounds {tuple(scene_bounds)}.")

    sources = [rasterio.open(t) for t in intersecting]
    nodata = sources[0].nodata
    mosaic, mosaic_transform = merge(sources, nodata=nodata) if nodata is not None else merge(sources)
    dem_crs = sources[0].crs
    for src in sources:
        src.close()

    fill_value = nodata if nodata is not None else 0.0
    dest = np.full(scene_shape, fill_value, dtype=np.float32)
    reproject(
        source=mosaic[0],
        destination=dest,
        src_transform=mosaic_transform,
        src_crs=dem_crs,
        dst_transform=scene_transform,
        dst_crs=scene_crs,
        src_nodata=nodata,
        dst_nodata=nodata,
        resampling=Resampling.bilinear,
    )
    valid = dest != nodata if nodata is not None else np.ones(scene_shape, dtype=bool)
    return dest, valid


def _pixel_spacing_m(transform, bounds) -> Tuple[float, float]:
    mid_lat_rad = np.radians((bounds.top + bounds.bottom) / 2.0)
    dx = abs(transform.a) * 111_320.0 * np.cos(mid_lat_rad)
    dy = abs(transform.e) * 110_540.0
    return dx, dy


def _slope_aspect(dem: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(dem, dy, dx)
    slope = np.arctan(np.hypot(grad_x, grad_y))
    aspect = np.arctan2(-grad_x, grad_y) % (2.0 * np.pi)
    return slope, aspect


def _dilate(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :]  |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:]  |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    return dilated


def layover_shadow_mask(
    scene_path: Path | str,
    dem_dir: Path | str,
    orbit_pass: str,
    near_incidence_deg: float,
    far_incidence_deg: float,
) -> np.ndarray:
    scene_path = Path(scene_path)
    dem_dir = Path(dem_dir)

    with rasterio.open(scene_path) as scene:
        scene_crs = scene.crs
        scene_transform = scene.transform
        scene_shape = (scene.height, scene.width)
        scene_bounds = scene.bounds

    dem, dem_valid = _mosaic_dem(dem_dir, scene_crs, scene_transform, scene_shape, scene_bounds)
    dx, dy = _pixel_spacing_m(scene_transform, scene_bounds)
    slope, aspect = _slope_aspect(dem, dx, dy)

    incidence = np.radians((near_incidence_deg + far_incidence_deg) / 2.0)
    look_azimuth = np.radians(LOOK_AZIMUTH_DEG[orbit_pass])

    range_slope = np.arctan(np.tan(slope) * np.cos(look_azimuth - aspect))
    local_incidence = incidence - range_slope

    layover = local_incidence <= 0.0
    shadow = local_incidence >= (np.pi / 2.0)
    dem_missing = _dilate(~dem_valid)
    return layover | shadow | dem_missing


def mask_layover_shadow(flood_mask: np.ndarray, invalid_mask: np.ndarray) -> np.ndarray:
    if flood_mask.shape != invalid_mask.shape:
        raise ValueError(f"flood_mask shape {flood_mask.shape} != invalid_mask shape {invalid_mask.shape}")
    return (flood_mask.astype(bool) & ~invalid_mask.astype(bool)).astype(np.uint8)
