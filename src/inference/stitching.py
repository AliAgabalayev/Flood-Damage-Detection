from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Tuple, Union

import numpy as np
import rasterio

ArrayLike = Union[np.ndarray, "object"]  # numpy array or torch tensor


class MaskTile(NamedTuple):
    # one model output mask plus where it sits in the scene

    mask: ArrayLike       # (H, W) probabilities in [0, 1]
    row_start: int
    col_start: int


def _to_2d_numpy(mask: ArrayLike) -> np.ndarray:
    # turn a tensor/array into a plain 2-D (H, W) array

    if hasattr(mask, "detach"):          # torch tensor
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask, dtype=np.float32)

    while array.ndim > 2 and array.shape[0] == 1:  # drop extra front axes
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"mask must reduce to 2-D (H, W) but got shape {tuple(array.shape)}."
        )
    return array


def _feather_window(h: int, w: int) -> np.ndarray:
    # weights that are small at the edges and big in the middle,
    # so overlapping tiles blend without a hard seam

    rows = np.minimum(np.arange(1, h + 1), np.arange(h, 0, -1))
    cols = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1))
    return np.minimum.outer(rows, cols).astype(np.float32)


def stitch_tiles(
    mask_tiles: List[MaskTile],
    scene_shape: Tuple[int, int],
) -> np.ndarray:
    # put all tiles back together into one full probability map

    if not mask_tiles:
        raise ValueError("mask_tiles is empty; nothing to stitch.")

    scene_H, scene_W = scene_shape
    accum = np.zeros((scene_H, scene_W), dtype=np.float32)
    weights = np.zeros((scene_H, scene_W), dtype=np.float32)

    for record in mask_tiles:
        mask = _to_2d_numpy(record.mask)
        r, c = record.row_start, record.col_start

        # cut off any padding so we stay inside the scene
        h = min(mask.shape[0], scene_H - r)
        w = min(mask.shape[1], scene_W - c)
        if h <= 0 or w <= 0:
            continue
        mask = mask[:h, :w]
        window = _feather_window(h, w)

        # add up masks and weights, then divide -> weighted average
        accum[r:r + h, c:c + w] += mask * window
        weights[r:r + h, c:c + w] += window

    covered = weights > 0
    accum[covered] /= weights[covered]
    return accum


def binarize(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    # 0/1 mask: 1 where probability is above the threshold

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1] but got {threshold}.")
    return (prob_map >= threshold).astype(np.uint8)


def write_geotiff(
    mask: np.ndarray,
    scene_path: Union[Path, str],
    out_path: Union[Path, str],
    dtype: str = "uint8",
) -> Path:
    # save the mask as a GeoTIFF, reusing the scene's CRS + transform

    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2-D (H, W) but got shape {tuple(mask.shape)}."
        )

    scene_path = Path(scene_path)
    with rasterio.open(scene_path) as src:  # copy the geo info from the scene
        profile = src.profile.copy()
        scene_hw = (src.height, src.width)

    if mask.shape != scene_hw:
        raise ValueError(
            f"mask shape {mask.shape} does not match scene shape {scene_hw}; "
            f"cannot preserve georeferencing."
        )

    profile.update(count=1, dtype=dtype, compress="lzw", nodata=None)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(dtype), 1)

    return out_path


def verify_overlay(mask_path: Union[Path, str], scene_path: Union[Path, str]) -> bool:
    # check the saved mask lines up with the scene (same CRS, transform, size)

    with rasterio.open(mask_path) as m, rasterio.open(scene_path) as s:
        problems: List[str] = []
        if m.crs != s.crs:
            problems.append(f"CRS mismatch: {m.crs} != {s.crs}")
        if m.transform != s.transform:
            problems.append(f"transform mismatch: {m.transform} != {s.transform}")
        if (m.height, m.width) != (s.height, s.width):
            problems.append(
                f"size mismatch: {(m.height, m.width)} != {(s.height, s.width)}"
            )

    if problems:
        raise ValueError("mask does not overlay scene:\n  " + "\n  ".join(problems))
    return True
