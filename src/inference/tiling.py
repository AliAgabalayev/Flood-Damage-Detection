from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from data.dataset import _load_image
from data.preprocessing import Preprocessor


class TileRecord(NamedTuple):

    tile: Tensor
    row_start: int
    col_start: int

def _compute_starts(total: int, tile_size: int, overlap: int) -> List[int]:
    #Return the top-left offsets for a 1-D tiling with overlap

    stride = tile_size - overlap
    starts: List[int] = [s for s in range(0, total, stride) if s + tile_size < total]

    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    last = total - tile_size
    if last > 0 and starts[-1] != last:
        starts.append(last)

    return starts


def _pad_tile(crop: np.ndarray, tile_size: int) -> np.ndarray:

    #Padding is applied only on the right/bottom edges so that the
    #top-left origin of each tile aligns perfectly with the scene grid (zero padding logici)

    _, h, w = crop.shape
    pad_h = tile_size - h
    pad_w = tile_size - w
    if pad_h == 0 and pad_w == 0:
        return crop

    return np.pad(crop, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


def generate_tiles(
    scene_path: Union[Path, str],
    preprocessor: Preprocessor,
    *,
    tile_size: int = 512,
    overlap: int = 64,
) -> Tuple[List[TileRecord], Tuple[int, int]]:
    #Reads a SAR scene, splits it into overlapping 512×512 tiles, 
    #pads edge tiles, preprocesses each tile using the existing Preprocessor
    #returns the processed tiles together with their positions for stitching

    if not (0 <= overlap < tile_size):
        raise ValueError(
            f"overlap must be in [0, tile_size) but got overlap={overlap}, "
            f"tile_size={tile_size}."
        )
    raw_scene: np.ndarray = _load_image(scene_path)   
    _, scene_H, scene_W = raw_scene.shape

    row_starts = _compute_starts(scene_H, tile_size, overlap)
    col_starts = _compute_starts(scene_W, tile_size, overlap)

    tiles: List[TileRecord] = []

    for r in row_starts:
        for c in col_starts:
            row_end = min(r + tile_size, scene_H)
            col_end = min(c + tile_size, scene_W)

            crop = raw_scene[:, r:row_end, c:col_end]   

            processed: np.ndarray = preprocessor(crop)
            processed = _pad_tile(processed, tile_size)

            tile_tensor: Tensor = torch.from_numpy(processed).float()

            tiles.append(TileRecord(tile=tile_tensor, row_start=r, col_start=c))

    return tiles, (scene_H, scene_W)
