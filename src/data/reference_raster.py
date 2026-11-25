"""Read only the reference mosaic window needed for a scene."""

from contextlib import ExitStack
from math import ceil, floor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, bounds, from_bounds


def aligned_reference(scene, directory: Path | str, *, missing_value=None) -> np.ndarray:
    """Warp an overlapping single-CRS mosaic to the scene grid, with NaN gaps."""
    destination = np.full(scene.shape, np.nan, dtype=np.float32)
    with ExitStack() as stack:
        sources = []
        for tile in sorted(Path(directory).glob("*.tif")):
            src = stack.enter_context(rasterio.open(tile))
            if src.crs is None:
                raise ValueError(f"Reference tile has no CRS: {tile.name}")
            footprint = transform_bounds(scene.crs, src.crs, *scene.bounds)
            if not disjoint_bounds(src.bounds, footprint):
                sources.append(src)
        if not sources:
            return destination

        first = sources[0]
        if any(src.crs != first.crs for src in sources):
            raise ValueError("Overlapping reference tiles must share a CRS.")

        # Snap to the mosaic grid and retain a source-pixel halo for bilinear warping.
        footprint = transform_bounds(scene.crs, first.crs, *scene.bounds)
        window = from_bounds(*footprint, first.transform)
        left, top = floor(window.col_off) - 2, floor(window.row_off) - 2
        right = ceil(window.col_off + window.width) + 2
        bottom = ceil(window.row_off + window.height) + 2
        extent = bounds(Window(left, top, right - left, bottom - top), first.transform)
        mosaic, transform = merge(sources, bounds=extent, indexes=[1], dtype="float32", nodata=np.nan)
        if missing_value is not None:
            mosaic[mosaic == missing_value] = np.nan
        reproject(
            mosaic[0], destination,
            src_transform=transform, src_crs=first.crs, src_nodata=np.nan,
            dst_transform=scene.transform, dst_crs=scene.crs, dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination
