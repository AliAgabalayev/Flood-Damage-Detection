from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.errors import RasterioError

from data.preprocessing import layover_shadow_mask, mask_layover_shadow, terrain_masks
from inference.permanent_water import occurrence_masks, permanent_water_mask, subtract_permanent_water
from utils.config import Config


def postprocess(
    mask: np.ndarray,
    scene_path: Path | str,
    cfg: Config,
    no_permanent_water: bool = False,
    no_layover_shadow: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    permanent = None
    if cfg.inference.permanent_water is not None and not no_permanent_water:
        pw = cfg.inference.permanent_water
        permanent = permanent_water_mask(scene_path, pw.gsw_dir, pw.occurrence_threshold)

    invalid = None
    if cfg.inference.layover_shadow is not None and not no_layover_shadow:
        ls = cfg.inference.layover_shadow
        invalid = layover_shadow_mask(
            scene_path, ls.dem_dir, ls.orbit_pass, ls.near_incidence_deg, ls.far_incidence_deg,
        )

    return apply_exclusions(mask, permanent, invalid), permanent, invalid


def apply_exclusions(
    mask: np.ndarray, permanent: np.ndarray | None, invalid: np.ndarray | None,
) -> np.ndarray:
    if permanent is not None:
        mask = subtract_permanent_water(mask, permanent)
    if invalid is not None:
        mask = mask_layover_shadow(mask, invalid)
    return mask


class ReferenceCoverageError(ValueError):
    """The configured references cannot support a combined upload product."""


def upload_exclusions(
    scene_path: Path, cfg: Config,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the offline exclusion masks and report their coverage on the upload."""
    pw, ls = cfg.inference.permanent_water, cfg.inference.layover_shadow
    if pw is None or ls is None:
        raise ReferenceCoverageError("Both JRC occurrence and DEM must be configured.")
    try:
        permanent, jrc_valid = occurrence_masks(scene_path, pw.gsw_dir, pw.occurrence_threshold)
        invalid, dem_valid, dem_missing = terrain_masks(
            scene_path, ls.dem_dir, ls.orbit_pass, ls.near_incidence_deg, ls.far_incidence_deg,
        )
    except (RasterioError, OSError, ValueError) as exc:
        raise ReferenceCoverageError("JRC occurrence or DEM could not be aligned to this scene.") from exc

    joint = int(np.count_nonzero(jrc_valid & ~dem_missing))
    if not joint:
        raise ReferenceCoverageError("No overlapping usable JRC and DEM coverage on this scene.")

    return permanent, invalid, {
        "status": "applied" if joint == jrc_valid.size else "partial_coverage",
        "raster_pixels": jrc_valid.size,
        "jrc_covered_pixels": int(jrc_valid.sum()),
        "dem_covered_pixels": int(dem_valid.sum()),
        "joint_covered_pixels": joint,
        "dem_gap_excluded_pixels": int(dem_missing.sum()),
        "geometry": {
            "source": "configured_nominal",
            "orbit_pass": ls.orbit_pass,
            "incidence_deg": (ls.near_incidence_deg + ls.far_incidence_deg) / 2,
        },
    }
