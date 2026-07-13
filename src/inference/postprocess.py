from __future__ import annotations

from pathlib import Path

import numpy as np

from data.preprocessing import layover_shadow_mask, mask_layover_shadow
from inference.permanent_water import permanent_water_mask, subtract_permanent_water
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
        mask = subtract_permanent_water(mask, permanent)

    invalid = None
    if cfg.inference.layover_shadow is not None and not no_layover_shadow:
        ls = cfg.inference.layover_shadow
        invalid = layover_shadow_mask(
            scene_path, ls.dem_dir, ls.orbit_pass, ls.near_incidence_deg, ls.far_incidence_deg,
        )
        mask = mask_layover_shadow(mask, invalid)

    return mask, permanent, invalid
