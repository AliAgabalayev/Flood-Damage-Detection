from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from PIL import Image

# Make src/ importable when run from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from data.preprocessing import default_preprocessor, layover_shadow_mask, mask_layover_shadow
from inference.permanent_water import permanent_water_mask, subtract_permanent_water
from inference.predict import _load_model, _run_tile_inference, _select_device
from inference.stitching import binarize, stitch_tiles, write_geotiff
from inference.tiling import generate_tiles
from utils.config import Config, load_config

logger = logging.getLogger(__name__)

# Flood-mask overlay colour: brand orange #c8622a at 60 % opacity.
_FLOOD_RGBA: Tuple[int, int, int, int] = (200, 98, 42, 153)
# Permanent-water overlay colour: blue at 60 % opacity.
_WATER_RGBA: Tuple[int, int, int, int] = (30, 100, 200, 153)
# Layover/shadow overlay colour: grey at 60 % opacity.
_LAYOVER_SHADOW_RGBA: Tuple[int, int, int, int] = (120, 120, 120, 153)


def _require(path: Path, label: str) -> None:
    if not path.exists():
        logger.error("Missing required file (%s): %s", label, path)
        sys.exit(1)


def _mask_to_png(mask: np.ndarray, rgba: Tuple[int, int, int, int], out: Path) -> None:
    # Write a binary mask as an RGBA PNG suitable for a Leaflet imageOverlay
    canvas = np.zeros((*mask.shape, 4), dtype=np.uint8)
    canvas[mask.astype(bool)] = rgba
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGBA").save(out)


def _sar_to_png(scene: Path, vv_clip: Tuple[float, float], out: Path) -> None:
    # Grayscale render of the VV band, clipped/scaled the same way as preprocessing
    with rasterio.open(scene) as src:
        vv = src.read(1).astype(np.float32)
    lo, hi = vv_clip
    scaled = np.nan_to_num(np.clip((vv - lo) / (hi - lo), 0.0, 1.0), nan=0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((scaled * 255).astype(np.uint8), mode="L").save(out)


def _geo_bounds_and_center(
    scene: Path,
) -> Tuple[List[List[float]], List[float]]:
    # Real bounds/center from the scene's own CRS, in the [[south, west], [north, east]]
    # shape the frontend expects — replaces hand-entered placeholder geometry.
    with rasterio.open(scene) as src:
        if src.crs is not None and src.crs.to_epsg() != 4326:
            from rasterio.warp import transform_bounds
            left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        else:
            left, bottom, right, top = src.bounds

    bounds = [[round(bottom, 6), round(left, 6)], [round(top, 6), round(right, 6)]]
    center = [round((bottom + top) / 2, 6), round((left + right) / 2, 6)]
    return bounds, center


def _compute_stats(mask: np.ndarray, scene: Path) -> Tuple[float, float]:
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


def _process(
    loc: Dict[str, Any],
    scene: Path,
    output_dir: Path,
    model: Any,
    device: str,
    cfg: Config,
) -> Dict[str, Any]:
    loc_id = loc["id"]
    logger.info("--- %s (%s)", loc_id, loc["name"])

    out_dir = output_dir / loc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inference
    preprocessor = default_preprocessor(cfg)
    tiles, shape = generate_tiles(
        scene, preprocessor,
        tile_size=cfg.inference.tile_size,
        overlap=cfg.inference.tile_overlap,
    )
    prob_map = stitch_tiles(_run_tile_inference(model, tiles, device), shape)
    flood = binarize(prob_map, threshold=cfg.inference.threshold)

    # Permanent-water (opt-in; off unless config sets inference.permanent_water)
    permanent_water_url = None
    if cfg.inference.permanent_water is not None:
        pw = cfg.inference.permanent_water
        gsw_dir = Path(pw.gsw_dir)
        _require(gsw_dir, "JRC GSW occurrence directory")
        perm = permanent_water_mask(scene, gsw_dir, pw.occurrence_threshold)
        flood = subtract_permanent_water(flood, perm)
        write_geotiff(perm, scene, out_dir / "permanent_water.tif")
        _mask_to_png(perm, _WATER_RGBA, out_dir / "permanent_water.png")
        permanent_water_url = f"/data/{loc_id}/permanent_water.png"

    # Layover/shadow (opt-in; off unless config sets inference.layover_shadow)
    layover_shadow_url = None
    if cfg.inference.layover_shadow is not None:
        ls = cfg.inference.layover_shadow
        dem_dir = Path(ls.dem_dir)
        _require(dem_dir, "Copernicus DEM directory")
        invalid = layover_shadow_mask(
            scene, dem_dir, ls.orbit_pass, ls.near_incidence_deg, ls.far_incidence_deg
        )
        flood = mask_layover_shadow(flood, invalid)
        invalid_u8 = invalid.astype(np.uint8)
        write_geotiff(invalid_u8, scene, out_dir / "layover_shadow.tif")
        _mask_to_png(invalid_u8, _LAYOVER_SHADOW_RGBA, out_dir / "layover_shadow.png")
        layover_shadow_url = f"/data/{loc_id}/layover_shadow.png"

    # Statistics
    area_km2, pct = _compute_stats(flood, scene)

    # GeoTIFF + overlay PNGs
    write_geotiff(flood, scene, out_dir / "flood_mask.tif")
    _mask_to_png(flood, _FLOOD_RGBA, out_dir / "flood_mask.png")
    _sar_to_png(scene, tuple(cfg.data.vv_clip), out_dir / "sar.png")

    bounds, center = _geo_bounds_and_center(scene)

    logger.info("  flooded_area_km2=%.2f  flooded_pct=%.2f%%", area_km2, pct)

    return {
        **loc,
        "center": center,
        "bounds": bounds,
        "flooded_area_km2": area_km2,
        "flooded_pct": pct,
        "mask_url":           f"/data/{loc_id}/flood_mask.png",
        "geotiff_url":        f"/data/{loc_id}/flood_mask.tif",
        "sar_url":            f"/data/{loc_id}/sar.png",
        "permanent_water_url": permanent_water_url,
        "layover_shadow_url": layover_shadow_url,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="G10 — generate flood-prediction artifacts for the 5 demo locations."
    )
    ap.add_argument("--config",      default="config/default.yaml")
    ap.add_argument("--locations",   default="web/public/data/locations.json")
    ap.add_argument("--output-dir",  default="web/public/data")
    ap.add_argument("--scene-dir",   default="data/demo_scenes",
                    help="Directory containing one GeoTIFF per location: <id>.tif")
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    config_path = Path(args.config)
    _require(config_path, "config")
    cfg = load_config(config_path)

    ckpt = Path(cfg.inference.checkpoint)
    _require(ckpt, "model checkpoint")

    locations_path = Path(args.locations)
    _require(locations_path, "locations.json")
    with locations_path.open(encoding="utf-8") as f:
        locations: List[Dict[str, Any]] = json.load(f)

    scene_dir  = Path(args.scene_dir)
    output_dir = Path(args.output_dir)

    have_scene = [loc for loc in locations if (scene_dir / f"{loc['id']}.tif").exists()]
    missing    = [loc for loc in locations if loc not in have_scene]
    if missing:
        logger.warning(
            "No scene yet for %s — leaving those entries unchanged (skip, not a failure).",
            ", ".join(loc["id"] for loc in missing),
        )
    if not have_scene:
        logger.error("No Sentinel-1 scenes found in %s for any location.", scene_dir)
        sys.exit(1)

    device = _select_device(cfg)
    model  = _load_model(cfg, device)
    logger.info("Loaded model on %s", device)

    processed = {
        loc["id"]: _process(loc, scene_dir / f"{loc['id']}.tif", output_dir, model, device, cfg)
        for loc in have_scene
    }
    updated = [processed.get(loc["id"], loc) for loc in locations]

    with locations_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")

if __name__ == "__main__":
    main()
