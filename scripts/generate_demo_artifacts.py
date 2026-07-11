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

from data.preprocessing import default_preprocessor
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


def process_scene(
    scene: Path,
    out_dir: Path,
    model: Any,
    device: str,
    cfg: Config,
) -> Tuple[float, float]:
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

    # Permanent-water
    if cfg.inference.permanent_water is not None:
        pw = cfg.inference.permanent_water
        gsw_dir = Path(pw.gsw_dir)
        _require(gsw_dir, "JRC GSW occurrence directory")
        perm = permanent_water_mask(scene, gsw_dir, pw.occurrence_threshold)
        flood = subtract_permanent_water(flood, perm)
    else:
        perm = np.zeros(shape, dtype=np.uint8)

    # Statistics
    area_km2, pct = _compute_stats(flood, scene)

    # GeoTIFFs
    write_geotiff(flood, scene, out_dir / "flood_mask.tif")
    write_geotiff(perm,  scene, out_dir / "permanent_water.tif")

    # PNGs
    _mask_to_png(flood, _FLOOD_RGBA, out_dir / "flood_mask.png")
    _mask_to_png(perm,  _WATER_RGBA, out_dir / "permanent_water.png")

    logger.info("  flooded_area_km2=%.2f  flooded_pct=%.2f%%", area_km2, pct)

    return area_km2, pct


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

    missing = [
        str(scene_dir / f"{loc['id']}.tif")
        for loc in locations
        if not (scene_dir / f"{loc['id']}.tif").exists()
    ]
    if missing:
        logger.error(
            "Missing Sentinel-1 scene file(s):\n%s",
            "\n".join(f"  {p}" for p in missing),
        )
        sys.exit(1)

    device = _select_device(cfg)
    model  = _load_model(cfg, device)
    logger.info("Loaded model on %s", device)

    updated = []
    for loc in locations:
        scene = scene_dir / f"{loc['id']}.tif"
        loc_id = loc["id"]
        logger.info("--- %s (%s)", loc_id, loc.get("name", ""))
        
        # Use existing date if available, otherwise 'latest'
        date_str = loc.get("scenes", [{}])[0].get("date", "latest")
        scene_out_dir = output_dir / loc_id / date_str
        
        area_km2, pct = process_scene(scene, scene_out_dir, model, device, cfg)
        
        new_scene = {
            "scene_id": loc.get("scenes", [{}])[0].get("scene_id", f"local_{date_str}"),
            "date": date_str,
            "flooded_area_km2": area_km2,
            "flooded_pct": pct,
            "mask_url":    f"/data/{loc_id}/{date_str}/flood_mask.png",
            "sar_url":     None,
            "geotiff_url": f"/data/{loc_id}/{date_str}/flood_mask.tif",
            "permanent_water_url": f"/data/{loc_id}/{date_str}/permanent_water.png",
        }
        
        if loc.get("scenes"):
            loc["scenes"][0].update(new_scene)
        else:
            loc["scenes"] = [new_scene]
        updated.append(loc)

    import tempfile
    import os
    # Atomic write to avoid corruption
    tmp_path = locations_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, locations_path)

if __name__ == "__main__":
    main()
