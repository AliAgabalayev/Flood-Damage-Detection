import argparse
import json
import logging
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import ee

# Make src/ importable and import from generate_demo_artifacts
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from inference.predict import _load_model, _select_device
from utils.config import load_config
from generate_demo_artifacts import process_scene

logger = logging.getLogger(__name__)

def find_newest_scene(aoi: ee.Geometry) -> ee.Image | None:
    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .sort("system:time_start", False)  # Descending
    )
    
    info = coll.limit(10).getInfo()
    if not info or not info.get("features"):
        return None
        
    for feature in info["features"]:
        img = ee.Image(feature["id"])
        coverage = img.geometry().intersection(aoi, 1).area(1).divide(aoi.area(1)).getInfo()
        if coverage >= 0.999:
            return img
            
    return None

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    
    ap = argparse.ArgumentParser(description="Automate Sentinel-1 harvesting and processing.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--locations", default="web/public/data/locations.json")
    ap.add_argument("--output-dir", default="web/public/data")
    args = ap.parse_args()

    logger.info("Initializing Earth Engine (unattended)...")
    ee.Initialize()  # Relies on GOOGLE_APPLICATION_CREDENTIALS or ADC

    config_path = Path(args.config)
    cfg = load_config(config_path)

    device = _select_device(cfg)
    model = _load_model(cfg, device)
    logger.info("Loaded model on %s", device)

    locations_path = Path(args.locations)
    if not locations_path.exists():
        logger.error("Locations file not found: %s", locations_path)
        sys.exit(1)

    with locations_path.open(encoding="utf-8") as f:
        locations = json.load(f)

    output_dir = Path(args.output_dir)
    changed = False

    for loc in locations:
        loc_id = loc["id"]
        bounds = loc["bounds"]
        min_lat, min_lon = bounds[0]
        max_lat, max_lon = bounds[1]
        
        aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
        
        logger.info("Checking %s (%s)", loc_id, loc.get("name", ""))
        img = find_newest_scene(aoi)
        if not img:
            logger.warning("No perfectly covering scene found.")
            continue
            
        scene_id = img.get("system:index").getInfo()
        timestamp = img.get("system:time_start").getInfo()
        date_str = ee.Date(timestamp).format("YYYY-MM-dd").getInfo()
        
        scenes = loc.get("scenes", [])
        if any(s.get("scene_id") == scene_id for s in scenes):
            logger.info("  [SKIP] Scene %s already processed.", scene_id)
            continue
            
        logger.info("Found new scene: %s (%s)", scene_id, date_str)
        
        logger.info("  Downloading scene from Earth Engine...")
        try:
            url = img.getDownloadURL({
                'scale': 10,
                'crs': 'EPSG:4326',
                'region': aoi,
                'format': 'GEO_TIFF'
            })
        except Exception as e:
            logger.error("  Failed to get download URL: %s", e)
            continue
            
        scene_out_dir = output_dir / loc_id / date_str
        scene_out_dir.mkdir(parents=True, exist_ok=True)
        raw_sar_path = scene_out_dir / "sar.tif"
        
        urllib.request.urlretrieve(url, raw_sar_path)
        logger.info("Downloaded %.1f MB", raw_sar_path.stat().st_size / 1_048_576)
        
        # Process the scene using existing pipeline
        logger.info("Generating artifacts...")
        area_km2, pct = process_scene(raw_sar_path, scene_out_dir, model, device, cfg)
        
        new_scene = {
            "scene_id": scene_id,
            "timestamp": timestamp,
            "date": date_str,
            "flooded_area_km2": area_km2,
            "flooded_pct": pct,
            "mask_url": f"/data/{loc_id}/{date_str}/flood_mask.png",
            "sar_url": f"/data/{loc_id}/{date_str}/sar.tif",
            "geotiff_url": f"/data/{loc_id}/{date_str}/flood_mask.tif",
            "permanent_water_url": f"/data/{loc_id}/{date_str}/permanent_water.png",
        }
        
        scenes.append(new_scene)
        scenes.sort(key=lambda s: s.get("timestamp", 0), reverse=True)
        loc["scenes"] = scenes
        changed = True

    if changed:
        logger.info("Writing updated locations metadata...")
        tmp_path = locations_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(locations, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, locations_path)
        logger.info("Update complete.")
    else:
        logger.info("No new scenes found. Up to date.")

if __name__ == "__main__":
    main()
