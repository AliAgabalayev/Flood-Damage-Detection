from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

logger = logging.getLogger(__name__)

# Sen1Floods11 public GCS: representative chips (hand-labeled, flood events)
GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand"

DATASET_CHIPS = {
    # India — Brahmaputra Valley event, 2016-08-12 (Event ID 3, 535 chips)
    # India_136196 chosen: one of the largest chips (1.6 MB), good flood signal
    "india": {
        "filename": "India_136196_S1Hand.tif",
        "scene_date": "2016-08-12",
    },
    # Pakistan — Indus River Floodplain event (confirmed event)
    # Pakistan_1027214 chosen: largest chip in the first-page listing
    "pakistan": {
        "filename": "Pakistan_1027214_S1Hand.tif",
        "scene_date": "2017-09-01",
    },
    # Bolivia — Llanos de Mojos event, 2018-02-15 (Event ID 1, 239 chips)
    # Bolivia_294583 chosen: one of the largest hand-labeled Bolivia chips
    "bolivia": {
        "filename": "Bolivia_294583_S1Hand.tif",
        "scene_date": "2018-02-15",
    },
}

# Microsoft Planetary Computer Sentinel-1 GRD scenes
# Scene IDs confirmed via STAC API to have full AOI coverage
MPC_SCENES = {
    "baku": {
        # S1A acquired 2019-03-15 — exactly matches locations.json scene_date
        # Covers bbox [48.22, 39.29, 51.44, 41.19] ⊇ AOI [49.5, 40.1, 50.3, 40.7]
        "scene_id": "S1A_IW_GRDH_1SDV_20190315T143704_20190315T143729_026350_02F270",
        "scene_date": "2019-03-15",
        "aoi": [49.5, 40.1, 50.3, 40.7],  # [min_lon, min_lat, max_lon, max_lat]
    },
    "sabirabad": {
        # S1A acquired 2021-06-19 — during spring flood season on Kura River
        # Covers bbox [46.63, 39.63, 49.91, 41.52] ⊇ AOI [48.1, 39.7, 48.8, 40.3]
        # NOTE: locations.json had "2010-05-20" which is pre-Sentinel-1 launch (2014).
        # This is the correct real-data acquisition closest to the 2021 spring floods.
        "scene_id": "S1A_IW_GRDH_1SDV_20210619T025229_20210619T025254_038403_048820",
        "scene_date": "2021-06-19",
        "aoi": [48.1, 39.7, 48.8, 40.3],  # [min_lon, min_lat, max_lon, max_lat]
    },
}

# Output pixel spacing for Azerbaijan windowed crops (~100 m, matches inference tile speed)
AZ_PIXEL_DEG = 0.001   # ~111 m N-S, ~83 m E-W at 40°N

def download_gcs_chip(chip_name: str, out_path: Path) -> None:
    url = f"{GCS_BASE}/{chip_name}"
    logger.info("  Downloading %s → %s", url, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)
    logger.info("  → %s  (%.1f MB)", out_path.name, out_path.stat().st_size / 1_048_576)


def _get_mpc_sas_token() -> str:
    import urllib.request as urllib_req
    req = urllib_req.Request(
        "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd",
        headers={"User-Agent": "flood-damage-detection/1.0"},
    )
    with urllib_req.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data["token"]


def _get_mpc_asset_urls(scene_id: str) -> dict[str, str]:
    import urllib.request as urllib_req
    url = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-grd/items/{scene_id}"
    req = urllib_req.Request(url, headers={"User-Agent": "flood-damage-detection/1.0"})
    with urllib_req.urlopen(req, timeout=20) as resp:
        item = json.loads(resp.read())
    return {
        "vv": item["assets"]["vv"]["href"],
        "vh": item["assets"]["vh"]["href"],
    }


def download_mpc_scene_crop(
    scene_id: str,
    aoi: list[float],       
    out_path: Path,
    pixel_deg: float = AZ_PIXEL_DEG,
) -> None:
    logger.info("  Fetching MPC scene %s", scene_id)
    sas_token = _get_mpc_sas_token()
    asset_urls = _get_mpc_asset_urls(scene_id)
    logger.info("  VV: %s", asset_urls["vv"])

    min_lon, min_lat, max_lon, max_lat = aoi

    n_cols = max(512, int((max_lon - min_lon) / pixel_deg))
    n_rows = max(512, int((max_lat - min_lat) / pixel_deg))
    out_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, n_cols, n_rows)
    out_crs = CRS.from_epsg(4326)

    bands_data: list[np.ndarray] = []

    for band_name, href in [("VV", asset_urls["vv"]), ("VH", asset_urls["vh"])]:
        signed_url = href + "?" + sas_token
        # Use rasterio with GDAL VSI curl to do windowed read
        with rasterio.Env(
            GDAL_HTTP_NETRC=False,
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_CURL_VERBOSE=False,
        ):
            vsi_path = f"/vsicurl/{signed_url}"
            try:
                with rasterio.open(vsi_path) as src:
                    logger.info(
                        "  Reading %s band: %s (shape %dx%d, CRS %s)",
                        band_name, vsi_path[:60] + "...",
                        src.height, src.width, src.crs,
                    )
                    dest = np.zeros((n_rows, n_cols), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=dest,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=out_transform,
                        dst_crs=out_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=0.0,
                        dst_nodata=np.nan,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read {band_name} band from MPC scene {scene_id}: {exc}"
                ) from exc

        # Convert linear power to dB (raw S1 GRD values are amplitude; square for power)
        # Sen1Floods11 chips are stored as linear amplitude float32 — do NOT convert to dB
        # The preprocessor clips VV to [-25, 0] and VH to [-35, 0] (dB ranges)
        # MPC GRD values are in linear amplitude [0, ~1]; we must convert to dB
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(dest**2 + 1e-10)  # amplitude → power → dB
        bands_data.append(db.astype(np.float32))
        logger.info("  %s: min=%.2f dB, max=%.2f dB, mean=%.2f dB",
                    band_name, np.nanmin(bands_data[-1]),
                    np.nanmax(bands_data[-1]), np.nanmean(bands_data[-1]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": n_cols,
        "height": n_rows,
        "count": 2,
        "crs": out_crs,
        "transform": out_transform,
        "compress": "lzw",
        "nodata": np.nan,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(bands_data[0], 1)  
        dst.write(bands_data[1], 2) 
    logger.info("  → %s  (%dx%d px, %.1f MB)",
                out_path.name, n_rows, n_cols, out_path.stat().st_size / 1_048_576)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description=(
            "Download real Sentinel-1 scenes for the 5 demo locations.\n"
            "Sen1Floods11 chips: public GCS (no auth).\n"
            "Azerbaijan scenes: Microsoft Planetary Computer (SAS token, no auth)."
        )
    )
    ap.add_argument("--scene-dir", default="data/demo_scenes")
    ap.add_argument("--locations", default="web/public/data/locations.json",
                    help="Path to locations.json (will update scene_date for Sabirabad).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download files that already exist.")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Category A: Sen1Floods11 chips from public GCS 
    logger.info("Downloading Sen1Floods11 chips from public GCS")
    for loc_id, spec in DATASET_CHIPS.items():
        out_path = scene_dir / f"{loc_id}.tif"
        if out_path.exists() and not args.overwrite:
            logger.info("  [SKIP] %s already exists (%.1f MB)", out_path.name,
                        out_path.stat().st_size / 1_048_576)
            continue
        logger.info("--- %s ---", loc_id)
        download_gcs_chip(spec["filename"], out_path)

    # Category B: MPC Sentinel-1 GRD windowed crops 
    logger.info("Downloading Azerbaijan scenes from Microsoft Planetary Computer")
    for loc_id, spec in MPC_SCENES.items():
        out_path = scene_dir / f"{loc_id}.tif"
        if out_path.exists() and not args.overwrite:
            logger.info("  [SKIP] %s already exists (%.1f MB)", out_path.name,
                        out_path.stat().st_size / 1_048_576)
            continue
        logger.info("--- %s ---", loc_id)
        download_mpc_scene_crop(spec["scene_id"], spec["aoi"], out_path)

    # Update locations.json scene_dates
    locations_path = Path(args.locations)
    if locations_path.exists():
        with locations_path.open(encoding="utf-8") as f:
            locations = json.load(f)

        updated = False
        for loc in locations:
            lid = loc["id"]
            if lid in MPC_SCENES:
                real_date = MPC_SCENES[lid]["scene_date"]
                if loc.get("scene_date") != real_date:
                    logger.info(
                        "  Updating %s scene_date: %s → %s",
                        lid, loc.get("scene_date"), real_date,
                    )
                    loc["scene_date"] = real_date
                    updated = True
            elif lid in DATASET_CHIPS:
                real_date = DATASET_CHIPS[lid]["scene_date"]
                if loc.get("scene_date") != real_date:
                    loc["scene_date"] = real_date
                    updated = True

        if updated:
            with locations_path.open("w", encoding="utf-8") as f:
                json.dump(locations, f, indent=2, ensure_ascii=False)
                f.write("\n")
            logger.info("Updated scene dates in %s", locations_path)

    logger.info("Demo scenes ready in %s", scene_dir)
    all_ok = True
    for loc_id in list(DATASET_CHIPS) + list(MPC_SCENES):
        p = scene_dir / f"{loc_id}.tif"
        if p.exists():
            logger.info("  ✓ %-14s  %.1f MB", p.name, p.stat().st_size / 1_048_576)
        else:
            logger.error("  ✗ MISSING: %s", p)
            all_ok = False

    if not all_ok:
        sys.exit(1)
    logger.info("All 5 scenes present. Run: python scripts/generate_demo_artifacts.py")


if __name__ == "__main__":
    main()
