import argparse
import gzip
import math
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

GLO30_BASE_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
GLO90_BASE_URL = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"
SKADI_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"
SRTM_NODATA = -32768.0


def chip_bounds(path: Path) -> tuple[float, float, float, float]:
    """Return raster bounds in EPSG:4326."""
    with rasterio.open(path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def tiles_for_bounds(bounds: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    """Return all 1deg x 1deg DEM tiles intersecting the given bounds."""
    minx, miny, maxx, maxy = bounds
    tiles: set[tuple[int, int]] = set()
    for lon in range(math.floor(minx), math.floor(maxx) + 1):
        for lat in range(math.floor(miny), math.floor(maxy) + 1):
            tiles.add((lon, lat))
    return tiles


def tile_stem(lon: int, lat: int, resolution_code: str) -> str:
    lon_dir = "E" if lon >= 0 else "W"
    lat_dir = "N" if lat >= 0 else "S"
    return f"Copernicus_DSM_COG_{resolution_code}_{lat_dir}{abs(lat):02d}_00_{lon_dir}{abs(lon):03d}_00_DEM"


def tile_url(lon: int, lat: int, resolution_code: str) -> str:
    base = GLO30_BASE_URL if resolution_code == "10" else GLO90_BASE_URL
    stem = tile_stem(lon, lat, resolution_code)
    return f"{base}/{stem}/{stem}.tif"


def skadi_tile_name(lon: int, lat: int) -> str:
    lon_dir = "E" if lon >= 0 else "W"
    lat_dir = "N" if lat >= 0 else "S"
    return f"{lat_dir}{abs(lat):02d}{lon_dir}{abs(lon):03d}"


def skadi_url(lon: int, lat: int) -> str:
    name = skadi_tile_name(lon, lat)
    return f"{SKADI_BASE_URL}/{name[:3]}/{name}.hgt.gz"


def download_srtm_tile(lon: int, lat: int, dest_tif: Path, tmp_dir: Path) -> None:
    """Download an SRTM1 (Skadi format) tile and convert it to a GeoTIFF with
    the SRTM nodata sentinel set, so downstream reprojection/merging can
    correctly exclude void pixels instead of treating them as real elevation."""
    name = skadi_tile_name(lon, lat)
    gz_path = tmp_dir / f"{name}.hgt.gz"
    hgt_path = tmp_dir / f"{name}.hgt"

    urllib.request.urlretrieve(skadi_url(lon, lat), gz_path)
    try:
        with gzip.open(gz_path, "rb") as f_in, open(hgt_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        with rasterio.open(hgt_path) as src:
            profile = src.profile.copy()
            data = src.read(1)

        profile.update(driver="GTiff", nodata=SRTM_NODATA, compress="deflate")
        with rasterio.open(dest_tif, "w", **profile) as dst:
            dst.write(data, 1)
    finally:
        gz_path.unlink(missing_ok=True)
        hgt_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DEM tiles covering a set of Sentinel-1 scenes. "
        "Tries Copernicus GLO-30 first, falls back to SRTM1 (30m, via the public "
        "Skadi mirror) where GLO-30 is unavailable (e.g. Azerbaijan), then to "
        "Copernicus GLO-90 (90m) as a last resort."
    )

    parser.add_argument(
        "--scene-dirs",
        nargs="+",
        default=[
            "data/processed/sen1floods11/S1Hand",
            "data/processed/sen1floods11_weak/S1Weak",
            "data/demo_scenes",
        ],
    )

    parser.add_argument(
        "--out-dir",
        default="data/reference/dem",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    all_tiles: set[tuple[int, int]] = set()

    for directory in args.scene_dirs:
        directory = Path(directory)

        if not directory.exists():
            print(f"Warning: {directory} does not exist.")
            continue

        for tif in directory.glob("*.tif"):
            all_tiles |= tiles_for_bounds(chip_bounds(tif))

    sorted_tiles = sorted(all_tiles)

    print(f"\nResolved {len(sorted_tiles)} DEM tile(s):\n")
    for lon, lat in sorted_tiles:
        print(" ", tile_stem(lon, lat, "10"))

    if args.dry_run:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    glo30 = 0
    srtm = 0
    glo90 = 0
    failed = 0

    print()

    for lon, lat in sorted_tiles:
        dest30 = out_dir / f"{tile_stem(lon, lat, '10')}.tif"
        dest_srtm = out_dir / f"SRTM1_{skadi_tile_name(lon, lat)}.tif"
        dest90 = out_dir / f"{tile_stem(lon, lat, '30')}.tif"

        if dest30.exists() and dest30.stat().st_size > 0:
            print(f"✓ exists (GLO-30): {dest30.name}")
            glo30 += 1
            continue
        if dest_srtm.exists() and dest_srtm.stat().st_size > 0:
            print(f"✓ exists (SRTM1): {dest_srtm.name}")
            srtm += 1
            continue
        if dest90.exists() and dest90.stat().st_size > 0:
            print(f"✓ exists (GLO-90): {dest90.name}")
            glo90 += 1
            continue

        url = tile_url(lon, lat, "10")
        print(f"↓ {url}")
        try:
            urllib.request.urlretrieve(url, dest30)
            print("  downloaded (GLO-30)")
            glo30 += 1
            continue
        except urllib.error.HTTPError as e:
            print(f"  GLO-30 unavailable (HTTP {e.code}); trying SRTM1")

        print(f"↓ {skadi_url(lon, lat)}")
        try:
            download_srtm_tile(lon, lat, dest_srtm, out_dir)
            print("  downloaded (SRTM1)")
            srtm += 1
            continue
        except urllib.error.HTTPError as e:
            print(f"  SRTM1 unavailable (HTTP {e.code}); trying GLO-90")
        except Exception as e:
            print(f"  SRTM1 error: {e}; trying GLO-90")

        url = tile_url(lon, lat, "30")
        print(f"↓ {url}")
        try:
            urllib.request.urlretrieve(url, dest90)
            print("  downloaded (GLO-90)")
            glo90 += 1
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}: {tile_stem(lon, lat, '30')}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n-----------------------------------")
    print(f"GLO-30     : {glo30}")
    print(f"SRTM1      : {srtm}")
    print(f"GLO-90     : {glo90}")
    print(f"Failed     : {failed}")
    print(f"Output dir : {out_dir}")


if __name__ == "__main__":
    main()
