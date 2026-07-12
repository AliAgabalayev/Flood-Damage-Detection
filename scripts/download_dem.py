import argparse
import math
import urllib.error
import urllib.request
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

GLO30_BASE_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
GLO90_BASE_URL = "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"


def chip_bounds(path: Path) -> tuple[float, float, float, float]:
    """Return raster bounds in EPSG:4326."""
    with rasterio.open(path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def tiles_for_bounds(bounds: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    """Return all 1deg x 1deg Copernicus DEM tiles intersecting the given bounds."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Copernicus DEM tiles covering a set of Sentinel-1 scenes. "
        "Tries GLO-30 first; falls back to GLO-90 where GLO-30 is unavailable (e.g. Azerbaijan)."
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
        default="data/reference/copernicus_dem",
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
    glo90 = 0
    failed = 0

    print()

    for lon, lat in sorted_tiles:
        dest = out_dir / f"{tile_stem(lon, lat, '10')}.tif"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"✓ exists (GLO-30): {dest.name}")
            glo30 += 1
            continue

        dest90 = out_dir / f"{tile_stem(lon, lat, '30')}.tif"
        if dest90.exists() and dest90.stat().st_size > 0:
            print(f"✓ exists (GLO-90): {dest90.name}")
            glo90 += 1
            continue

        url = tile_url(lon, lat, "10")
        print(f"↓ {url}")
        try:
            urllib.request.urlretrieve(url, dest)
            print("  downloaded (GLO-30)")
            glo30 += 1
            continue
        except urllib.error.HTTPError as e:
            print(f"  GLO-30 unavailable (HTTP {e.code}); trying GLO-90")

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
    print(f"GLO-90     : {glo90}")
    print(f"Failed     : {failed}")
    print(f"Output dir : {out_dir}")


if __name__ == "__main__":
    main()
