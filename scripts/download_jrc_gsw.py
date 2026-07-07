import argparse
import math
import urllib.error
import urllib.request
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

# JRC Global Surface Water v1.4 (2021)
BASE_URL = (
    "https://storage.googleapis.com/global-surface-water/"
    "downloads2021/occurrence"
)

FILENAME_TEMPLATE = (
    "occurrence_{lon}{lon_dir}_{lat}{lat_dir}v1_4_2021.tif"
)


def chip_bounds(path: Path) -> tuple[float, float, float, float]:
    """Return raster bounds in EPSG:4326."""
    with rasterio.open(path) as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def tiles_for_bounds(bounds: tuple[float, float, float, float]) -> set[tuple[int, int]]:
    """
    Return all 10°×10° JRC tiles intersecting the given bounds.

    Tile names are based on the tile's upper-left (NW) corner.
    """
    minx, miny, maxx, maxy = bounds

    tiles: set[tuple[int, int]] = set()

    lon = math.floor(minx / 10.0) * 10
    lon_end = math.floor(maxx / 10.0) * 10

    while lon <= lon_end:
        lat = math.ceil(maxy / 10.0) * 10
        lat_end = math.ceil(miny / 10.0) * 10

        while lat >= lat_end:
            tiles.add((int(lon), int(lat)))
            lat -= 10

        lon += 10

    return tiles


def tile_filename(lon: int, lat: int) -> str:
    lon_dir = "E" if lon >= 0 else "W"
    lat_dir = "N" if lat >= 0 else "S"

    return FILENAME_TEMPLATE.format(
        lon=abs(lon),
        lon_dir=lon_dir,
        lat=abs(lat),
        lat_dir=lat_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--s1-dirs",
        nargs="+",
        default=[
            "data/processed/sen1floods11/S1Hand",
            "data/processed/sen1floods11_weak/S1Weak",
        ],
    )

    parser.add_argument(
        "--out-dir",
        default="data/reference/jrc_gsw/occurrence",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    all_tiles: set[tuple[int, int]] = set()

    for directory in args.s1_dirs:
        directory = Path(directory)

        if not directory.exists():
            print(f"Warning: {directory} does not exist.")
            continue

        for tif in directory.glob("*.tif"):
            all_tiles |= tiles_for_bounds(chip_bounds(tif))

    filenames = sorted(
        tile_filename(lon, lat)
        for lon, lat in all_tiles
    )

    print(f"\nResolved {len(filenames)} JRC tile(s):\n")

    for f in filenames:
        print(" ", f)

    if args.dry_run:
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    print()

    for name in filenames:

        dest = out_dir / name

        if dest.exists() and dest.stat().st_size > 0:
            print(f"✓ exists : {name}")
            success += 1
            continue

        url = f"{BASE_URL}/{name}"

        print(f"↓ {url}")

        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  downloaded")
            success += 1

        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}: {name}")
            failed += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n-----------------------------------")
    print(f"Downloaded : {success}")
    print(f"Failed     : {failed}")
    print(f"Output dir : {out_dir}")


if __name__ == "__main__":
    main()