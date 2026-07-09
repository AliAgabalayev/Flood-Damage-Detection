from __future__ import annotations

import argparse

import ee


def find_and_export(aoi, start, end, description, out_folder, gee_project):
    ee.Initialize(project=gee_project)

    coll = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .sort("system:time_start")
    )

    n = coll.size().getInfo()
    if n == 0:
        raise SystemExit(f"No Sentinel-1 scenes found for {start}..{end} over this AOI.")

    imgs = coll.toList(n)
    for i in range(n):
        img = ee.Image(imgs.get(i))
        coverage = img.geometry().intersection(aoi, 1).area(1).divide(aoi.area(1)).getInfo()
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd HH:mm").getInfo()
        if coverage < 0.999:
            print(f"  skip {date}: only {coverage:.3f} AOI coverage")
            continue

        print(f"  using {date} (full coverage)")
        task = ee.batch.Export.image.toDrive(
            image=img.clip(aoi),
            description=description,
            folder=out_folder,
            region=aoi,
            scale=10,
            crs="EPSG:4326",
            fileFormat="GeoTIFF",
            maxPixels=1e9,
        )
        task.start()
        print(f"  export task started: {task.id} -> Drive folder '{out_folder}/{description}.tif'")
        return task

    raise SystemExit(f"Found {n} scene(s) for {start}..{end} but none fully cover the AOI. Widen the date range or AOI.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull a Sentinel-1 GRD scene (VV+VH) over an AOI via Google Earth Engine.")
    ap.add_argument("--min-lon", type=float, required=True)
    ap.add_argument("--min-lat", type=float, required=True)
    ap.add_argument("--max-lon", type=float, required=True)
    ap.add_argument("--max-lat", type=float, required=True)
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--name", required=True, help="export description / output filename stem")
    ap.add_argument("--drive-folder", default="earthengine")
    ap.add_argument("--gee-project", required=True, help="your GEE-registered Google Cloud project id")
    args = ap.parse_args()

    aoi = ee.Geometry.Rectangle([args.min_lon, args.min_lat, args.max_lon, args.max_lat])
    find_and_export(aoi, args.start_date, args.end_date, args.name, args.drive_folder, args.gee_project)

    print("\nCheck Task progress at https://code.earthengine.google.com/tasks")
    print(f"Once finished, download from Google Drive folder '{args.drive_folder}' and pull it locally, e.g.:")
    print(f"  scp <remote-or-drive-path>/{args.name}.tif data/raw/<location>/")


if __name__ == "__main__":
    main()
