"""Multipart upload regressions with real raster IO and deterministic logits."""

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.transform import from_origin

from api import app as upload_api
from inference.postprocess import postprocess
from utils.config import load_config

GRID = from_origin(49, 41, 0.01, 0.01)


def write_raster(path, array, *, transform=GRID, crs="EPSG:4326", nodata=None):
    if array.ndim == 2:
        array = array[None]
    with rasterio.open(
        path, "w", driver="GTiff", count=array.shape[0], height=array.shape[1],
        width=array.shape[2], dtype=array.dtype, crs=crs, transform=transform, nodata=nodata,
    ) as dataset:
        dataset.write(array)


class WaterModel(torch.nn.Module):
    def forward(self, batch):
        return torch.full_like(batch[:, :1], 8.0)


@pytest.fixture
def service(tmp_path, monkeypatch):
    cfg = load_config(Path(__file__).resolve().parents[1] / "config/default.yaml")
    cfg.inference.tile_size = 8
    cfg.inference.tile_overlap = 2
    cfg.training.device = "cpu"
    cfg.inference.permanent_water.gsw_dir = str(tmp_path / "jrc")
    cfg.inference.layover_shadow.dem_dir = str(tmp_path / "dem")
    (tmp_path / "jrc").mkdir()
    (tmp_path / "dem").mkdir()
    scene = tmp_path / "sar.tif"
    write_raster(scene, np.full((2, 16, 16), -15, dtype=np.float32))
    monkeypatch.setattr(upload_api, "load_config", lambda _: cfg)
    monkeypatch.setattr(upload_api, "_load_model", lambda *_: WaterModel())
    with TestClient(upload_api.app) as client:
        yield SimpleNamespace(client=client, scene=scene, root=tmp_path, cfg=cfg)


def upload(service):
    with service.scene.open("rb") as stream:
        return service.client.post("/predict", files={"file": ("sar.tif", stream, "image/tiff")})


def mask_from(response):
    buffer = io.BytesIO(base64.b64decode(response["mask_png_base64"]))
    with Image.open(buffer) as image:
        return np.asarray(image.getchannel("A")) > 0


def test_upload_applies_both_filters_on_scene_grid(service):
    # A coarse JRC source has occurrence above the threshold in its west half.
    jrc = np.zeros((8, 8), dtype=np.uint8)
    jrc[:, :4] = 80
    write_raster(service.root / "jrc/occ.tif", jrc, transform=from_origin(49, 41, 0.02, 0.02))
    # A steep DEM strip forces terrain flags where occurrence is zero.
    dem = np.zeros((16, 16), dtype=np.int16)
    dem[:, 11] = 8000
    write_raster(service.root / "dem/elevation.tif", dem)
    response = upload(service)
    assert response.status_code == 200, response.text[:300]
    payload = response.json()
    mask = mask_from(payload)
    offline, permanent, invalid = postprocess(
        np.ones((16, 16), dtype=np.uint8), service.scene, service.cfg,
    )
    assert permanent[:, 0].all()
    assert (invalid & ~permanent.astype(bool)).any()
    assert np.array_equal(mask, offline)
    assert mask.any() and not mask.all()
    assert payload["fusion"] == {
        "status": "applied", "raster_pixels": 256, "jrc_covered_pixels": 256,
        "dem_covered_pixels": 256, "joint_covered_pixels": 256, "dem_gap_excluded_pixels": 0,
        "geometry": {"source": "configured_nominal", "orbit_pass": "ASCENDING", "incidence_deg": 37.55},
    }
    assert payload["bounds"] == [[40.84, 49.0], [41.0, 49.16]]
    assert payload["flooded_pct"] == round(100 * mask.sum() / mask.size, 2)
    px_area = (0.01 * 111.32) ** 2 * np.cos(np.deg2rad(40.92))
    assert payload["flooded_area_km2"] == round(mask.sum() * px_area, 2)


def test_upload_distinguishes_jrc_nodata_and_dem_tile_gaps(service):
    jrc = np.zeros((16, 16), dtype=np.uint8)
    jrc[:, :4] = 255  # JRC nodata even when the tile has no nodata tag.
    write_raster(service.root / "jrc/occ.tif", jrc)
    write_raster(service.root / "dem/elevation.tif", np.zeros((16, 10), dtype=np.int16))
    response = upload(service)
    assert response.status_code == 200, response.text[:300]
    payload = response.json()
    mask = mask_from(payload)
    assert mask[:, :9].all() and not mask[:, 9:].any()
    assert payload["fusion"]["status"] == "partial_coverage"
    assert payload["fusion"]["jrc_covered_pixels"] == 192
    assert payload["fusion"]["dem_covered_pixels"] == 160
    assert payload["fusion"]["dem_gap_excluded_pixels"] == 112
    assert payload["fusion"]["joint_covered_pixels"] == 80
    assert payload["flooded_pct"] == 56.25


@pytest.mark.parametrize("missing", ["jrc", "dem", "nodata", "disjoint", "invalid_reference"])
def test_upload_reports_unavailable_references_without_returning_raw_map(service, missing):
    dem = np.zeros((16, 16), dtype=np.int16)
    dem_grid = GRID
    jrc = np.zeros((16, 16), dtype=np.uint8)
    if missing == "nodata":
        jrc[:] = 255
    if missing == "disjoint":
        jrc[:, 8:] = 255
        dem = dem[:, :8]
        dem_grid = from_origin(49.08, 41, 0.01, 0.01)
    if missing != "dem":
        write_raster(service.root / "dem/elevation.tif", dem, transform=dem_grid)
    if missing != "jrc":
        write_raster(service.root / "jrc/occ.tif", jrc, crs=None if missing == "invalid_reference" else "EPSG:4326")
    response = upload(service)
    assert response.status_code == 503
    assert response.json().keys() == {"detail"}
    assert "coverage" in response.json()["detail"] or "aligned" in response.json()["detail"]


@pytest.mark.parametrize("crs,transform", [
    (None, GRID),
    ("EPSG:3857", GRID),
    ("EPSG:4326", GRID.rotation(10) * GRID),
])
def test_upload_rejects_unusable_viewer_grids(service, crs, transform):
    write_raster(service.scene, np.ones((2, 16, 16), dtype=np.float32), crs=crs, transform=transform)
    response = upload(service)
    assert response.status_code == 422
    assert "north-up WGS 84" in response.json()["detail"]
