from __future__ import annotations

import base64
import io
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import rasterio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from data.preprocessing import default_preprocessor
from inference.postprocess import postprocess
from inference.predict import _load_model, _run_tile_inference, _select_device
from inference.rendering import FLOOD_RGBA, compute_stats, geo_bounds_and_center, mask_to_png, sar_to_png
from inference.stitching import binarize, stitch_tiles
from inference.tiling import generate_tiles
from utils.config import load_config

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 300 * 1024 * 1024  # generous over the ~110 MB demo scenes
CONFIG_PATH = "config/default.yaml"

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config(CONFIG_PATH)
    device = _select_device(cfg)
    _state["cfg"] = cfg
    _state["device"] = device
    _state["model"] = _load_model(cfg, device)
    logger.info("Model loaded on %s — ready to serve /predict", device)
    yield
    _state.clear()


app = FastAPI(title="Flood Damage Detection — Inference API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "File must be a GeoTIFF (.tif or .tiff).")

    contents = file.file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File too large — max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    cfg = _state["cfg"]
    device = _state["device"]
    model = _state["model"]

    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        tmp.write(contents)
        tmp.flush()
        scene_path = Path(tmp.name)

        try:
            with rasterio.open(scene_path) as src:
                band_count = src.count
        except rasterio.errors.RasterioIOError:
            raise HTTPException(400, "Could not read this file as a GeoTIFF.")

        if band_count < 2:
            raise HTTPException(
                400,
                f"Expected a Sentinel-1 scene with VV and VH bands (>= 2), got {band_count}.",
            )

        try:
            preprocessor = default_preprocessor(cfg)
            tiles, scene_shape = generate_tiles(
                scene_path, preprocessor,
                tile_size=cfg.inference.tile_size,
                overlap=cfg.inference.tile_overlap,
            )
            mask_tiles = _run_tile_inference(model, tiles, device, tta=cfg.inference.tta)
            prob_map = stitch_tiles(mask_tiles, scene_shape)
            flood = binarize(prob_map, threshold=cfg.inference.threshold)

            flood, _, _ = postprocess(
                flood, scene_path, cfg,
                no_permanent_water=True,
                no_layover_shadow=True,
            )

            area_km2, pct = compute_stats(flood, scene_path)
            bounds, center = geo_bounds_and_center(scene_path)
            mask_png = mask_to_png(flood, FLOOD_RGBA)
            sar_png = sar_to_png(scene_path, tuple(cfg.data.vv_clip))
        except HTTPException:
            raise
        except Exception:
            logger.exception("Prediction failed for upload %r", file.filename)
            raise HTTPException(500, "Prediction failed — the scene may be malformed or unsupported.")

    return {
        "flooded_area_km2": area_km2,
        "flooded_pct": pct,
        "bounds": bounds,
        "center": center,
        "mask_png_base64": _image_to_base64(mask_png),
        "sar_png_base64": _image_to_base64(sar_png),
    }
