from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import torch
from torch import Tensor

from data.preprocessing import default_preprocessor
from inference.stitching import MaskTile, binarize, stitch_tiles, write_geotiff
from inference.tiling import TileRecord, generate_tiles
from training.lightning_module import FloodModel
from utils.config import Config, load_config

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run tiled inference on a Sentinel-1 SAR scene "
            "and write a binary flood-mask GeoTIFF."
        ),
    )
    ap.add_argument(
        "--input",
        required=True,
        metavar="SCENE",
        help="Path to the input GeoTIFF (Sentinel-1 SAR scene).",
    )
    ap.add_argument(
        "--output",
        required=True,
        metavar="MASK",
        help="Destination path for the binary flood-mask GeoTIFF.",
    )
    ap.add_argument(
        "--config",
        default="config/default.yaml",
        metavar="YAML",
        help="Path to the YAML configuration file (default: config/default.yaml).",
    )
    return ap.parse_args()

def _select_device(cfg: Config) -> str:
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _load_model(cfg: Config, device: str) -> FloodModel:
    checkpoint = cfg.inference.checkpoint
    logger.info("Loading checkpoint: %s", checkpoint)
    model: FloodModel = FloodModel.load_from_checkpoint(checkpoint, cfg=cfg)
    model.eval()
    model.to(device)
    return model

def _run_tile_inference(
    model: FloodModel,
    tiles: List[TileRecord],
    device: str,
) -> List[MaskTile]:
    mask_tiles: List[MaskTile] = []

    with torch.no_grad():
        for record in tiles:
            # Add batch dimension: (C, H, W) → (1, C, H, W)
            batch: Tensor = record.tile.unsqueeze(0).to(device)
            logits: Tensor = model(batch)
            prob: Tensor = torch.sigmoid(logits)
            prob_2d: Tensor = prob.squeeze(0).squeeze(0)
            mask_tiles.append(
                MaskTile(
                    mask=prob_2d,
                    row_start=record.row_start,
                    col_start=record.col_start,
                )
            )

    return mask_tiles

def predict(
    scene_path: Path,
    output_path: Path,
    cfg: Config,
) -> Path:
    if not scene_path.exists():
        raise SystemExit(f"Input scene not found: {scene_path}")

    device = _select_device(cfg)
    logger.info("Using device: %s", device)

    model = _load_model(cfg, device)

    preprocessor = default_preprocessor(cfg)

    logger.info("Tiling scene: %s", scene_path)
    tiles, scene_shape = generate_tiles(
        scene_path,
        preprocessor,
        tile_size=cfg.inference.tile_size,
        overlap=cfg.inference.tile_overlap,
    )
    logger.info(
        "Generated %d tile(s) — scene shape %s (tile_size=%d, overlap=%d)",
        len(tiles),
        scene_shape,
        cfg.inference.tile_size,
        cfg.inference.tile_overlap,
    )

    logger.info("Running inference over %d tile(s)…", len(tiles))
    mask_tiles = _run_tile_inference(model, tiles, device)

    logger.info("Stitching tiles into full probability map…")
    prob_map = stitch_tiles(mask_tiles, scene_shape)

    binary_mask = binarize(prob_map, threshold=cfg.inference.threshold)
    flood_pixels = int(binary_mask.sum())
    logger.info(
        "Threshold %.2f applied — %d flood pixel(s) detected (%.2f%% of scene).",
        cfg.inference.threshold,
        flood_pixels,
        100.0 * flood_pixels / binary_mask.size,
    )

    out = write_geotiff(binary_mask, scene_path, output_path)
    logger.info("Mask written to: %s", out)
    return out

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = load_config(args.config)

    out = predict(
        scene_path=Path(args.input),
        output_path=Path(args.output),
        cfg=cfg,
    )
    print(f"Done. Flood mask saved to: {out}")


if __name__ == "__main__":
    main()
