"""Recolor all baked mask overlays of one kind under web/public/data.

Repaints every non-transparent pixel with the given hex color. By
default the existing alpha is preserved; pass an explicit alpha
(0-255) as a third argument to overwrite it too, e.g. to match a new
RGBA constant in src/inference/rendering.py.

Usage:
    python scripts/recolor_flood_masks.py "#0080ff" flood_mask.png
    python scripts/recolor_flood_masks.py "#4682b4" permanent_water.png 130
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise SystemExit(f"expected a 6-digit hex color like #0080ff, got {value!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    r, g, b = parse_hex(sys.argv[1])
    filename = sys.argv[2]
    alpha = int(sys.argv[3]) if len(sys.argv) == 4 else None

    root = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
    for png in sorted(root.rglob(filename)):
        arr = np.array(Image.open(png).convert("RGBA"))
        opaque = arr[..., 3] > 0
        arr[opaque, 0] = r
        arr[opaque, 1] = g
        arr[opaque, 2] = b
        if alpha is not None:
            arr[opaque, 3] = alpha
        Image.fromarray(arr, mode="RGBA").save(png)
        print(f"recolored {png.relative_to(root)} ({int(opaque.sum())} px)")


if __name__ == "__main__":
    main()
