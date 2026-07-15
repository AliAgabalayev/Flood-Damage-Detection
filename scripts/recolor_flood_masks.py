"""Recolor all baked flood_mask.png overlays under web/public/data.

Repaints every non-transparent pixel with the given hex color while
preserving each pixel's alpha (so the 60 % overlay opacity is kept).

Usage:
    python scripts/recolor_flood_masks.py "#e60000"
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise SystemExit(f"expected a 6-digit hex color like #e60000, got {value!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    r, g, b = parse_hex(sys.argv[1])

    root = Path(__file__).resolve().parents[1] / "web" / "public" / "data"
    for png in sorted(root.rglob("flood_mask.png")):
        arr = np.array(Image.open(png).convert("RGBA"))
        opaque = arr[..., 3] > 0
        arr[opaque, 0] = r
        arr[opaque, 1] = g
        arr[opaque, 2] = b
        Image.fromarray(arr, mode="RGBA").save(png)
        print(f"recolored {png.relative_to(root)} ({int(opaque.sum())} px)")


if __name__ == "__main__":
    main()
