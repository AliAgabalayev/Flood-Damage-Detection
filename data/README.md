# Data

This folder holds the dataset. Its contents are git-ignored; only the folder
structure is tracked (via `.gitkeep`).

```
data/
├── raw/         # original imagery + segmentation masks as downloaded
└── processed/   # tiled / resized / normalized image+mask patches
```

## How to populate

1. Place the original flood imagery and pixel-level masks under `data/raw/`.
2. Run the preparation step to generate model-ready patches into `data/processed/`.

> Expected layout for `raw/`: paired image and mask files (e.g. `images/` and
> `masks/`), where each mask encodes flooded/damaged regions per pixel.
