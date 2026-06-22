# Flood Damage Detection

End-to-end **semantic segmentation** pipeline for detecting flood damage in
aerial/satellite imagery, built with **PyTorch** (U-Net / DeepLab-style models
producing pixel-level flooded/damaged masks).

## Project structure

```
Flood-Damage-Detection/
├── config/                 # YAML configs (paths, hyperparams, model choice)
│   └── default.yaml
├── data/                   # dataset (git-ignored, structure kept via .gitkeep)
│   ├── raw/                # original imagery + masks
│   └── processed/          # tiled / resized / normalized patches
├── notebooks/              # exploration & prototyping notebooks
├── src/                    # source code
│   ├── data/               # dataset, transforms, preprocessing
│   ├── models/             # segmentation architectures (U-Net, etc.)
│   ├── training/           # train loop, losses, metrics
│   ├── inference/          # checkpoint loading + prediction
│   └── utils/              # config loading, visualization
├── scripts/                # CLI entry points (prepare data, train, infer)
├── models/                 # saved checkpoints/weights (git-ignored)
└── tests/                  # unit tests
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow (planned)

1. **Prepare data** — tile/resize raw imagery and masks into `data/processed/`.
2. **Train** — fit a segmentation model using `config/default.yaml`.
3. **Evaluate** — IoU / Dice / pixel accuracy on a held-out split.
4. **Infer** — run a trained checkpoint on new imagery to produce damage masks.
