# Flood Damage Detection

## Team: Gale

## SAR-Based Flood Damage Detection Using Deep Learning

End-to-end **binary flood-water semantic segmentation** from **Sentinel-1 SAR**
(Sen1Floods11), built with **PyTorch Lightning + segmentation-models-pytorch**.
Input is 3-channel SAR (VV, VH, VV−VH ratio); architectures (U-Net, DeepLabV3+,
YOLO-seg) are compared on the same data contract and scorer.

## Project Team

| Team Member | Responsibilities |
|:------------|:----------------|
| **Ali Agabalazade** | Machine Learning Engineering, Model Training, Data Engineering, Documentation |
| **Nigar Rustamova** | Training Optimization, Backend Development, Model Development |
| **Isgandar Panahov** | Frontend Development, User Interface Design, Project Presentation |
| **Jala Suleymanova** | Data Engineering, Data Pipeline Development, Documentation, Presentation |

## Decisions (locked)
S1-only 3-channel · binary · 512×512 · Lightning+smp · MLflow · DVC ·
yaml+pydantic config · local GPU (RTX 3060 6GB → bs=2, accum=4, 16-mixed).
See [DATA_CONTRACT.md](DATA_CONTRACT.md).

## Where to start (team)
This repo currently holds the **plan and the frozen interfaces**, not the
implementation yet — code is built per the task board.

Planning docs (task board, decision log, data contract) are kept outside the
repo in `~/Downloads/flood-docs/`. Locked hyperparameters / paths live in
[config/default.yaml](config/default.yaml).

```bash
source .venv/bin/activate && pip install -r requirements.txt
export PYTHONPATH=.
```

## Target CLI (to be implemented under the tasks)
```bash
python -m src.training.train     --config config/default.yaml
python -m src.inference.evaluate --config config/default.yaml --checkpoint models/best.ckpt
python -m src.inference.predict  --config config/default.yaml --input scene_S1.tif --output flood_mask.tif
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
```

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
