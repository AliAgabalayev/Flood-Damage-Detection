# Flood Damage Detection

## Team: Gale

## SAR-Based Flood Damage Detection Using Deep Learning

End-to-end **binary flood-water semantic segmentation** from **Sentinel-1 SAR**
(Sen1Floods11), built with **PyTorch Lightning + segmentation-models-pytorch**.
Input is 3-channel SAR (VV, VH, VV−VH ratio); architectures (DeepLabV3+,
SegFormer-B2) are compared on the same data contract and scorer.

## Project Team

| Team Member | Responsibilities |
|:------------|:----------------|
| **Ali Agabalazade** | Team Lead, DL Engineer, Data Engineering, Documentation |
| **Nigar Rustamova** | Training Optimization, Backend Development, Model Development |
| **Isgandar Panahov** | Frontend Development, User Interface Design, Project Presentation |
| **Jala Suleymanova** | Data Engineering, Data Pipeline Development, Documentation, Presentation |

## Decisions (locked)
S1-only 3-channel · binary · 512×512 · Lightning+smp · MLflow · DVC ·
yaml+pydantic config · local GPU (RTX 3060 6GB → bs=2, accum=4, 16-mixed).
Locked hyperparameters / paths live in [config/default.yaml](config/default.yaml).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Target CLI
```bash
make config                      # validate config/default.yaml
make train                       # train the production baseline
make eval                        # evaluate a checkpoint on a split
make predict INPUT=.. OUTPUT=.. [PROB=..]  # tiled predict -> georeferenced GeoTIFF
make finalists ONLY=<run_name>   # architecture/loss sweep runs
make pretrain-finetune           # weak-label pretrain + hand-label fine-tune
make mlflow-ui                   # MLflow UI (sqlite:///mlflow.db)
make dvc-push / make dvc-pull    # data & model versioning
```

## Project structure

```
Flood-Damage-Detection/
├── config/
│   ├── default.yaml        # production config
│   └── experiments/        # per-study configs (loss study, weak pretrain/fine-tune, ...)
├── data/
│   ├── raw/
│   ├── processed/          # Sen1Floods11 hand- and weak-labeled chips (DVC-tracked)
│   └── splits/             # official + weak train/val/test CSVs
├── docs/                   # study reports (e.g. loss_study.md)
├── models/                 # checkpoints (best.ckpt DVC-tracked; rest git-ignored)
├── notebooks/              # EDA + Colab sweep notebooks
├── scripts/                # data download, split generation, sweep/pretrain-finetune runners
├── src/
│   ├── data/                # dataset, datamodule, preprocessing, transforms
│   ├── models/               # architecture builders
│   ├── training/              # training loop, losses, MLflow logging
│   ├── inference/              # tiling, evaluation, prediction
│   └── utils/                  # config, MLflow helpers
├── web/                    # Next.js frontend (map viewer)
└── Makefile
```

## Workflow

1. **Prepare data** — `bash scripts/download_data.sh` (hand-labeled) and
   `make download-weak-data` (weak-labeled, for pretraining).
2. **Train** — `make train` (production config) or `make finalists` /
   `make pretrain-finetune` for sweeps and transfer learning.
3. **Evaluate** — masked IoU / F1 on a held-out split (`make eval`), test split
   touched once.
4. **Infer** — `make predict INPUT=scene.tif OUTPUT=mask.tif` runs a trained
   checkpoint on a full scene, tiling and stitching into a georeferenced
   flood-mask GeoTIFF. Add `PROB=prob.tif` to also save the per-pixel flood
   probability as a float32 GeoTIFF.

Every run logs to MLflow (params, git SHA, DVC data hash, metrics, checkpoint) —
see `make mlflow-ui`.
