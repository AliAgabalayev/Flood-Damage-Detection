# Flood Damage Detection

<img src="docs/assets/logo.png" alt="FlooScan logo" width="220">

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

## Data & model versioning

DVC-tracked data (`data/processed/`, `data/reference/`) and checkpoints
(`models/best.ckpt`, `models/last.ckpt`, `models/weak_pretrain_finetune/`) are
stored in a shared Google Drive remote, not locally.

### Team members — DVC setup

Private remote, dedicated OAuth client (DVC's default shared client gets
rate-limited by Google). Ask the repo owner for Drive access + the client
ID/secret, then:

```bash
make dvc-pull   # fetch tracked data/checkpoints
make dvc-push   # publish tracked data/checkpoints
```

### For graders — full setup, no DVC/OAuth needed

The full DVC store above is private (team-only Drive + OAuth client) because most
of it is dev-only: the 6.8 GiB weak-labeled corpus, every sweep/ablation/LOCO
checkpoint, etc. Grading doesn't need any of that, so a separate **public**
Google Drive artifact carries just the slice needed to run the real pipeline
end-to-end: the production checkpoint (`models/segformer_b4/finetune/best.ckpt`),
the DEM + JRC permanent-water reference data (shipped as the same full,
country-spanning sets `make eval` uses, not trimmed down to just the demo
scenes), the five demo scenes, and the hand-labeled Sen1Floods11 split. Needs
~14 GiB free disk (6 GiB compressed download + 6 GiB extracted, briefly
coexisting).

```bash
make fetch-jury-data   # one command, no Google login, ~6 GiB download
make predict INPUT=data/demo_scenes/baku.tif OUTPUT=/tmp/baku_mask.tif
make eval
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
├── models/                 # checkpoints (best.ckpt, last.ckpt, weak_pretrain_finetune/ DVC-tracked; rest git-ignored)
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
   flood-mask GeoTIFF. By default, the JRC permanent-water mask is fused into
   the predicted flood mask (subtracting permanent water from flooded areas).
   Add `NO_PERMANENT_WATER=1` (or `--no-permanent-water` on the CLI) to disable
   this fusion. Add `PROB=prob.tif` to also save the per-pixel flood probability
   as a float32 GeoTIFF. Add `PERMANENT_WATER=permanent_water.tif` (requires
   `inference.permanent_water` in the config) to also save the JRC permanent-water
   mask as its own georeferenced GeoTIFF.

Every run logs to MLflow (params, git SHA, DVC data hash, metrics, checkpoint) —
see `make mlflow-ui`.
