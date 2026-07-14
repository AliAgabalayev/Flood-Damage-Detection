# Flood Damage Detection

<img src="docs/assets/logo.png" alt="FlooScan logo" width="220">

## Team: Gale

## SAR-Based Flood Damage Detection Using Deep Learning

End-to-end **binary flood-water semantic segmentation** from **Sentinel-1 SAR**
(Sen1Floods11), built with **PyTorch Lightning + segmentation-models-pytorch**.
Input is 3-channel SAR (VV, VH, VV−VH ratio) at 512×512. The production model is
**SegFormer-B4** (`nvidia/mit-b4`); DeepLabV3+, DeepLabV3, U-Net, U-Net++, FPN and
SegFormer-B2 are all available through the same architecture registry, data
contract and scorer. Predictions are refined with JRC permanent-water and
DEM-derived layover/shadow post-processing, and browsable in a Next.js map viewer.

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
Production model is **SegFormer-B4**, trained two-stage (weak-label pretrain →
hand-label fine-tune). Inference fuses JRC permanent-water and DEM-derived
layover/shadow masks into the flood mask. Locked hyperparameters / paths live in
[config/default.yaml](config/default.yaml).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: make install
```

## Target CLI
```bash
make help                        # list all targets
make install                     # install pinned deps into .venv
make config                      # validate config/default.yaml
make train                       # train the production baseline (SegFormer-B4)
make eval                        # evaluate a checkpoint on a split
make predict INPUT=.. OUTPUT=.. [PROB=..] [PERMANENT_WATER=..] [LAYOVER_SHADOW=..] \
             [NO_PERMANENT_WATER=1] [NO_LAYOVER_SHADOW=1]   # tiled predict -> georeferenced GeoTIFF
make finalists [ONLY=<run_name>] # architecture/loss sweep runs
make download-weak-data          # Sen1Floods11 weak-labeled chips (~6.8 GiB)
make weak-splits                 # build train/val split CSVs for the weak chips
make pretrain-finetune           # weak-label pretrain + hand-label fine-tune
make demo-artifacts              # regenerate web/public/data mask/SAR PNGs + GeoTIFFs
make vast-bootstrap              # set up a fresh Vast.ai GPU box (no training launched)
make mlflow-ui                   # MLflow UI (sqlite:///mlflow.db)
make dvc-push / make dvc-pull    # data & model versioning
make lint                        # byte-compile all source
```

## Data & model versioning

DVC stores the large artifacts in a shared Google Drive remote, not in git:

- `data/processed/sen1floods11/` — hand- and weak-labeled SAR chips
- `data/reference/dem/` — Copernicus GLO-30 DEM (SRTM1 / GLO-90 fallback per tile)
- `data/reference/jrc_gsw/` — JRC Global Surface Water occurrence (permanent water)
- `data/demo_scenes/*.tif` — wide-AOI Sentinel-1 scenes for the web viewer
- `models/segformer_b4/`, `models/best.ckpt`, `models/last.ckpt`,
  `models/weak_pretrain_finetune/` — checkpoints
- `mlflow.db` — experiment-tracking database

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
│   ├── default.yaml        # production config (SegFormer-B4)
│   └── experiments/        # loss study, weak pretrain/fine-tune, SegFormer/B4 & USA variants
├── data/
│   ├── raw/
│   ├── processed/          # Sen1Floods11 hand- and weak-labeled chips (DVC-tracked)
│   ├── reference/          # DEM (GLO-30) + JRC GSW permanent water (DVC-tracked)
│   ├── demo_scenes/        # wide-AOI Sentinel-1 scenes for the viewer (DVC-tracked)
│   └── splits/             # official + weak train/val/test CSVs
├── docs/                   # study reports (loss_study, usa_country_ablation, fixed_locations)
├── models/                 # checkpoints (segformer_b4/, best/last, weak_pretrain_finetune/ DVC-tracked)
├── notebooks/              # EDA + Colab sweep notebooks
├── scripts/                # data download/harvest, split generation, sweep/pretrain-finetune runners, demo artifacts
├── src/
│   ├── data/               # dataset, datamodule, preprocessing, transforms, split loader
│   ├── models/             # architecture registry + SegFormer wrapper (build.py)
│   ├── training/           # Lightning module, losses, train loop, MLflow logging
│   ├── inference/          # tiling, stitching, TTA, postprocess, permanent-water, predict, evaluate
│   └── utils/              # config (pydantic), MLflow helpers
├── web/                    # Next.js map viewer (flood-mask browser)
└── Makefile
```

## Workflow

1. **Prepare data** — `bash scripts/download_data.sh` (hand-labeled) and
   `make download-weak-data` + `make weak-splits` (weak-labeled, for pretraining).
   For post-processing reference data: `scripts/download_dem.py` (Copernicus GLO-30
   DEM) and `scripts/download_jrc_gsw.py` (JRC permanent water). Or `make dvc-pull`
   to fetch everything already tracked.
2. **Train** — `make train` (production SegFormer-B4) or `make finalists` /
   `make pretrain-finetune` for sweeps and two-stage transfer learning.
3. **Evaluate** — masked IoU / F1 on a held-out split (`make eval`), test split
   touched once. Optional flip-based TTA via `inference.tta` (or `--tta`).
4. **Infer** — `make predict INPUT=scene.tif OUTPUT=mask.tif` runs a trained
   checkpoint on a full scene, tiling and stitching into a georeferenced
   flood-mask GeoTIFF. By default the JRC permanent-water mask is fused in
   (subtracting permanent water from flooded areas) and, when
   `inference.layover_shadow` is configured, DEM-derived layover/shadow regions
   are handled too. Disable per stage with `NO_PERMANENT_WATER=1` /
   `NO_LAYOVER_SHADOW=1` (or the matching `--no-*` CLI flags). Add `PROB=prob.tif`
   for the per-pixel probability, `PERMANENT_WATER=..` / `LAYOVER_SHADOW=..` to
   also export those masks as their own georeferenced GeoTIFFs.
5. **Harvest real scenes** — `scripts/pull_s1_scene.py` / `scripts/harvest_s1_scenes.py`
   pull wide-AOI Sentinel-1 scenes from Google Earth Engine for the fixed demo
   locations (Baku, Sabirabad, India, Pakistan, Bolivia).
6. **Publish to the viewer** — `make demo-artifacts` regenerates the mask / SAR
   PNGs and GeoTIFFs under `web/public/data/` from the demo scenes.

Every run logs to MLflow (params, git SHA, DVC data hash, metrics, checkpoint) —
see `make mlflow-ui`.

## Web viewer

`web/` is a Next.js 16 / React 19 / TypeScript app (Leaflet + Tailwind 4) that
browses flood predictions for the fixed demo locations. The overview page shows a
zone gallery and a severity dashboard; each location page renders a Leaflet map
with a historical date picker and toggleable layers — flood mask, JRC
permanent-water (recolored to stay distinct from OSM's river blue), model
confidence/probability, and the raw SAR scene for verification — plus PNG/GeoTIFF
download. All data is static (`web/public/data/locations.json` + generated assets,
produced by `make demo-artifacts`); there is no backend API.

```bash
cd web
npm install
npm run dev     # http://localhost:3000
npm run build   # production build
```

## Experiments & studies

- [docs/loss_study.md](docs/loss_study.md) — loss / class-imbalance study.
- [docs/usa_country_ablation.md](docs/usa_country_ablation.md) — single-country
  (USA) vs. multi-country training ablation.
- [docs/fixed_locations.md](docs/fixed_locations.md) — the five fixed evaluation
  locations. Per-country generalization is measured with the LOCO runner
  (`scripts/run_loco.py`, `scripts/make_loco_splits.py`).
