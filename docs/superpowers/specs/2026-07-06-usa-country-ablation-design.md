# USA Single-Country Ablation vs. Current Best Model

## Purpose

For the paper, verify whether the current best model (two-stage weak-pretrain →
hand-finetune DeepLabV3+/resnet50, `models/weak_pretrain_finetune/finetune/best.ckpt`,
trained on all 10 Sen1Floods11 regions) actually benefits from multi-country
training data, versus a model trained with the identical recipe but restricted
to a single country (USA) at both training stages. Comparison is made on the
USA-only validation split so the only varying factor between the two models is
training-data breadth.

## Scope

- "Country" is the only geographic granularity in Sen1Floods11 (filenames are
  `{Country}_{chipID}_S1Hand.tif`); there is no city-level metadata. "City" in
  the user's request is treated as "one country/region."
- Country selected: **USA**.
- Both training stages (weak pretrain and hand-label finetune) are restricted
  to USA-only data — no other-country data leaks into either stage.
- Comparison is on validation only; the official test split is not touched by
  this ablation (test split is a project-wide single-use resource per
  `run_training`'s docstring).
- Hyperparameters (epochs, lr, loss, backbone, batch size, etc.) are identical
  to the existing `weak_pretrain.yaml` / `weak_finetune.yaml` — only the data
  paths, checkpoint directory, and MLflow experiment name differ. This isolates
  training-data breadth as the sole independent variable.

## Data prep

New split files, filtered by `USA_` filename prefix from the existing splits,
committed to git like the existing split CSVs (not DVC-tracked, they're small
text files):

- `data/splits/weak_usa/train.csv` (430 rows), `data/splits/weak_usa/val.csv` (56 rows)
  — filtered from `data/splits/weak/{train,val}.csv`
- `data/splits/official_usa/train.csv` (41 rows), `data/splits/official_usa/val.csv` (14 rows)
  — filtered from `data/splits/official/{train,val}.csv`

No `test.csv` is created for either — this ablation does not touch test data.

## Configs

Two new experiment configs, cloned from `weak_pretrain.yaml` / `weak_finetune.yaml`
with only these fields changed:

`config/experiments/weak_pretrain_usa.yaml`
- `data.split_dir: data/splits/weak_usa`
- `training.checkpoint_dir: models/country_ablation_usa/pretrain`
- `mlflow.experiment: flood-water-seg/country_ablation_usa`
- `inference.checkpoint: models/country_ablation_usa/pretrain/best.ckpt`

`config/experiments/weak_finetune_usa.yaml`
- `data.split_dir: data/splits/official_usa`
- `training.checkpoint_dir: models/country_ablation_usa/finetune`
- `mlflow.experiment: flood-water-seg/country_ablation_usa`
- `inference.checkpoint: models/country_ablation_usa/finetune/best.ckpt`

One new eval-only config, cloned from `weak_finetune.yaml` (same data/model
block as the current best model, so it loads the current best checkpoint
correctly), used only to point the current best model at the USA-only val set:

`config/experiments/eval_bestmodel_usa_val.yaml`
- `data.split_dir: data/splits/official_usa`
- `inference.checkpoint: models/weak_pretrain_finetune/finetune/best.ckpt` (unchanged — current best)

New checkpoints under `models/country_ablation_usa/` are git-ignored like all
other files under `models/*` (matches existing behavior for
`models/weak_pretrain_finetune/*`).

## Training

Reuse `scripts/run_pretrain_finetune.py` as-is (already parameterized via
`--pretrain-config`/`--finetune-config`), no code changes:

```
python scripts/run_pretrain_finetune.py \
  --pretrain-config config/experiments/weak_pretrain_usa.yaml \
  --finetune-config config/experiments/weak_finetune_usa.yaml
```

## Evaluation

**Bug fix required:** `src/inference/evaluate.py`'s `main()` unconditionally
calls `dm.setup("test")`, so `--split val` currently crashes with
`RuntimeError: val_dataset is not initialised` (val/train datasets are only
built for stages in `{None, "fit", "validate"}`, not `"test"`). Fix:

```python
dm.setup("test" if args.split == "test" else "validate")
```

Then evaluate both checkpoints on the same USA val split:

```
python -m inference.evaluate --config config/experiments/eval_bestmodel_usa_val.yaml --split val
python -m inference.evaluate --config config/experiments/weak_finetune_usa.yaml --split val
```

## Output

A side-by-side report for the paper:

| Model | Train data | Val IoU | Val F1 |
|---|---|---|---|
| Current best (multi-country) | 252 chips, 10 countries | ? | ? |
| USA-only ablation | 41 chips, 1 country | ? | ? |

Explicitly state the validation set used: **`data/splits/official_usa/val.csv`,
14 USA chips** (a subset of the same official Sen1Floods11 hand-labeled val
split the current best model was originally validated on, filtered to USA
only). Caveat to include in the paper: 14 images is a small sample, so the
comparison is indicative rather than statistically conclusive.

## Out of scope

- Not modifying the production `weak_pretrain.yaml`/`weak_finetune.yaml` or
  the current best checkpoint.
- Not touching the official test split.
- Not adding a generic `--checkpoint` CLI override to `evaluate.py` — the
  dedicated eval-only config is sufficient for this one comparison.
