# Historical Leave-One-Country-Out (LOCO) Experiment

**Date:** 2026-07-14
**Source:** `flood-water-seg/loco/<country>` MLflow experiments (`scripts/run_loco.py`)

> **Current study.** See [Interpolation vs. Generalization in SAR Water Segmentation](https://github.com/AliAgabalayev/sar-flood-segmentation-loco)
> for the paired comparison on identical official Sen1Floods11 test chips.
> The paper reports nine recoverable event–seed comparisons across seven events,
> with five positive and four negative water-IoU differences.

## 1. Goal

The July 2026 experiment measured how event-excluded models performed on each
target event when that event was removed from the training and validation
manifests used by the LOCO runner. The historical evaluation pooled all
available hand-labeled chips of the target event across the official training,
validation, and test splits.

## 2. Method

- `scripts/make_loco_splits.py` builds, per held-out country, `train9`/`weak9`
  (official + weak splits with that country's chips removed) and `heldout`
  (all official chips — train ∪ val ∪ test — belonging to that country only),
  under `data/splits/loco/<country>/`.
- `scripts/run_loco.py` runs the fold: weak-pretrain on `weak9` →
  hand-fine-tune on `train9` (`config/experiments/weak_pretrain.yaml` /
  `weak_finetune.yaml`) → evaluate the fine-tuned checkpoint on `heldout`.
  Each stage logs to its own MLflow experiment
  (`flood-water-seg/loco/<country>`); held-out metrics are logged separately
  via `MlflowClient` rather than the fluent API, since `mlflow.set_experiment()`
  is sticky across countries in the same long-lived process.
- The 10 target groups: Ghana, India, Mekong, Nigeria, Pakistan, Paraguay,
  Somalia, Spain, Sri-Lanka, USA.

## 3. Historical results

**July 2026 run status: 9 of 10 folds complete.** Somalia's pretrain stage is logged
as `RUNNING` with no finished run — the sweep was interrupted before that
fold finished. Ghana's pretrain stage shows two
`RUNNING` and one `FAILED` attempt before the `FINISHED` run used below, and
USA needed a full retry of both pretrain and finetune (an abandoned
`RUNNING` finetune attempt and a superseded pretrain checkpoint, kept as
`best-v1`/`last-v1` in `models/loco/USA/`) before either produced the
checkpoints used below.

| Target group | All-chip target IoU | All-chip target F1 | Other-group validation IoU† | IoU difference‡ |
|---|---|---|---|---|
| Mekong | **0.8635** | **0.9267** | 0.6362 | +0.2273 |
| Nigeria | 0.7831 | 0.8784 | 0.6352 | +0.1479 |
| Spain | 0.7272 | 0.8420 | 0.6426 | +0.0846 |
| Sri-Lanka | 0.6753 | 0.8062 | 0.6623 | +0.0130 |
| Paraguay | 0.5470 | 0.7071 | 0.6905 | −0.1435 |
| India | 0.4992 | 0.6659 | 0.6516 | −0.1524 |
| USA | 0.4814 | 0.6499 | 0.6516 | −0.1702 |
| Ghana | 0.4804 | 0.6490 | 0.6589 | −0.1786 |
| Pakistan | 0.3104 | 0.4738 | 0.6704 | **−0.3600** |
| Somalia | — (interrupted) | — | — | — |

**Mean over the 9 completed folds:** IoU 0.5964, F1 0.7332.

† Best validation IoU logged during fine-tuning on the other groups.

‡ All-chip target IoU minus the reported best validation IoU from the
corresponding fine-tune run. The scores use different chip pools and data roles.

## 4. Findings

In this historical run, four completed folds had positive differences and five
had negative differences under the metric in the table. The table records a
comparison with the LOCO run's own validation score. The current paper compares
event-excluded models with a fixed generalist using the same official test
chips and valid pixels. Use Table 1 of the paper for that comparison.

## 5. Historical commands

```bash
python scripts/make_loco_splits.py                 # build data/splits/loco/<country>/*
python scripts/run_loco.py                         # all 10 folds
python scripts/run_loco.py --only Somalia                       # run the historical Somalia fold
python scripts/run_loco.py --skip-pretrain --only Pakistan      # cheaper: skip weak-pretrain
```

Held-out metrics are logged as `heldout_iou` / `heldout_f1` on a
`heldout_<country>` run in `flood-water-seg/loco/<country>`, alongside
`finetune_val_iou` for the in-distribution comparison — see `make mlflow-ui`.
