# Leave-One-Country-Out (LOCO) Generalization Study

**Date:** 2026-07-14
**Source:** `flood-water-seg/loco/<country>` MLflow experiments (`scripts/run_loco.py`)

---

## 1. Goal

The official Sen1Floods11 split does not hold out entire regions — every
country in val/test also has chips in train, so headline metrics
([docs/loss_study.md](loss_study.md), [docs/usa_country_ablation.md](usa_country_ablation.md))
measure interpolation across known regions, not generalization to an unseen
climate/geography ([docs/usa_country_ablation.md](usa_country_ablation.md)
flagged this as a limitation).

LOCO cross-validation closes that gap: for each of the 10 Sen1Floods11
countries, the two-stage recipe (weak-label pretrain → hand-label fine-tune)
is retrained from scratch on the other 9 countries only, then evaluated on
**every** official chip (train + val + test) of the held-out country — data
the model never saw in any form.

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
- The 10 countries: Ghana, India, Mekong, Nigeria, Pakistan, Paraguay,
  Somalia, Spain, Sri-Lanka, USA.

## 3. Results

**Run status: 9 of 10 folds complete.** Somalia's pretrain stage is logged
as `RUNNING` with no finished run — the sweep was interrupted before that
fold executed and it hasn't been resumed. Ghana's pretrain stage shows two
`RUNNING` and one `FAILED` attempt before the `FINISHED` run used below, and
USA needed a full retry of both pretrain and finetune (an abandoned
`RUNNING` finetune attempt and a superseded pretrain checkpoint, kept as
`best-v1`/`last-v1` in `models/loco/USA/`) before either produced the
checkpoints used below.

| Held-out country | Held-out IoU | Held-out F1 | In-distribution val IoU† | Generalization gap‡ |
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

† `best_val_iou` from the fine-tune run — the model's score on a held-out
*validation split of the other 9 countries*, i.e. still in-distribution.
‡ Held-out IoU minus in-distribution val IoU. Positive means the model
generalized to the unseen country *better* than to its own training
distribution's val split; negative means the unseen country was harder.

## 4. Findings

- **Generalization is highly country-dependent, not uniformly degraded.**
  Mekong, Nigeria, Spain and Sri-Lanka all score *higher* held out than the
  fine-tune stage scored on its own in-distribution val set — for Mekong and
  Nigeria especially, these countries' flood signature (large, high-contrast
  inundation extents) is apparently easy to pick up from the other 9
  countries alone, with no country-specific training data needed. Spain and
  Sri-Lanka generalize positively too, but by a much smaller margin
  (+0.08 / +0.01), so the effect isn't binary.
- **Pakistan is the hardest country to generalize to** in this set: a
  −0.36 IoU gap, the largest by more than double the next-worst (Ghana,
  −0.18). Combined with the [USA ablation](usa_country_ablation.md) (which
  showed a *single-country* specialist beating the 10-country generalist by
  +0.03 IoU on its own region), this is consistent with Pakistan's flood
  conditions being under-represented or distinct enough in the other 9
  countries that the generalist model doesn't transfer to it well.
- **The missing Somalia fold is a real gap in this study, not a rounding
  error** — its in-flight pretrain run should be resumed or restarted
  rather than treated as data. With 9/10 folds, the 0.5964 mean IoU is a
  near-complete estimate, but not final until Somalia is included.
- **Ghana's and USA's retries** — Ghana's pretrain stage shows 2 `RUNNING` +
  1 `FAILED` attempt before the run that finished; USA needed a full retry
  of both stages (an abandoned `RUNNING` finetune attempt and a superseded
  pretrain checkpoint, kept as `best-v1`/`last-v1`) — suggest the LOCO sweep
  isn't yet resilient to whatever caused those interruptions, worth
  investigating before resuming Somalia so the same failure doesn't recur.

## 5. Reproduce

```bash
python scripts/make_loco_splits.py                 # build data/splits/loco/<country>/*
python scripts/run_loco.py                         # all 10 folds
python scripts/run_loco.py --only Somalia                       # resume the missing fold
python scripts/run_loco.py --skip-pretrain --only Pakistan      # cheaper: skip weak-pretrain
```

Held-out metrics are logged as `heldout_iou` / `heldout_f1` on a
`heldout_<country>` run in `flood-water-seg/loco/<country>`, alongside
`finetune_val_iou` for the in-distribution comparison — see `make mlflow-ui`.
