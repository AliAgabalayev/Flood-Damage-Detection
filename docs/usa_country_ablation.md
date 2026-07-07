# G9 — USA Single-Country Ablation vs. Current Best Model

**Goal.** Check whether the current best model (two-stage weak-pretrain →
hand-finetune DeepLabV3+/resnet50, trained on all 10 Sen1Floods11 regions)
actually benefits from multi-country training data, versus an identical
recipe restricted to a single country (USA) at both stages. See
[design spec](superpowers/specs/2026-07-06-usa-country-ablation-design.md)
for full methodology.

## Result

Both models evaluated on `data/splits/official_usa/val.csv` (14 USA chips,
a subset of the official hand-labeled val split — the official test split
is not touched by this ablation):

| Model | Train data | Val IoU | Val F1 |
|---|---|---|---|
| Current best (multi-country) | 252 chips, 10 countries | 0.6286 | 0.7720 |
| USA-only ablation | 41 chips, 1 country | 0.6611 | 0.7960 |

The USA-only model scores higher on its own region than the generalist
model — consistent with a specialist/generalist trade-off: the multi-country
model has to fit ten climates/geographies at once, so on any single region
it is a compromise rather than the best achievable model for that region.

**Caveat.** 14 chips is a small sample — this comparison is indicative, not
statistically conclusive. It does, however, support a broader limitation
worth stating in the paper: the official Sen1Floods11 split does not hold
out entire regions (every country in the val/test split also has chips in
train), so headline metrics measure interpolation across known regions
rather than generalization to an unseen climate/geography.

## Reproduce

```
python scripts/run_pretrain_finetune.py \
  --pretrain-config config/experiments/weak_pretrain_usa.yaml \
  --finetune-config config/experiments/weak_finetune_usa.yaml

python -m inference.evaluate --config config/experiments/eval_bestmodel_usa_val.yaml --split val
python -m inference.evaluate --config config/experiments/weak_finetune_usa.yaml --split val
```
