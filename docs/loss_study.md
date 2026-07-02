# M1 — Loss & Class-Imbalance Study

**Goal.** Pick the loss function (and imbalance weighting) that maximises
validation **IoU** and **F1** for binary flood-water segmentation on
Sen1Floods11, with everything else held fixed. We compare four losses —
`bce`, `dice`, `dice_bce`, `focal` — and tune the class-imbalance controls
(`pos_weight` for BCE, `alpha` for focal).

> **Status:** code, sweep runner and Colab notebook are ready and smoke-tested.
> The results table below is auto-filled from the sweep — see
> [How to reproduce](#how-to-reproduce). Until the GPU sweep is run it shows a
> placeholder.

---

## 1. Why this matters: the data is heavily imbalanced

Flood water is the rare class. Over the hand-labeled training chips (valid
pixels only):

| Class | Pixels | Share |
| --- | --- | --- |
| No-flood (0) | 90,277,709 | 89.4 % |
| Flood (1) | 10,705,605 | 10.6 % |
| **Ratio (neg / pos)** | **8.43** | — |

A naive pixel loss (plain BCE) can reach ~89 % pixel accuracy by predicting
"no water" everywhere, while scoring ~0 IoU on the class we actually care
about. The whole point of this study is to counteract that bias. IoU and F1 are
computed **on the positive (flood) class only**, so they punish that failure
mode directly.

## 2. Fixed experimental setup

Only the loss (and its imbalance knob) changes between runs. Everything below is
locked from [`config/default.yaml`](../config/default.yaml):

| Component | Value |
| --- | --- |
| Model | DeepLabV3+ / ResNet-34 (ImageNet-pretrained), 3-ch SAR in, 1 logit out |
| Input | 512×512, channels = VV, VH, VV−VH ratio (clipped, per data contract) |
| Optimizer / LR | Adam, 3e-4 |
| Batch | 2 × grad-accum 4 = effective 8; 16-mixed precision on GPU |
| Epochs | 40 with early stopping on `val_iou` (patience 8) for the sweep |
| Seed | 42 (`seed_everything(..., workers=True)`) |
| Split | Official Sen1Floods11 hand-labeled: 252 train / 89 val / 90 test |
| Metrics | `BinaryJaccardIndex` (IoU) + `BinaryF1Score`, masked to valid pixels |

All losses are **masked**: pixels marked invalid (label `-1` or non-finite SAR)
are excluded so they never contribute gradient or metric. See
[`src/training/losses.py`](../src/training/losses.py).

## 3. The candidate losses

### 3.1 BCE (`bce`) — with `pos_weight`
Per-pixel binary cross-entropy. `pos_weight = w` multiplies the loss on positive
pixels, directly countering imbalance: `w = 8.4` makes one flood pixel count as
much as ~8.4 background pixels (the exact class ratio).

- **Pros:** smooth, well-behaved gradients everywhere; `pos_weight` is a single,
  interpretable imbalance dial; fast and stable.
- **Cons:** optimises per-pixel likelihood, **not overlap** — good BCE ≠ good
  IoU. Too-high `pos_weight` causes over-prediction (high recall, low precision,
  ragged masks). Doesn't exploit region structure.
- **Imbalance knob:** `pos_weight` (swept: 1, 4, 8.4, 13).

### 3.2 Dice (`dice`) — region overlap
Soft Dice ≈ differentiable F1 on the positive class. It is a **ratio** of
overlap to area, so it is *inherently scale-invariant to class frequency* — no
weighting term needed.

- **Pros:** directly optimises the overlap we score on; naturally imbalance-
  robust; usually strong IoU/F1 out of the box.
- **Cons:** gradients can be noisy/unstable early (small denominators when
  predictions are near-empty); no per-pixel calibration; a single-pixel
  chip-level miss swings the loss a lot on tiny-water chips.
- **Imbalance knob:** none (built-in). `pos_weight` does not apply.

### 3.3 Dice + BCE (`dice_bce`) — the combination
Weighted sum `bce_weight · BCE + dice_weight · Dice` (default 1·BCE + 1·Dice).
BCE supplies dense, stable per-pixel gradients; Dice pulls the solution toward
overlap. This "best of both" is the most common default for medical/remote-
sensing segmentation.

- **Pros:** typically the most robust; BCE stabilises Dice's early training while
  Dice fixes BCE's overlap blind spot; still accepts `pos_weight` on the BCE
  term for extra imbalance control.
- **Cons:** two effects to balance (the `bce_weight`/`dice_weight`/`pos_weight`
  interaction); slightly more to tune; can double-count the imbalance signal if
  `pos_weight` is pushed hard on top of Dice.
- **Imbalance knob:** `pos_weight` on the BCE term (run at 8.4) + component
  weights.

### 3.4 Focal (`focal`) — hard-example mining
Focal loss `−α·(1−p_t)^γ·log(p_t)` down-weights easy, confidently-correct pixels
(the vast easy background) so training focuses on hard/rare pixels. `gamma`
controls focusing strength; `alpha` balances the positive class.

- **Pros:** attacks imbalance from the "easy-negative swamping" angle; can beat
  plain BCE when background is overwhelmingly easy; `gamma` adds a second lever.
- **Cons:** more sensitive to `alpha`/`gamma`; like BCE it optimises pixels not
  overlap, so IoU can lag Dice-based losses; can under-train if `gamma` too high.
- **Imbalance knob:** `alpha` (0.25) and `gamma` (2.0) — **not** `pos_weight`.

### Quick comparison

| Loss | Optimises | Imbalance handled by | Gradient stability | Typical IoU strength |
| --- | --- | --- | --- | --- |
| `bce` | per-pixel likelihood | `pos_weight` | high | low–medium |
| `dice` | region overlap | intrinsic (ratio) | medium (noisy early) | high |
| `dice_bce` | both | `pos_weight` + Dice | high | high |
| `focal` | hard pixels | `alpha`/`gamma` | medium | medium |

## 4. Experiment matrix (lean, ~7 runs)

Defined in [`config/experiments/loss_study.yaml`](../config/experiments/loss_study.yaml).
`pos_weight` is only meaningful for the BCE term, so it is swept for `bce` and
applied at the dataset ratio for `dice_bce`; `dice` uses none and `focal` uses
`alpha` instead.

| Run | Loss | pos_weight | focal α/γ |
| --- | --- | --- | --- |
| `bce_pw1` | bce | 1.0 | — |
| `bce_pw4` | bce | 4.0 | — |
| `bce_pw8.4` | bce | 8.4 | — |
| `bce_pw13` | bce | 13.0 | — |
| `dice` | dice | — | — |
| `focal_a0.25` | focal | — | 0.25 / 2.0 |
| `dice_bce_pw8.4` | dice_bce | 8.4 | — |

Rationale for keeping it lean: `pos_weight` only interacts with BCE, so a 4-point
sweep on `bce` fully characterises that axis; `dice`/`focal` each need a single
run; `dice_bce` is anchored at the ratio (8.4). This isolates the loss effect
cheaply. A wider grid (focal `alpha` sweep, `dice_bce` `pos_weight` sweep, custom
component weights) is the obvious follow-up if two candidates finish close.

## 5. How to reproduce

### Local smoke test (proves the pipeline, seconds)
```bash
# Windows PowerShell (repo root); Bash equivalent: PYTHONPATH=src python ...
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe scripts/run_loss_study.py --smoke
```
Runs every loss for 1 epoch on 2 batches. Used to validate the code path — the
numbers are meaningless.

### Full sweep on Colab (produces the real numbers)
Open [`notebooks/02_loss_study_colab.ipynb`](../notebooks/02_loss_study_colab.ipynb)
in Colab, set the runtime to **GPU (T4)**, and run all cells. It clones the repo,
installs deps (keeping Colab's CUDA PyTorch), downloads the public Sen1Floods11
data via `gsutil`, runs the sweep, and fills the table below. Then locally:
```bash
python scripts/make_results_table.py --update-report   # refresh the table here
```

### Why not local GPU?
This machine has an **NVIDIA MX350 (2 GB VRAM)** and a **CPU-only PyTorch
build** (`torch==2.12.1+cpu`, `cuda.is_available() == False`). Measured from the
smoke test (~3 s / training step on CPU at 512²), a single 40-epoch run is
≈ 5 hours and the full 7-run sweep ≈ 1.5 days — and DeepLabV3+/ResNet-34 at 512²
does not fit in 2 GB anyway. Hence the sweep runs on Colab. (The README's
"RTX 3060 6 GB" refers to the intended target hardware, not this box.)

## 6. Results

Validation metrics at the best-`val_iou` epoch, ranked (winner in **bold**).
Auto-generated by `scripts/make_results_table.py --update-report`.

<!-- RESULTS:START -->

_No results yet — run the Colab sweep, then `python scripts/make_results_table.py --update-report`._

<!-- RESULTS:END -->

## 7. How to read the results & pick a winner

Decision rule, in order:
1. **Primary:** highest **val IoU** (this is what `ModelCheckpoint` selects on).
2. **Tie-break (within ~0.005 IoU):** higher **val F1**, then better
   precision/recall balance (a model that only wins by over-predicting water is
   worse than the number suggests).
3. **Stability:** prefer the loss that reached its best earlier / trained
   smoothly (see `best_epoch` and MLflow curves) — cheaper and more reliable to
   retrain.

**Prior expectation from the literature** (SAR water / imbalanced segmentation):
`dice_bce` and `dice` usually top plain `bce`/`focal` on IoU, because they
optimise overlap directly; `bce` peaks around `pos_weight ≈` the class ratio
(8.4) and degrades (over-prediction) beyond it; `focal` helps over plain BCE but
rarely beats Dice-based losses on IoU. **This is a hypothesis to confirm with the
table above, not the conclusion.**

## 8. Recommendation

_To be finalised once the sweep completes._ Fill in after Section 6:

- **Chosen loss:** `<loss>` (`pos_weight`/`alpha` = `<value>`) — best val IoU
  `<x.xxx>`, F1 `<x.xxx>`.
- **Runner-up & margin:** `<loss>` at `<ΔIoU>`.
- **Action:** set `training.loss` (and `pos_weight`) in `config/default.yaml`,
  then run the full 50-epoch production training and evaluate on the **test**
  split with `python -m inference.evaluate`.

## 9. Reproducibility notes
- Seed 42 fixed for weights, data order and workers; per-run checkpoints under
  `models/loss_study/<run>/` so runs don't clobber each other.
- Every run is logged to MLflow (`sqlite:///mlflow.db`, experiment
  `flood-water-seg`) with the config and git SHA; the sweep also appends a row to
  `experiments/loss_study/results.csv` (git-ignored artifact).
- The sweep is resumable: re-running skips runs already in the CSV (`--force` to
  redo, `--only <names>` for a subset).
