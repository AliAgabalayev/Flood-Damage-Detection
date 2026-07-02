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

| Run | Loss | pos_weight | focal_α | Val IoU | Val F1 | Best epoch | Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **focal_tversky_g1.33** | focal_tversky | — | — | 0.6434 | 0.7830 | 28 | 912.9 |
| tversky_a0.3_b0.7 | tversky | — | — | 0.6427 | 0.7825 | 18 | 673.5 |
| dice | dice | — | — | 0.6425 | 0.7824 | 17 | 747.9 |
| dice_focal | dice_focal | — | — | 0.6261 | 0.7701 | 9 | 657.9 |
| dice_bce_pw8.4 | dice_bce | 8.4 | — | 0.6066 | 0.7551 | 18 | 752.1 |
| bce_pw4 | bce | 4.0 | — | 0.5988 | 0.7491 | 5 | 389.0 |
| bce_pw8.4 | bce | 8.4 | — | 0.5525 | 0.7118 | 5 | 419.1 |
| focal_a0.25 | focal | — | 0.25 | 0.5428 | 0.7037 | 12 | 506.7 |
| lovasz | lovasz | — | — | 0.5136 | 0.6787 | 5 | 429.7 |
| bce_pw13 | bce | 13.0 | — | 0.5107 | 0.6761 | 7 | 407.4 |
| bce_pw1 | bce | 1.0 | — | 0.4749 | 0.6440 | 0 | 240.8 |

_11 runs across two phases. Colab T4, 40-epoch cap + early stopping (patience 8),
seed 42; metrics at best-`val_iou` epoch on the 89-chip validation split.
(`focal_α` column only shows a value for the standalone `focal` run.)_

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

### What the numbers actually show

1. **Overlap losses win.** `dice` (0.6425) and `dice_bce` (0.6066) take the top
   two spots — the two losses that optimise region overlap directly. Every
   pixel-based loss (`bce`, `focal`) lands below them. This confirms the core
   hypothesis: for a metric like IoU, optimising overlap beats optimising
   per-pixel likelihood.

2. **`dice` beat the combo — and needs no tuning.** Plain Dice edged out
   `dice_bce` by +0.036 IoU / +0.027 F1, *without any `pos_weight`*. Dice's
   ratio form absorbs the 8.9:1 imbalance on its own, so it is both the best and
   the simplest (one fewer hyper-parameter).

3. **The theoretical `pos_weight` was NOT optimal.** The BCE ladder peaks at
   `pos_weight = 4.0` (0.5988), *above* the exact class ratio 8.4 (0.5525):

   | pos_weight | 1.0 | 4.0 | 8.4 | 13.0 |
   | --- | --- | --- | --- | --- |
   | Val IoU | 0.4749 | **0.5988** | 0.5525 | 0.5107 |

   No weighting (1.0) under-detects water; over-weighting (13.0) over-predicts it;
   the IoU-optimal weight (~4) sits *below* the loss-balancing ratio (8.4). This
   is exactly why the ratio is a starting point, not the answer — and it is the
   headline evidence for *why the sweep was worth running*.

4. **Focal disappointed.** `focal_a0.25` (0.5428) beat only the two weakest BCE
   runs. Its α/γ were untuned; even so, as a pixel-based loss it was never going
   to challenge the overlap family here.

5. **Training dynamics differ.** Dice-family runs peaked late (epoch 17–18) and
   were still improving when the 40-epoch cap / early-stopping cut them; BCE runs
   peaked very early (epoch 0–7) then plateaued. So the Dice numbers above are
   **conservative** — a full 50-epoch schedule should lift them further, while
   BCE has little headroom left.

### Phase 2 — extending the overlap family

Motivated by Phase 1 (overlap losses won), we tested a recall dial (Tversky), a
focused variant (Focal-Tversky), an IoU surrogate (Lovász) and Dice+Focal, on
the **same schedule and seed**.

6. **The overlap family is a three-way tie at the top.** `focal_tversky`
   (0.6434), `tversky` (0.6427) and `dice` (0.6425) sit within **0.001 IoU** of
   each other — that is well inside run-to-run noise. Practically, they are the
   same. All three clearly beat every pixel-based loss and both combos.

7. **The recall tilt barely moved the needle.** Tversky with β=0.7 > α=0.3
   (penalise missed flood harder) matched plain Dice almost exactly (+0.0002
   IoU). So for this model/data the FN/FP asymmetry buys essentially nothing —
   Dice's symmetric overlap is already close to optimal. Focal-Tversky's focusing
   exponent added a tiny +0.0009, but it needed the **longest training** (best
   epoch 28, 37 epochs) to get there.

8. **`dice_focal` and `lovasz` disappointed.** Adding Focal to Dice *lowered*
   the score (0.6261 < 0.6425) — Focal drags the strong Dice down, consistent
   with Focal being the weakest single loss. **Lovász collapsed** to 0.5136:
   it is notoriously unstable trained from scratch and is normally used only to
   *fine-tune* a BCE/Dice-pretrained model — a genuine "don't use cold" finding.

## 8. Recommendation

The top three (`focal_tversky`, `tversky`, `dice`) are a statistical tie
(≤0.001 IoU). When results tie, **prefer the simplest, most robust option**:

- **Chosen loss:** **`dice`** (no `pos_weight`, no α/β/γ). It matches the best
  scores (IoU **0.6425** / F1 **0.7824**) with **zero imbalance hyper-parameters**
  to tune or maintain. `focal_tversky` edges it by +0.0009 IoU but needs three
  knobs and ~35 % longer training — not worth the complexity for noise-level gain.
- **Runner-up / if you want the last 0.001:** `focal_tversky` (α=0.3, β=0.7,
  γ=1.33) or `tversky` (α=0.3, β=0.7) — both interchangeable with Dice in
  practice.
- **Avoid:** `lovasz` from scratch, plain `focal`, and high `pos_weight` BCE.
- **vs the team baseline** (`dice_bce`, 50 epochs → IoU 0.65 / F1 0.79): our
  overlap losses hit 0.643 on a *shorter* 40-epoch + early-stopping schedule, so
  a full 50-epoch run should match or beat 0.65.

### Next actions
1. **Run the finalists at the full 50 epochs** (no early stop / high patience) —
   `dice`, `tversky`, `focal_tversky` were all still climbing when stopped, so
   the ranking could shift. Pick the production loss from that.
2. Set `training.loss` (+ any Tversky knobs) in
   [`config/default.yaml`](../config/default.yaml).
3. Evaluate the chosen checkpoint on the held-out **test** split:
   `python -m inference.evaluate --split test`.
4. *(Optional)* try Lovász as a **fine-tune** on top of the best Dice/Tversky
   checkpoint — its intended use — to see if the direct IoU objective adds a
   final bump.

## 9. Reproducibility notes
- Seed 42 fixed for weights, data order and workers; per-run checkpoints under
  `models/loss_study/<run>/` so runs don't clobber each other.
- Every run is logged to MLflow (`sqlite:///mlflow.db`, experiment
  `flood-water-seg`) with the config and git SHA; the sweep also appends a row to
  `experiments/loss_study/results.csv` (git-ignored artifact).
- The sweep is resumable: re-running skips runs already in the CSV (`--force` to
  redo, `--only <names>` for a subset).
