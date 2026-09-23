# dlvr-g10

Group 10, Deep Learning for Visual Recognition, Aarhus University, autumn 2026.

Benign vs malignant skin-lesion classification on ISIC 2020, comparing four
treatments for a ~56:1 class imbalance at two fine-tuning depths.

| Arm | Treatment |
|-----|-----------|
| E0  | Plain BCE (baseline) |
| E0b | E0 re-thresholded — no training, the control |
| E1  | BCE with `pos_weight` |
| E2  | Balanced sampling (`WeightedRandomSampler`) |
| E3  | Focal loss |

E0b is post-hoc: it reuses E0's saved `preds.csv` and moves the decision
threshold, so it has no entry in `arms.json` and never trains.

## Rules

1. **`splits/dev_split.csv` and `splits/test_split.csv` are canonical.** Never
   regenerate them. If a split assertion fails, the answer is to use the saved
   CSVs, not to rebuild them.
2. **All training goes through `notebooks/G10_run.ipynb`.** No training loop in
   any other notebook — a second copy is a second explanation for any gap
   between arms.
3. **All metrics come from `eval.py`**, computed from saved `preds.csv`.
4. **`test_split.csv` stays closed until December.** Model selection,
   thresholds and arm settings use the inner validation split only.

## Setup

Three Colab Secrets, each with *Notebook access* switched on:

| Secret | Used for |
|--------|----------|
| `KAGGLE_USERNAME` | downloading the images |
| `KAGGLE_KEY` | downloading the images |
| `GH_TOKEN` | pushing results back to this repo |

`GH_TOKEN` is a GitHub fine-grained personal access token scoped to this
repository with **Contents: Read and write**. Each person makes their own.

## Data

Images are not in this repo. Each session pulls them from Kaggle:

```python
!kaggle datasets download -d cdeotte/jpeg-melanoma-256x256 -p /content -q
!unzip -q /content/jpeg-melanoma-256x256.zip -d /content/isic
```

792 MB, 256×256 JPEGs, centre-cropped then resized by the dataset author.
Not the ~108 GB competition archive. 33,126 images on disk; 32,693 survive
deduplication into the splits (28,063 dev / 4,630 test, 581 malignant).

The notebook does this for you and skips it if the data is already unpacked.

Two metadata traps: `width` and `height` are the *original* dimensions before
resize, never the actual 256×256 — don't use them for resizing. And
`benign_malignant` is the target in text form, so it and `diagnosis` must be
dropped explicitly.

## One run

1. Open `notebooks/G10_run.ipynb` in Colab (colab.research.google.com →
   GitHub tab → paste this repo's URL).
2. Runtime → Change runtime type → **T4 GPU**.
3. Edit cell 4 — `ARM`, `FOLD`, `SEED` — and nothing else.
4. Run the cells top to bottom. The last cell commits and pushes the results.

`ARM` is a key from `configs/arms.json`: `E0`/`E1`/`E2`/`E3` × `frozen`/`full`.

Each run writes `results/runs/<arm>_<depth>_f<fold>_s<seed>/` containing
`config.json` (settings + git hash), `history.csv` (per-epoch loss and AUROC)
and `preds.csv` (per-image probabilities). Every reported metric is derived
from `preds.csv`, so the analysis can change in November without retraining.

Checkpoints are deleted at the end of every run and blocked by `.gitignore`.
For the December test run, retrain the chosen config and predict in the same
session.

## Measured so far

Two baseline runs on fold 0, seed 0, T4 GPU:

| Run | Best inner AUROC | Peak at | Time |
|-----|------------------|---------|------|
| `E0_frozen_f0_s0` | 0.805 | epoch 6 (plateau from ~3) | 7.6 min |
| `E0_full_f0_s0` | 0.871 | epoch 2, declines after | 8.5 min |

The epoch counts in `arms.json` — 8 frozen, 5 full — were guesses and both
check out: frozen is still inching up at the end, full turns over at epoch 2
and the extra epochs are what prove it. Leave them alone.

`pos_weight: 7.5` (E1) and `gamma: 2.0` (E3) are still guesses. Each is the
owner's job to fix on inner validation before their real runs.

Budget: ~8 minutes per run, 40 runs, so roughly 5–6 GPU hours split three ways.

## The grid

4 arms × 2 depths × 5 folds = 40 runs, split **by arm**:

| Person | Arm |
|--------|-----|
| Luan | E1 — weighted loss |
| David | E2 — balanced sampling |
| Mikkel | E3 — focal loss |

E0 (and therefore E0b) is shared. Each person owns their method end to end:
builds it, tunes it, runs all its folds.

This means method and machine coincide. We know, and we state it as a
limitation in Methods rather than pretending otherwise.

## Layout

```
.gitignore
README.md
configs/    arms.json
notebooks/  G10_run.ipynb
splits/     dev_split.csv  test_split.csv
src/        data.py  models.py  losses.py  train.py  eval.py
results/    runs/<arm>_<depth>_f<fold>_s<seed>/
```

Still to come: `notebooks/G10_analysis.ipynb` (reads every `preds.csv`, builds
`results/results.csv` — the report's table, owned by one person) and
`notebooks/G10_data.ipynb` (dataset figures).

## Dates

Proposal 25 September 2026. Final report 4 December 2026.
