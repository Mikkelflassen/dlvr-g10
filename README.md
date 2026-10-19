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

Every trained arm (E0–E3) is scored at threshold 0.5. E0b is E0 scored at the
threshold with the best F1 on inner validation. So E0 and E0b share AUROC,
AUPRC and ECE and differ only in recall, precision and F1 — E0b shows how much
of an arm's gain a threshold move alone would give. `eval.build_results` adds
the E0b rows automatically.

## Rules

1. **`splits/dev_split.csv`, `splits/test_split.csv` and `splits/inner_split.csv`
   are canonical.** Never regenerate them. `inner_split.csv` (frozen 17 Oct 2026)
   fixes the inner validation images per held-out fold, independent of the seed
   and the scikit-learn version; it reproduces every earlier seed-0 run exactly. If a split assertion fails, the answer is to use the saved
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
3. Edit cell 4 — `ARM`, `FOLD`, `SEED`, `TAG`, `FOLDER`, `AUG` — and nothing
   else. For a real grid run leave `TAG`, `FOLDER` and `AUG` empty. For a test
   run set `TAG` (e.g. `"pw15"`) and `FOLDER`, the experiment it belongs to
   (e.g. `"e1_pos_weight"`); it is saved in `results/tuning/<FOLDER>/` and the
   folder's `summary.md` table is rebuilt and pushed with it.
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

`results/tuning/` and `results/runs/` are empty: all pilot runs were deleted on
5 October when the inner validation split and the epoch count changed. They are
still in the git history.

What the pilots showed (`E0_full`, fold 0, T4 GPU):

- Validation AUROC rises until about epoch 4, is flat (0.85–0.87) through
  epoch 8, and drops after that while training loss keeps falling. Hence
  8 epochs for every arm in `arms.json`.
- One epoch of full fine-tuning takes about 1.7 minutes, so a run is ~13.5 min.
- Inner validation has ~80 malignant images, so a single run's AUROC still moves
  by about ±0.02. Compare augmentation variants on at least two folds or seeds.

E3 uses `torchvision.ops.sigmoid_focal_loss`. Its two knobs are set per arm in
`arms.json`: `gamma` (focusing, default 2.0) and optionally `alpha` (class
weight, default -1 = off).

`pos_weight` (E1) is set to 21, chosen on inner validation of fold 0 with full
fine-tuning (`results/tuning/e1_pos_weight/summary.md`): the highest weight at
which each extra melanoma found costs at most about 10 extra false alarms.
`gamma: 2.0` without `alpha` (E3) was chosen on 17 Oct 2026 (`results/tuning/e3_gamma/`).
E2 samples 15% malignant images per batch (`mal_frac: 0.15`), chosen on 17 Oct 2026
from 10/15/20/25/50% on folds 0-2 (`results/tuning/e2_sampling/`): 15% and 20% were in
the top 3 for AUPRC on every fold, and 15% had the best AUROC and the fewest false
alarms to find 50% and 80% of melanomas. It is close to square-root sampling
(Kang et al., 2020, https://arxiv.org/abs/1910.09217), which here gives ~12%.

Budget: ~13.5 minutes per full run and ~8 per frozen run, 40 runs, so roughly
7 GPU hours split three ways.

## Augmentation test

Before the grid, one augmentation is chosen and locked. `make_transform` in
`src/data.py` has a ladder of five steps; each keeps everything before it:

| `AUG` | Adds | Why |
|-------|------|-----|
| `none` | nothing | the control: what augmentation adds at all |
| `base` | crop + flips | lesions appear at different scales |
| `geo` | transpose, shift / scale / rotate | a lesion has no "up" |
| `light` | brightness, contrast | cameras and lighting differ |
| `colour` | hue, saturation | tests whether colour should be left alone |
| `cutout` | one erased square | hair, rulers and ink cover parts of a lesion |
| `randaug` | base + RandAugment (not on the ladder) | a generic automatic policy, as a contrast |

Operations and strengths from `geo` upwards are those of the ISIC 2020 winning
solution (Ha et al., 2020), implemented with torchvision.

Protocol:

1. `E0_full` only, so the choice cannot favour one of the treatments.
2. Folds 0 and 1 for every step, `TAG` = the step name.
3. Compare the mean best inner-validation AUROC, and the `val_loss` curves in
   `history.csv` for overfitting. The held-out fold is not used for this choice.
4. Pick the simplest step whose mean AUROC is within noise (±0.02) of the best.
5. Lock it: change the default `aug` in `src/train.py`, then leave it alone.

Round 1 (in `results/tuning/aug_round1/`, `E0_full`, folds 0 and 1): best
inner AUROC base 0.862 / 0.863, geo 0.880 / 0.856, light 0.881 / 0.868,
colour 0.871 / 0.862. All within noise of each other. Training loss rose with
each step, but that is measured on augmented images, so it could not show
whether overfitting really fell.

Round 2 adds `none` and logs `train_clean_loss` / `train_clean_auroc`: a fixed
sample of training images (same size as inner validation) scored each epoch
*without* augmentation. The gap between those and `val_loss` / `val_auroc` is
the overfitting measure.
Round 2 runs are in `results/tuning/aug_round2/`.

### Decision (11 October 2026): `light` is locked

Best inner-validation AUROC, `E0_full`, seed 0:

| `AUG` | Fold 0 | Fold 1 | Mean |
|-------|--------|--------|------|
| `none` | 0.853 | 0.849 | 0.851 |
| `base` | 0.862 | 0.863 | 0.863 |
| `geo` | 0.880 | 0.856 | 0.868 |
| `light` | 0.881 | 0.868 | 0.875 |
| `colour` | 0.871 | 0.862 | 0.867 |
| `cutout` | 0.877 | 0.856 | 0.867 |
| `randaug` | 0.869 | 0.865 | 0.867 |

Bootstrap uncertainty of a single value is about ±0.015–0.02.

- Without augmentation the model memorises the training set: AUROC on clean
  training images reaches 1.000 by epoch 4 while validation AUROC falls from
  0.85 to 0.54–0.75. Its best score only exists because epoch 1 is kept.
- All augmented steps train stably for 8 epochs and are within noise of each
  other. Colour, cutout and RandAugment (which also changes colour strongly) do
  not help.
- `light` was chosen because it had the highest AUROC on both folds and leaves
  lesion colour unchanged. The steps cannot be separated statistically, so this
  is a judgement call. Not tested: whether the choice also suits E1–E3, in
  particular E2, whose sampler repeats each malignant image ~27 times per epoch.

Every grid run uses `light` (the default in `src/train.py`); leave `AUG` empty.

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
configs/    arms.json  m12.json
notebooks/  G10_run.ipynb  G10_grid.ipynb  G10_e2_sampling.ipynb  G10_analysis.ipynb  G10_m12.ipynb
            G10_panderm.ipynb  G10_isic2019_check.ipynb
splits/     dev_split.csv  test_split.csv  inner_split.csv
src/        data.py  models.py  losses.py  train.py  eval.py  make_inner_split.py
            panderm.py  check_institutions.py  isic2019.py
results/    runs/<arm>_<depth>_f<fold>_s<seed>/          the grid
            tuning/<experiment>/<arm>_<depth>_f<fold>_s<seed>_<tag>/  test runs,
                   one folder per experiment, each with a summary.md table
```

`notebooks/G10_grid.ipynb` runs the 40 grid runs as a queue (resumes after a
disconnect). `notebooks/G10_m12.ipynb` runs the extension (M12): one base
configuration with one change at a time from `configs/m12.json` (384 px images,
EfficientNet-B0), results in `results/m12/`. `notebooks/G10_panderm.ipynb` is the
M12 step with PanDerm_Base as a frozen encoder plus a linear probe (`src/panderm.py`),
also reported without the University of Queensland images, which may overlap with
PanDerm's pretraining data (`src/check_institutions.py`). The M12 step `isic2019`
adds ISIC 2019 images to training only; `notebooks/G10_isic2019_check.ipynb` must run
first and removes every ISIC 2019 image that duplicates an ISIC 2020 image (`src/isic2019.py`). `notebooks/G10_analysis.ipynb` reads every `preds.csv` in
`results/runs/`, builds `results/results.csv` (the report's table, owned by one
person), applies the Phase 3 decision rules and draws the report figures into
`results/figures/`. No GPU needed. Still to come: `notebooks/G10_data.ipynb`
(dataset figures).

## Sources

The code is built from library calls and from published code that we adapted.
Each function names its source in a comment; this is the full list.

**Code we adapted**

| Where | Source | Licence |
|-------|--------|---------|
| `models.py`; training loop and augmentation base in `train.py`, `data.py` | S. Chilamkurthy, [Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) ([code](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py)) | BSD-3-Clause |
| `ISICDataset`, `set_seed`, `val_epoch` / `predict`, fold handling, best epoch by AUROC, augmentation steps `geo`–`cutout` (operations and strengths) | Ha, Liu & Liu (2020), [ISIC 2020 1st-place solution](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution), [paper](https://arxiv.org/abs/2010.05351) | MIT |
| `ece` in `eval.py` | G. Pleiss, [temperature_scaling](https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py), code for Guo et al. (2017) | MIT |
| Balanced sampler weights in `data.py` | ptrblck, PyTorch forum, [How to handle imbalanced classes](https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264) | forum post |

**Libraries called directly**

| What | Function |
|------|----------|
| Focal loss (E3) | [`torchvision.ops.sigmoid_focal_loss`](https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py) |
| Weighted BCE (E1) | [`torch.nn.BCEWithLogitsLoss(pos_weight=...)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) |
| Balanced sampling (E2) | [`torch.utils.data.WeightedRandomSampler`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) |
| Inner validation split | [`sklearn.model_selection.StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) |
| Threshold tuning | [`sklearn.metrics.precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) |
| AUROC, AUPRC, recall, precision, F1 | [`sklearn.metrics`](https://scikit-learn.org/stable/modules/model_evaluation.html) |
| Image download | [Kaggle API](https://www.kaggle.com/docs/api) |

**Methods and workflow**

- RandAugment: Cubuk et al. (2020), [arXiv:1909.13719](https://arxiv.org/abs/1909.13719); [`torchvision.transforms.RandAugment`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html).
- Cutout: DeVries & Taylor (2017), [arXiv:1708.04552](https://arxiv.org/abs/1708.04552).
- Focal loss: Lin et al. (2017), [arXiv:1708.02002](https://arxiv.org/abs/1708.02002). Focal loss and calibration: Mukhoti et al. (2020), [arXiv:2002.09437](https://arxiv.org/abs/2002.09437).
- Expected calibration error: Guo et al. (2017), [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- ResNet: He et al. (2016), [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Adam: Kingma & Ba (2015), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Dataset: Rotemberg et al. (2021), [doi:10.1038/s41597-021-00815-z](https://doi.org/10.1038/s41597-021-00815-z); 256×256 version by C. Deotte, [jpeg-melanoma-256x256](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256).
- Patient-grouped folds on ISIC 2020: C. Deotte, [Triple Stratified KFold](https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords).
- Separate data for model selection and for the final estimate (inner validation, held-out fold, locked test set): Cawley & Talbot (2010), [JMLR 11](https://jmlr.org/papers/v11/cawley10a.html).
- Fixed seeds, a simple baseline first, one change at a time, evaluating on the whole validation set: A. Karpathy (2019), [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).

## Dates

Proposal 25 September 2026. Final report 4 December 2026.
