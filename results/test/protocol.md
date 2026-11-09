# Our test set protocol

We wrote this on 2 November 2026, before we opened the test set.

The locked test set is `splits/test_split.csv`. It has 4,630 images. We made it once on 22 September 2026 and did not use it until this final test.

## Models

We only score these three models on the test set.

| Name | What it is | Why we test it |
|---|---|---|
| `winner` | PanDerm_Base fully fine-tuned, with ISIC 2019 added to training | our best step in the extension: AUPRC 0.322, better than the base on 5 of 5 folds |
| `E0` | ResNet-18, plain BCE | our baseline (E0b is E0 with a threshold) |
| `E6` | ResNet-18, BCE then AUC-margin loss | the only change of loss or sampling that passed our rule (0.169, 4 of 5 folds) |

## Training

- We train each model once on all 28,063 dev images. The winner also trains on the 23,671 ISIC 2019 images that passed the duplicate check. Each model uses the same settings as its fold runs, with seed 0.
- There is no validation set left, so we keep the last epoch: 8 epochs (E6: 8 + 4 AUC-margin epochs). The fold runs peaked at epochs 5-7, and our long training check on fold 0 showed that longer training does not help.

## Threshold

- For each model we use the score at which 80% of the melanomas are found in the pooled held-out predictions of its 5 fold runs (499 melanomas). The values are in `configs/test.json` (winner 0.0181, E0 0.0112, E6 0.1222).
- We chose this rule because "best F1" on about 80 inner-validation melanomas did not transfer well (15-44% found on the held-out folds), while "find 80%" transferred to 74-86%.
- One thing to be careful about: the final model is trained again on all dev images, so its scores may be a little different from the fold models.

## What we report (`src/test_eval.py`)

- Main results: AUPRC and AUROC with 95% confidence intervals (bootstrap over patients, 2,000 resamples).
- The differences winner - E0 and E6 - E0, with their 95% intervals.
- At the threshold: melanomas found, false alarms, malignant and benign accuracy, and how many malignant calls were right.
- Calibration (ECE), and AUPRC without the Queensland and Sydney images, because of possible overlap with PanDerm's pretraining.

## Rules

- We open the test set once. `notebooks/G10_test.ipynb` skips a model if its test predictions exist and never trains it again.
- We do not tune anything after seeing the result, and we report every number, whatever it shows.
