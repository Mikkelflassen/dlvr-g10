# M13 test-set protocol — fixed 2 November 2026, before the test set was opened

Written down and pushed before any model saw the locked test set
(`splits/test_split.csv`, 4,630 images, made once on 22 Sep 2026 and never used since).
The git history shows this file and `configs/test.json` existing before any file in `results/test/`.

## Models (3)
| Name | What | Why it is on the test set |
|---|---|---|
| `winner` | PanDerm_Base fully fine-tuned, ISIC 2019 added to training | best M12 step: AUPRC 0.322, better than the base on 5/5 folds |
| `E0` | ResNet-18, plain BCE | the baseline (E0b = E0 with a threshold) |
| `E6` | ResNet-18, BCE then AUC-margin loss | the only loss/sampling change that passed the rule (0.169, 4/5 folds) |

No other model is scored on the test set.

## Training
- Each model is trained **once** on **all 28,063 dev images** (the winner also on the 23,671 ISIC 2019
  images that passed the duplicate check), with exactly the settings of its fold runs, seed 0.
- No validation set is left, so **the last epoch is kept**: 8 epochs (E6: 8 + 4 AUC-margin epochs).
  The fold runs peaked at epochs 5–7 and the 50-epoch check (fold 0) showed that longer training does not help.

## Threshold
- Rule: the score at which **80% of the melanomas are found** in the pooled out-of-fold predictions of the
  model's 5 fold runs (499 melanomas). Values in `configs/test.json` (winner 0.0181, E0 0.0112, E6 0.1222).
- Chosen because "best F1" on ~80 inner-validation melanomas transferred badly (15–44% found on held-out folds),
  while "find 80%" transferred to 74–86%.
- Caveat: the final model is retrained on all dev images, so its scores may be shifted slightly from the fold models'.

## What is reported (src/test_eval.py)
- Main: **AUPRC** and AUROC with 95% confidence intervals (bootstrap over patients, 2,000 resamples).
- Paired differences winner − E0 and E6 − E0, with their 95% intervals.
- At the threshold: melanomas found, false alarms, malignant / benign accuracy, malignant calls right.
- Calibration (ECE) and AUPRC without the Queensland and Sydney images (possible PanDerm pretraining overlap).

## Rules
- The test set is opened once: `notebooks/G10_test.ipynb` skips a model whose test predictions exist and never
  retrains one. Nothing is tuned after seeing the result; every number is reported, whatever it shows.
