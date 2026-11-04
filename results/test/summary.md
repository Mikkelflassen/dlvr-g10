# Test set (opened once) — 4630 images, 82 melanomas, 309 patients

| Model | AUPRC [95% CI] | AUROC [95% CI] | Found / malignant | False alarms | ECE | AUPRC w/o Queensland+Sydney |
|---|---|---|---|---|---|---|
| winner | 0.291 [0.197, 0.394] | 0.936 [0.914, 0.954] | 71 / 82 | 561 | 0.008 | 0.320 |
| E0 | 0.154 [0.092, 0.237] | 0.870 [0.829, 0.905] | 62 / 82 | 1002 | 0.005 | 0.171 |
| E6 | 0.187 [0.113, 0.289] | 0.883 [0.845, 0.915] | 61 / 82 | 958 | 0.101 | 0.231 |

| Difference | Value [95% CI] | Resamples above 0 |
|---|---|---|
| winner - E0 auprc | +0.138 [+0.059, +0.218] | 100% |
| winner - E0 auroc | +0.066 [+0.037, +0.101] | 100% |
| E6 - E0 auprc | +0.033 [+0.009, +0.067] | 100% |
| E6 - E0 auroc | +0.013 [+0.000, +0.026] | 98% |
