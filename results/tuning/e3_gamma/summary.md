# e3_gamma

Inner validation, threshold 0.5 (an image is called malignant when the network's output is 0.5 or more).

| gamma | alpha | fold | melanomas found | malignant accuracy | false alarms | benign accuracy | malignant guesses that are right | benign guesses that are right |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.95 | 0 | 23 of 79 | 29.1% | 184 | 95.7% | 23 of 207 (11.1%) | 4,092 of 4,148 (98.6%) |
| 1 | None | 0 | 0 of 79 | 0.0% | 0 | 100.0% | 0 of 0 (0.0%) | 4,276 of 4,355 (98.2%) |
| 2 | 0.9 | 0 | 19 of 79 | 24.1% | 120 | 97.2% | 19 of 139 (13.7%) | 4,156 of 4,216 (98.6%) |
| 2 | 0.93 | 0 | 13 of 79 | 16.5% | 105 | 97.5% | 13 of 118 (11.0%) | 4,171 of 4,237 (98.4%) |
| 2 | 0.95 | 0 | 40 of 79 | 50.6% | 316 | 92.6% | 40 of 356 (11.2%) | 3,960 of 3,999 (99.0%) |
| 2 | 0.97 | 0 | 10 of 79 | 12.7% | 49 | 98.9% | 10 of 59 (16.9%) | 4,227 of 4,296 (98.4%) |
| 2 | 0.98 | 0 | 66 of 79 | 83.5% | 1062 | 75.2% | 66 of 1128 (5.9%) | 3,214 of 3,227 (99.6%) |
| 2 | None | 0 | 0 of 79 | 0.0% | 0 | 100.0% | 0 of 0 (0.0%) | 4,276 of 4,355 (98.2%) |
