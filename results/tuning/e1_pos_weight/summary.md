# e1_pos_weight

Inner validation, threshold 0.5 (an image is called malignant when the network's output is 0.5 or more).

| pos_weight | seed | fold | melanomas found | malignant accuracy | false alarms | benign accuracy | malignant guesses that are right | benign guesses that are right |
|---|---|---|---|---|---|---|---|---|
| 7 | 0 | 0 | 12 of 79 | 15.2% | 49 | 98.9% | 12 of 61 (19.7%) | 4,227 of 4,294 (98.4%) |
| 15 | 0 | 0 | 34 of 79 | 43.0% | 235 | 94.5% | 34 of 269 (12.6%) | 4,041 of 4,086 (98.9%) |
| 21 | 0 | 0 | 42 of 79 | 53.2% | 290 | 93.2% | 42 of 332 (12.7%) | 3,986 of 4,023 (99.1%) |
| 27 | 0 | 0 | 52 of 79 | 65.8% | 481 | 88.8% | 52 of 533 (9.8%) | 3,795 of 3,822 (99.3%) |
| 37 | 0 | 0 | 55 of 79 | 69.6% | 598 | 86.0% | 55 of 653 (8.4%) | 3,678 of 3,702 (99.4%) |
| 37 | 1 | 0 | 60 of 101 | 59.4% | 611 | 85.6% | 60 of 671 (8.9%) | 3,624 of 3,665 (98.9%) |
| 57 | 0 | 0 | 41 of 79 | 51.9% | 328 | 92.3% | 41 of 369 (11.1%) | 3,948 of 3,986 (99.0%) |
