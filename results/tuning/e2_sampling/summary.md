# e2_sampling

Inner validation, threshold 0.5 (an image is called malignant when the network's output is 0.5 or more).

| mal_frac | sampling | fold | melanomas found | malignant accuracy | false alarms | benign accuracy | malignant guesses that are right | benign guesses that are right |
|---|---|---|---|---|---|---|---|---|
| 0.1 | None | 0 | 18 of 79 | 22.8% | 82 | 98.1% | 18 of 100 (18.0%) | 4,194 of 4,255 (98.6%) |
| 0.15 | fixed | 0 | 10 of 79 | 12.7% | 43 | 99.0% | 10 of 53 (18.9%) | 4,233 of 4,302 (98.4%) |
| 0.2 | fixed | 0 | 10 of 79 | 12.7% | 61 | 98.6% | 10 of 71 (14.1%) | 4,215 of 4,284 (98.4%) |
| 0.25 | None | 0 | 37 of 79 | 46.8% | 338 | 92.1% | 37 of 375 (9.9%) | 3,938 of 3,980 (98.9%) |
| 0.5 | None | 0 | 35 of 79 | 44.3% | 329 | 92.3% | 35 of 364 (9.6%) | 3,947 of 3,991 (98.9%) |
| 0.1 | fixed | 1 | 9 of 74 | 12.2% | 28 | 99.4% | 9 of 37 (24.3%) | 4,629 of 4,694 (98.6%) |
| 0.15 | fixed | 1 | 7 of 74 | 9.5% | 20 | 99.6% | 7 of 27 (25.9%) | 4,637 of 4,704 (98.6%) |
| 0.2 | fixed | 1 | 25 of 74 | 33.8% | 142 | 97.0% | 25 of 167 (15.0%) | 4,515 of 4,564 (98.9%) |
| 0.25 | fixed | 1 | 17 of 74 | 23.0% | 121 | 97.4% | 17 of 138 (12.3%) | 4,536 of 4,593 (98.8%) |
| 0.5 | fixed | 1 | 28 of 74 | 37.8% | 190 | 95.9% | 28 of 218 (12.8%) | 4,467 of 4,513 (99.0%) |
