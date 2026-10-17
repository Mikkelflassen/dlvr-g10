# e2_sampling

Inner validation, threshold 0.5 (an image is called malignant when the network's output is 0.5 or more).

| mal_frac | fold | melanomas found | malignant accuracy | false alarms | benign accuracy | malignant guesses that are right | benign guesses that are right |
|---|---|---|---|---|---|---|---|
| 0.1 | 0 | 18 of 79 | 22.8% | 82 | 98.1% | 18 of 100 (18.0%) | 4,194 of 4,255 (98.6%) |
| 0.25 | 0 | 37 of 79 | 46.8% | 338 | 92.1% | 37 of 375 (9.9%) | 3,938 of 3,980 (98.9%) |
| 0.5 | 0 | 35 of 79 | 44.3% | 329 | 92.3% | 35 of 364 (9.6%) | 3,947 of 3,991 (98.9%) |
