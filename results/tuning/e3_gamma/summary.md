# e3_gamma

Inner validation, threshold 0.5 (an image is called malignant when the network's output is 0.5 or more).

| alpha | fold | melanomas found | malignant accuracy | false alarms | benign accuracy | malignant guesses that are right | benign guesses that are right |
|---|---|---|---|---|---|---|---|
| 0.95 | 0 | 40 of 79 | 50.6% | 316 | 92.6% | 40 of 356 (11.2%) | 3,960 of 3,999 (99.0%) |
| 0.97 | 0 | 10 of 79 | 12.7% | 49 | 98.9% | 10 of 59 (16.9%) | 4,227 of 4,296 (98.4%) |
| None | 0 | 0 of 79 | 0.0% | 0 | 100.0% | 0 of 0 (0.0%) | 4,276 of 4,355 (98.2%) |
