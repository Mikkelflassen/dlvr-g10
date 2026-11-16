# Group 10: Finding melanomas in skin images

This is our course project in Deep Learning for Visual Recognition at Aarhus University, autumn 2026.

## The project

The network in this project looks for melanomas in the ISIC 2020 images, where only 1.8% of the images show a melanoma.

The question we ask: when the data are unbalanced, do the usual tricks make the network better, or do they only move the threshold?

1. **The grid.** Plain BCE is compared with three tricks on a ResNet-18, both frozen and fine-tuned, on 5 folds (40 runs). E5 and E6 were added later.
2. **The extension (M12).** Since the tricks helped very little, we tried to improve the network itself.
3. **The test set (M13).** In the end, three models were tested one time on our locked test set.

## Results

Our main score is AUPRC, which is 0.018 for random guessing. Unless we say something else, a number is the mean of the 5 held-out folds with full fine-tuning.

The rule we use: a change helps only when its mean AUPRC is above the base and it wins on at least 4 of 5 folds.

| Setup | Mean AUPRC | Folds better than the base | Helps? |
|---|---|---|---|
| E0: plain BCE (the base) | 0.159 | | |
| E1: weighted loss | 0.149 | 2 of 5 | no |
| E2: balanced sampling | 0.158 | 3 of 5 | no |
| E3: focal loss | 0.165 | 2 of 5 | no |
| E5: balanced sampling + focal loss | 0.149 | 2 of 5 | no |
| E6: BCE, then AUC-margin loss | 0.169 | 4 of 5 | yes, a little |
| 384 px images | 0.163 | 3 of 5 | no |
| EfficientNet-B0 | 0.169 | 2 of 5 | no |
| ResNet-18 + ISIC 2019 in training | 0.206 | 4 of 5 | yes |
| PanDerm, frozen | 0.184 | 4 of 5 | yes |
| PanDerm, frozen, + ISIC 2019 | 0.176 | 3 of 5 | no |
| PanDerm, fine-tuned | 0.264 | 5 of 5 | yes |
| PanDerm, fine-tuned, + ISIC 2019 | 0.322 | 5 of 5 | yes, our best |
| PanDerm, fine-tuned, + ISIC 2019 + age, sex, body site | 0.317 | 5 of 5 | yes |

What the tricks did was mostly to move the threshold. The largest gains came from real extra melanomas and from a network pretrained on skin images.

The test set (4,630 images, 82 melanomas), scored one time:

| Model | AUPRC (95% interval) | AUROC (95% interval) | Melanomas found | False alarms |
|---|---|---|---|---|
| PanDerm, fine-tuned, + ISIC 2019 | 0.291 (0.197 to 0.394) | 0.936 (0.914 to 0.954) | 71 of 82 | 561 |
| E6: BCE, then AUC-margin loss | 0.187 (0.113 to 0.289) | 0.883 (0.845 to 0.915) | 61 of 82 | 958 |
| E0: plain BCE | 0.154 (0.092 to 0.237) | 0.870 (0.829 to 0.905) | 62 of 82 | 1,002 |

`results/test/summary.md` has more details.

## The decisions we made

**Data and splits**

- **256 px images.** The resized dataset is 792 MB, which let us download it in each Colab session. The full one is about 108 GB.
- **All images of one patient stay together.** If they were spread out, the network could learn to recognise the person and not the lesion.
- **A locked test set from the start (14%).** This gives us a single fair test at the end, which had no influence on any choice.
- **Inner validation (20% of the training folds).** The best epoch and the settings have to be chosen on some data, and this is that data.
- **The inner split is saved in a file.** With a newer scikit-learn version, the same seed produced another split.

**How we score**

- **AUPRC as the main score.** Only 1.8% are melanomas, so always answering benign already gives 98% accuracy. AUPRC tells us whether the melanomas get high ranks.
- **False alarms to find 50% and 80% of the melanomas.** This number is easy to grasp, and no threshold is involved.
- **Rules before results.** We wrote the rule down first, because the differences are small and the folds vary a lot. That way we could not simply choose what looked good.
- **We save predictions, not models.** A model takes a lot of space. From the predictions, any metric can be computed later with no new training.

**Training**

- **ResNet-18.** Being small and fast, it made 40 runs possible.
- **Frozen and fully fine-tuned.** According to Kang et al., these tricks mostly affect the last layer, which is why we ran both.
- **8 epochs.** On fold 0, a pilot with 12 epochs was not better. That run was deleted on 5 October together with the other pilots, and now exists only in the git history.
- **We keep the epoch with the best inner validation AUROC.** Each setup has its own loss, which makes the loss values impossible to compare.
- **Augmentation `light`.** With no augmentation, the network memorised its training images. The augmented versions all scored close to each other. `light` came out best on both folds, and it leaves the colour alone, which is important for melanoma. We stopped there, since more tests would not have told the versions apart.

**The tricks**

- **E1: `pos_weight` 21.** Raising the weight finds more melanomas and also gives more false alarms. We kept raising it while each extra melanoma cost no more than about 10 false alarms. The cost was about 7 going from 15 to 21, and about 19 going from 21 to 27.
- **E2: 15% malignant per batch.** On every fold, 15% and 20% were among the 3 best, and 15% gave the fewest false alarms. With high shares, every melanoma is repeated many times and the network memorises them. To get started on the grid, we left 16-19% untested.
- **E3: `gamma` 2, no `alpha`.** All `alpha` did was shift the scores and lower AUPRC. It also overlaps with E1.
- **E5: E2 and E3 together.** The idea was to check whether the two help in combination. They did not.
- **E6: AUC-margin loss.** Because the tricks only moved the threshold, we tried a loss that rewards good ranking. It gave a small gain.

**The extension**

- **Change one thing at a time.** This shows us what caused each difference. The base is E0, since E5 did not pass our rule.
- **384 px and EfficientNet-B0.** Larger images and this network family were used by the ISIC 2020 winners. Our rule was not passed by either.
- **ISIC 2019.** E2 only repeats melanomas we already have, and we wanted new real ones. It is used for training only, which keeps the numbers comparable. Melanoma alone counts as 1, since that is what ISIC 2020 labels.
- **Duplicate check before using ISIC 2019.** If one photo were in both training and test, our results would look better than they are. The check removed 1,601 images. The limit is loose on purpose, since the only cost is a few training images.
- **PanDerm.** Its pretraining used skin images, not ImageNet. The frozen version came first because it is cheap. When it passed, we fine-tuned it too.
- **Scoring without Queensland and Sydney.** Those images may have been in PanDerm's pretraining. Leaving them out did not change the winner.
- **Age, sex and body site.** The winners used these. Images per patient and file size were left out, since they reveal the dataset an image belongs to. The metadata did not improve our best model.
- **Patient context ("ugly duckling"): not done.** According to the challenge organisers, it did not improve results (Kurtansky et al.).
- **Longer training.** To see whether 8 epochs was too short, we ran 50 epochs on fold 0, and Colab stopped the run after 29. By epoch 24 the network fitted its training images perfectly, while inner validation had peaked at epoch 10. The held-out AUPRC was 0.309, lower than the 0.333 of our 8-epoch run. Continuing was therefore not worth it.

**The test set**

- **Three models only.** They are our best model, the baseline, and E6, which was the only loss or sampling change to pass our rule.
- **Trained on all dev images, last epoch kept.** There is no validation data left to pick an epoch with. In the fold runs the best epoch was 5-7.
- **Threshold that finds 80% of the melanomas.** The goal is to catch most melanomas. "Find 80%" carried over from inner validation (74-86% found), and "best F1" did not (15-44%).

**What we know is not perfect**

- **We chose the best model on the held-out folds.** This makes 0.322 somewhat too high. The fair number is the test score, 0.291.
- **One seed per run.** There was not enough GPU time for more. Noise could explain the small differences between the tricks.
- **The learning rates are not tuned.** From the start we used 1e-3 (frozen) and 1e-4 (full), and for PanDerm its own value scaled to our batch size.
- **E6 trains longer.** With its 4 extra epochs it gets 12 in total.
- **Duplicates inside ISIC 2020.** The hash check was built after the grid, for ISIC 2019. Across our splits it found 32 pairs with the same hash, all benign, and 7 of them involve the test set. The scores stay the same when those 7 test images are removed.

## How we worked

All code lived in one GitHub repository, and training was done in Google Colab for its GPU. A notebook pulled the latest code from GitHub and, when a run was done, pushed the results back. Colab stops often, and this way no finished run was lost and each run is logged together with its settings.

- Within a run, choices were made on inner validation only.
- Setups were compared on the held-out folds.
- The test set was used one time, at the end (`results/test/protocol.md`).

## The setups

| Name | What it does | Setting |
|---|---|---|
| E0 | Plain BCE | |
| E0b | E0 with a tuned threshold | no new training |
| E1 | BCE with `pos_weight` | 21 |
| E2 | Balanced sampling | 15% malignant per batch |
| E3 | Focal loss | `gamma` 2, no `alpha` |
| E5 | E2 and E3 together | as above |
| E6 | BCE, then AUC-margin loss | 8 + 4 epochs |

Every setup uses ResNet-18 from ImageNet, 8 epochs, batch size 64 and the `light` augmentation.

## What is in this repo

```
notebooks/   the notebooks we ran in Colab
src/         the Python code
configs/     the settings for each run
splits/      our fixed splits
results/     runs/     the grid
             tuning/   our tests of settings and augmentation
             m12/      the extension
             test/     the final models, scores and protocol
             results.csv  scores of E0-E3
report/      make_report_figs.py, the script that makes the figures of the report
requirements.txt  the Python packages
```

A run folder contains `config.json` (settings), `history.csv` (one row per epoch) and `preds.csv` (predictions).

## The data

The repo does not contain the images.

| Data | Where |
|---|---|
| ISIC 2020, 256 px | [`cdeotte/jpeg-melanoma-256x256`](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256) |
| ISIC 2020, 384 px | [`cdeotte/jpeg-melanoma-384x384`](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-384x384) |
| ISIC 2019, 256 px | [`cdeotte/jpeg-isic2019-256x256`](https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-256x256) |
| PanDerm_Base weights | [Link from the PanDerm README](https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB) |

```
kaggle datasets download -d cdeotte/jpeg-melanoma-256x256
```

## How to check our work

No training is needed, since every result is saved in `results/`.

- `notebooks/G10_analysis.ipynb` builds the tables and figures of the grid. Start Jupyter from the `notebooks` folder. It needs no GPU and no images.
- The test set scores are in `results/test/summary.md`.
- `python report/make_report_figs.py . figures` builds the report figures from `results/`. `torch` and `torchmetrics` must be installed, but no GPU or images are needed.

The remaining notebooks show how the training was done. They depend on our GitHub repository and therefore cannot be run from these files.

| Notebook | What it does |
|---|---|
| `G10_run` | Trains one run |
| `G10_e2_sampling` | Finds the sampling share for E2 |
| `G10_grid` | Trains all grid runs |
| `G10_analysis` | Tables and figures for the grid |
| `G10_isic2019_check` | Removes duplicates from ISIC 2019 |
| `G10_panderm` | PanDerm as a frozen network |
| `G10_m12` | The extension |
| `G10_test` | The final models and the test set |

## Sources

Where we use code from others, a comment names the source. The full list is below.
Comments like `# Written with AI (Claude).` mark the more complex parts written with AI.

**Code we adapted**

| Where | Source | Licence |
|-------|--------|---------|
| `models.py`; training loop and augmentation base in `train.py`, `data.py` | S. Chilamkurthy, [Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) ([code](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py)) | BSD-3-Clause |
| `ISICDataset`, `set_seed`, `val_epoch` / `predict`, fold handling, best epoch by AUROC, augmentation steps `geo` to `cutout` (operations and strengths) | Ha, Liu & Liu (2020), [ISIC 2020 1st-place solution](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution), [paper](https://arxiv.org/abs/2010.05351) | MIT |
| `ece` in `eval.py` | G. Pleiss, [temperature_scaling](https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py), code for Guo et al. (2017) | MIT |
| Balanced sampler weights in `data.py` | ptrblck, PyTorch forum, [How to handle imbalanced classes](https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264) | forum post |
| Metadata features and network in `meta.py`; reading ISIC 2019 in `isic2019.py` | Ha, Liu & Liu (2020), same repository (`get_meta_data`, `Effnet_Melanoma`, `get_df`) | MIT |
| Model definition, weight loading, layer-wise learning rates and cosine schedule in `panderm.py` | Yan et al. (2025), [PanDerm](https://github.com/SiyuanYan1/PanDerm); parts build on [BEiT](https://github.com/microsoft/unilm/tree/master/beit) | CC-BY-NC-ND 4.0 (PanDerm), MIT (BEiT) |
| `auc_stage` in `train.py` | [LibAUC tutorial](https://github.com/Optimization-AI/LibAUC/blob/main/examples/02_v140_Optimizing_AUROC_with_ResNet20_on_Imbalanced_CIFAR10.ipynb) | MIT |

**Course material**

Comments such as `# Based on lab 2` or `# See also lecture 6, Code 5` point to the labs and lecture notes of
[Deep Learning for Visual Recognition 2026](https://github.com/klaverhenrik/Deep-Learning-for-Visual-Recognition-2026):
the training loop (lecture 2, Lab 2), BCE loss (lectures 2 and 3), ResNet-18 transfer learning (lectures 4 and 7),
Dataset pattern and duplicate check (Lab 3), metrics (bonus lecture, week 4), optimisers, schedules, augmentation and
best-epoch saving (lecture 6, Lab 7), the loss on clean training images (Lab 7), the plots of the report (Lab 4a), focal loss (lecture 8), checkpoints (Lab 6).

**Libraries called directly**

| What | Function |
|------|----------|
| Focal loss (E3) | [`torchvision.ops.sigmoid_focal_loss`](https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py) |
| Weighted BCE (E1) | [`torch.nn.BCEWithLogitsLoss(pos_weight=...)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) |
| Balanced sampling (E2) | [`torch.utils.data.WeightedRandomSampler`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) |
| Inner validation split | [`sklearn.model_selection.StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) |
| Threshold tuning | [`sklearn.metrics.precision_recall_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) |
| AUROC, AUPRC, recall, precision, F1 | [`sklearn.metrics`](https://scikit-learn.org/stable/modules/model_evaluation.html) |
| AUC-margin loss (E6) | [LibAUC](https://libauc.org) `AUCMLoss`, `PESG` |
| Duplicate check | [imagehash](https://github.com/JohannesBuchner/imagehash) `phash` |
| Mixed precision | [`torch.autocast`, `torch.amp.GradScaler`](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) |
| Image download | [Kaggle API](https://www.kaggle.com/docs/api) |

**Methods and workflow**

- RandAugment: Cubuk et al. (2020), [arXiv:1909.13719](https://arxiv.org/abs/1909.13719); [`torchvision.transforms.RandAugment`](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html).
- Cutout: DeVries & Taylor (2017), [arXiv:1708.04552](https://arxiv.org/abs/1708.04552).
- Focal loss: Lin et al. (2017), [arXiv:1708.02002](https://arxiv.org/abs/1708.02002). Focal loss and calibration: Mukhoti et al. (2020), [arXiv:2002.09437](https://arxiv.org/abs/2002.09437).
- Deep AUC maximisation: Yuan et al. (2021), [arXiv:2012.03173](https://arxiv.org/abs/2012.03173). PanDerm: Yan et al. (2025), [arXiv:2410.15038](https://arxiv.org/abs/2410.15038).
- Tricks act on the classifier, progressive sampling: Kang et al. (2020), [arXiv:1910.09217](https://arxiv.org/abs/1910.09217).
- Patient context in ISIC 2020: Kurtansky et al., results of the 2020 SIIM-ISIC challenge.
- Patient-level bootstrap: Efron & Tibshirani (1993), An Introduction to the Bootstrap; Field & Welsh (2007), [doi:10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x).
- Expected calibration error: Guo et al. (2017), [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- ResNet: He et al. (2016), [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Adam: Kingma & Ba (2015), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Dataset: Rotemberg et al. (2021), [doi:10.1038/s41597-021-00815-z](https://doi.org/10.1038/s41597-021-00815-z); 256×256 version by C. Deotte, [jpeg-melanoma-256x256](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256).
- Patient-grouped folds on ISIC 2020: C. Deotte, [Triple Stratified KFold](https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords).
- Separate data for model selection and for the final estimate (inner validation, held-out fold, locked test set): Cawley & Talbot (2010), [JMLR 11](https://jmlr.org/papers/v11/cawley10a.html).
- Fixed seeds, a simple baseline first, one change at a time, evaluating on the whole validation set: A. Karpathy (2019), [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).
