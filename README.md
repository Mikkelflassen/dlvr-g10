# Group 10: Finding melanomas in skin images

Course project in Deep Learning for Visual Recognition, Aarhus University, autumn 2026.

## The project

We train a network to find melanomas in the ISIC 2020 images. Only 1.8% of the images are melanomas.

Our question: do the usual tricks for unbalanced data make the network better, or do they only move the threshold?

1. **The grid.** Plain BCE against three tricks, on ResNet-18, frozen and fine-tuned, on 5 folds (40 runs). Later we added E5 and E6.
2. **The extension (M12).** The tricks did not help much, so we tried to make the network itself better.
3. **The test set (M13).** At the end we tested three models once on our locked test set.

## Results

AUPRC is our main score. Random guessing gives 0.018. The numbers are the mean of 5 held-out folds, with full fine-tuning unless we say otherwise.

Our rule: a change only helps if its mean AUPRC is higher than the base and it is better on at least 4 of 5 folds.

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

The tricks mostly moved the threshold. Real extra melanomas and a network pretrained on skin images helped the most.

Test set (4,630 images, 82 melanomas), once:

| Model | AUPRC (95% interval) | AUROC (95% interval) | Melanomas found | False alarms |
|---|---|---|---|---|
| PanDerm, fine-tuned, + ISIC 2019 | 0.291 (0.197 to 0.394) | 0.936 (0.914 to 0.954) | 71 of 82 | 561 |
| E6: BCE, then AUC-margin loss | 0.187 (0.113 to 0.289) | 0.883 (0.845 to 0.915) | 61 of 82 | 958 |
| E0: plain BCE | 0.154 (0.092 to 0.237) | 0.870 (0.829 to 0.905) | 62 of 82 | 1,002 |

More details are in `results/test/summary.md`.

## The decisions we made

**Data and splits**

- **256 px images.** The full dataset is about 108 GB. The resized one is 792 MB, so we could download it in every Colab session.
- **All images of one patient stay together.** Otherwise the network could recognise the person instead of the lesion.
- **A locked test set from the start (14%).** We wanted one fair test at the end that never influenced any choice.
- **Inner validation (20% of the training folds).** We needed data to choose the best epoch and the settings.
- **The inner split is saved in a file.** A new scikit-learn version gave a different split with the same seed.

**How we score**

- **AUPRC as the main score.** With 1.8% melanomas, a network that always says benign gets 98% accuracy. AUPRC shows if melanomas are ranked high.
- **False alarms to find 50% and 80% of the melanomas.** It is easy to understand and does not depend on the threshold.
- **Rules before results.** The differences are small and the folds differ a lot, so we wrote our rule first. Then we could not just pick what looked good.
- **We save predictions, not models.** Models are big. With the predictions we can calculate any metric later without training again.

**Training**

- **ResNet-18.** It is small and fast, so 40 runs were possible.
- **Frozen and fully fine-tuned.** Kang et al. suggest these tricks mainly act on the last layer, so we tested both.
- **8 epochs.** A test with 12 epochs did not do better.
- **We keep the epoch with the best inner validation AUROC.** The loss differs between the setups, so it cannot be compared.
- **Augmentation `light`.** Without augmentation the network memorised the training images. All augmented versions were close. `light` was best on both folds and does not change the colour, which matters for melanoma. More tests would not have separated them, so we stopped.

**The tricks**

- **E1: `pos_weight` 21.** A higher weight finds more melanomas but gives more false alarms. We went up as long as one extra melanoma cost at most about 10 false alarms. From 15 to 21 it cost about 7, from 21 to 27 about 19.
- **E2: 15% malignant per batch.** 15% and 20% were in the top 3 on every fold, and 15% had the fewest false alarms. High shares repeat each melanoma many times, so the network memorises them. We skipped 16-19%, because we wanted to start the grid.
- **E3: `gamma` 2, no `alpha`.** `alpha` only moved the scores and lowered AUPRC, and it overlaps with E1.
- **E5: E2 and E3 together.** We wanted to see if they help when combined. They did not.
- **E6: AUC-margin loss.** The tricks only moved the threshold, so we tried a loss that rewards ranking. It helped a little.

**The extension**

- **Change one thing at a time.** Then we know what caused a difference. E5 did not pass our rule, so the base is E0.
- **384 px and EfficientNet-B0.** The ISIC 2020 winners used larger images and this network family. Neither passed our rule.
- **ISIC 2019.** E2 repeats the same melanomas, so we wanted real new ones. We only use it for training, so the numbers can still be compared. Only melanoma counts as 1, because that is what ISIC 2020 labels.
- **Duplicate check before using ISIC 2019.** The same photo in training and test would make our results look too good. We removed 1,601 images. We chose a loose limit, because it only costs a few training images.
- **PanDerm.** It is pretrained on skin images instead of ImageNet. We tried it frozen first, because that is cheap. It passed, so we also fine-tuned it.
- **Scoring without Queensland and Sydney.** PanDerm may have seen those images in its pretraining. It still won without them.
- **Age, sex and body site.** The winners used this. We left out images per patient and file size, because they show which dataset an image comes from. It did not help our best model.
- **Patient context ("ugly duckling"): not done.** The challenge organisers found it did not improve results (Kurtansky et al.).
- **Longer training.** We tested 50 epochs on fold 0 to see if 8 was too short. Colab stopped the run after 29 epochs. The network already knew its training images perfectly by epoch 24, inner validation peaked at epoch 10, and the held-out AUPRC (0.309) was below our 8-epoch run (0.333). So it was not worth continuing.

**The test set**

- **Three models only.** Our best model, the baseline, and E6, the only change of loss or sampling that passed our rule.
- **Trained on all dev images, last epoch kept.** No validation data is left. The fold runs peaked at epochs 5-7.
- **Threshold that finds 80% of the melanomas.** We want to catch most melanomas. "Best F1" did not carry over from inner validation (15-44% found), but "find 80%" did (74-86%).

**What we know is not perfect**

- **We chose the best model on the held-out folds.** So 0.322 is a bit too high. The test score, 0.291, is the fair one.
- **One seed per run.** We did not have GPU time for more. The small differences between the tricks may be noise.
- **Duplicates inside ISIC 2020.** We only built the hash check after the grid, for ISIC 2019. It found 32 pairs across our splits with the same hash, 7 of them with the test set, all benign. Without those 7 test images the scores do not change.

## How we worked

We kept all code in one GitHub repository and trained in Google Colab, because Colab has a GPU. Each notebook got the newest code from GitHub and pushed the results back when a run was done. Colab often stops, so this meant finished runs were never lost, and every run is logged with its settings.

- Inside a run, we only used inner validation to choose things.
- We used the held-out folds to compare setups.
- We used the test set once, at the end (`results/test/protocol.md`).

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

All use ResNet-18 from ImageNet, 8 epochs, batch size 64 and the `light` augmentation.

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
             figures/  figures from the notebooks
             results.csv  scores of E0-E3
requirements.txt  the Python packages
```

Each run folder has `config.json` (settings), `history.csv` (per epoch) and `preds.csv` (predictions).

## The data

The images are not included.

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

Nothing needs to be trained. All results are saved in `results/`.

- `notebooks/G10_analysis.ipynb` makes the grid tables and figures. Start Jupyter in the `notebooks` folder. No GPU or images needed.
- `results/test/summary.md` has the test set scores.

The other notebooks show how we trained. They need our GitHub repository, so they cannot run from these files.

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

Code from other people names its source in a comment. This is the full list.
Comments like `# Written with AI (Claude).` mark the more complex parts written with AI.

**Code we adapted**

| Where | Source | Licence |
|-------|--------|---------|
| `models.py`; training loop and augmentation base in `train.py`, `data.py` | S. Chilamkurthy, [Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) ([code](https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py)) | BSD-3-Clause |
| `ISICDataset`, `set_seed`, `val_epoch` / `predict`, fold handling, best epoch by AUROC, augmentation steps `geo`–`cutout` (operations and strengths) | Ha, Liu & Liu (2020), [ISIC 2020 1st-place solution](https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution), [paper](https://arxiv.org/abs/2010.05351) | MIT |
| `ece` in `eval.py` | G. Pleiss, [temperature_scaling](https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py), code for Guo et al. (2017) | MIT |
| Balanced sampler weights in `data.py` | ptrblck, PyTorch forum, [How to handle imbalanced classes](https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264) | forum post |
| Metadata features and network in `meta.py`; reading ISIC 2019 in `isic2019.py` | Ha, Liu & Liu (2020), same repository (`get_meta_data`, `Effnet_Melanoma`, `get_df`) | MIT |
| Model definition, weight loading, layer-wise learning rates and cosine schedule in `panderm.py` | Yan et al. (2025), [PanDerm](https://github.com/SiyuanYan1/PanDerm); parts build on [BEiT](https://github.com/microsoft/unilm/tree/master/beit) | CC-BY-NC-ND 4.0 (PanDerm), MIT (BEiT) |
| `auc_stage` in `train.py` | [LibAUC tutorial](https://github.com/Optimization-AI/LibAUC/blob/main/examples/02_v140_Optimizing_AUROC_with_ResNet20_on_Imbalanced_CIFAR10.ipynb) | MIT |

**Course material**

Comments like `# Based on lab 2` or `# See also lecture 6, Code 5` refer to the labs and lecture notes in
[Deep Learning for Visual Recognition 2026](https://github.com/klaverhenrik/Deep-Learning-for-Visual-Recognition-2026):
the training loop (lecture 2, Lab 2), BCE loss (lectures 2 and 3), ResNet-18 transfer learning (lectures 4 and 7),
Dataset pattern and duplicate check (Lab 3), metrics (bonus lecture, week 4), optimisers, schedules, augmentation and
best-epoch saving (lecture 6), focal loss (lecture 8), checkpoints (Lab 6).

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
