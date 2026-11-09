# Group 10: Finding melanomas in skin images

This is our course project in Deep Learning for Visual Recognition at Aarhus University, autumn 2026.

## What we did

We train a network to tell benign and malignant skin lesions apart, using the ISIC 2020 images.
Only about 1.8% of the images are malignant, so the two classes are very unbalanced.

Our main question was this: do the usual tricks for unbalanced data help the network separate the
two classes better, or do they only move the decision threshold?

The project has three parts:

1. **The grid.** We compared plain BCE with three tricks (weighted loss, balanced sampling and focal
   loss). We used ResNet-18, both frozen and fully fine-tuned, on 5 folds. That is 40 runs.
   Afterwards we added two more setups, E5 and E6.
2. **The extension.** The tricks did not help much, so we tried to make the network itself better.
   We changed one thing at a time. In the code this part is called M12.
3. **The test set.** At the end we opened our locked test set once, for three models.
   In the code this part is called M13.

## What we found

Our main score is AUPRC, the area under the precision-recall curve. Random guessing gives about 0.018.

The numbers below are the mean over the 5 held-out folds, with full fine-tuning unless we say otherwise.
We decided on a rule before we looked at the results: a change only helps if its mean AUPRC is above
the base and it is better on at least 4 of the 5 folds.

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
| ResNet-18 with ISIC 2019 images added to training | 0.206 | 4 of 5 | yes |
| PanDerm, frozen | 0.184 | 4 of 5 | yes |
| PanDerm, frozen, + ISIC 2019 | 0.176 | 3 of 5 | no |
| PanDerm, fine-tuned | 0.264 | 5 of 5 | yes |
| PanDerm, fine-tuned, + ISIC 2019 | 0.322 | 5 of 5 | yes, our best model |
| PanDerm, fine-tuned, + ISIC 2019 + age, sex and body site | 0.317 | 5 of 5 | yes |

So the three tricks mostly moved the threshold. The big gains came from real extra melanomas
(ISIC 2019) and from a network pretrained on skin images (PanDerm). We also tried longer training
for the best model on fold 0. Colab stopped after 29 of the planned 50 epochs, and it did not help.

On the locked test set (4,630 images, 82 melanomas) we got the numbers below. We fixed each model's
threshold before we opened the test set, at the score that finds 80% of the melanomas in its held-out predictions.

| Model | AUPRC (95% interval) | AUROC (95% interval) | Melanomas found | False alarms |
|---|---|---|---|---|
| PanDerm, fine-tuned, + ISIC 2019 | 0.291 (0.197 to 0.394) | 0.936 (0.914 to 0.954) | 71 of 82 | 561 |
| E6: BCE, then AUC-margin loss | 0.187 (0.113 to 0.289) | 0.883 (0.845 to 0.915) | 61 of 82 | 958 |
| E0: plain BCE | 0.154 (0.092 to 0.237) | 0.870 (0.829 to 0.905) | 62 of 82 | 1,002 |

PanDerm may have been pretrained on some of the same patients as ISIC 2020. So we also scored the
models without the Queensland and Sydney images. The order stayed the same (AUPRC 0.320, 0.231 and 0.171).

The full details are in `results/test/summary.md`.

## How we worked

- **Patients stay together.** All images from one patient are in the same split.
- **Splits.** 86% of the images are for development, in 5 folds. 14% are the locked test set.
  We use 20% of the training folds as inner validation. The split files are in `splits/` and we never
  made them again.
- **Choices.** Inside a run we only used inner validation, for example to choose the best epoch and
  the settings. We used the held-out folds to compare the setups, and the test set only once at the end.
- **One change at a time**, so we know what caused a difference.
- **Rules first.** We wrote our decision rules before we ran the experiments. For the test set we
  wrote `results/test/protocol.md` before we opened it.
- **Metrics from saved files.** Every run saves its predictions in `preds.csv`, and all metrics are
  calculated from those files.

## How we used GitHub and Colab

We kept all our code in one GitHub repository and trained the networks in Google Colab, because Colab gives
us a GPU.

Each notebook first got the newest code from GitHub. When a run was finished, the notebook pushed the results
back to the repository. This gave us three things:

- **Nothing was lost.** Colab sessions often stop, but finished runs were already saved. Long runs also
  pushed their progress after every epoch.
- **Every run is logged.** Each run folder has the settings that were used and what came out of the run.
- **Everyone saw the same results.** We all used the files in `results/`.

## The setups

| Name | What it does | Setting we chose |
|---|---|---|
| E0 | Plain BCE, our baseline | |
| E0b | E0 with a tuned threshold, no new training | |
| E1 | BCE with `pos_weight` | 21 |
| E2 | Balanced sampling | 15% malignant images per batch |
| E3 | Focal loss | `gamma` 2, no `alpha` |
| E5 | E2 and E3 together | as above |
| E6 | BCE first, then AUC-margin loss | 8 + 4 epochs |

All setups use ResNet-18 pretrained on ImageNet, 8 epochs and batch size 64. We tested the settings
on inner validation (`results/tuning/`). For augmentation we tested a ladder of steps and chose `light`
(crop, flips, small rotations, brightness and contrast). It had the best inner validation AUROC, but
the steps were very close. In the analysis, every run is also scored with a tuned threshold (E0b, E1b and so on).

## What is in this repo

```
notebooks/   the notebooks we run in Colab
src/         the Python code (data, models, losses, training, metrics)
configs/     the settings for each run
splits/      our fixed train, validation and test splits
results/     runs/    the grid (E0-E3, E5, E6)
             tuning/  our tests of settings and augmentation
             m12/     the extension
             test/    the final models, the test set scores and our test set protocol
             figures/ figures from the notebooks
             results.csv  scores of the main grid (E0-E3)
requirements.txt  the Python packages we used
```

Each run has its own folder with three files: `config.json` (the settings), `history.csv` (loss and
AUROC per epoch) and `preds.csv` (the prediction for each image). We do not keep the trained models.

## The data

The images are not in this zip. We used these datasets:

| Data | Where | Used for |
|---|---|---|
| ISIC 2020, 256×256 (792 MB) | [`cdeotte/jpeg-melanoma-256x256`](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256) | all runs |
| ISIC 2020, 384×384 | [`cdeotte/jpeg-melanoma-384x384`](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-384x384) | the 384 px step |
| ISIC 2019, 256×256 | [`cdeotte/jpeg-isic2019-256x256`](https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-256x256) | extra training images |
| PanDerm_Base weights | [Google Drive link from the PanDerm README](https://drive.google.com/file/d/17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB) | the PanDerm steps |

With the Kaggle tool, one dataset is downloaded like this:

```
kaggle datasets download -d cdeotte/jpeg-melanoma-256x256
```

ISIC 2020 has 33,126 images, and 32,693 of them are in our splits after removing duplicates.
The Python packages we used are listed in `requirements.txt`.

## How to check our work

These files are for reading and checking. Nothing needs to be trained, because all our results are saved in
`results/`. You do not need a GPU or the images for this:

- `notebooks/G10_analysis.ipynb` makes the tables and figures for the grid. Start Jupyter in the `notebooks` folder.
- `results/test/summary.md` has the final test set scores, and each run folder has its predictions.

The other notebooks show how we trained the networks. They were run in Google Colab with our GitHub
repository, so they cannot be run from these files.

| Notebook | What it does |
|---|---|
| `G10_run` | Trains one run, for example to test a setting |
| `G10_e2_sampling` | Finds the sampling share for E2 |
| `G10_grid` | Trains all the grid runs in a queue |
| `G10_analysis` | Makes the tables and figures for the grid (no GPU needed) |
| `G10_isic2019_check` | Finds and removes ISIC 2019 images that are also in ISIC 2020 |
| `G10_panderm` | Tests PanDerm as a frozen network |
| `G10_m12` | Runs the steps of the extension |
| `G10_test` | Trains the three final models and scores the test set, once |

## Sources

The code is built from library calls and from published code that we adapted.
Code taken from other people names its source in a comment; this is the full list.
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
- Patient-level bootstrap: Efron & Tibshirani (1993), An Introduction to the Bootstrap; Field & Welsh (2007), [doi:10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x).
- Expected calibration error: Guo et al. (2017), [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
- ResNet: He et al. (2016), [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Adam: Kingma & Ba (2015), [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Dataset: Rotemberg et al. (2021), [doi:10.1038/s41597-021-00815-z](https://doi.org/10.1038/s41597-021-00815-z); 256×256 version by C. Deotte, [jpeg-melanoma-256x256](https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256).
- Patient-grouped folds on ISIC 2020: C. Deotte, [Triple Stratified KFold](https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords).
- Separate data for model selection and for the final estimate (inner validation, held-out fold, locked test set): Cawley & Talbot (2010), [JMLR 11](https://jmlr.org/papers/v11/cawley10a.html).
- Fixed seeds, a simple baseline first, one change at a time, evaluating on the whole validation set: A. Karpathy (2019), [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).
