"""Builds the report figures from the saved results. No training happens here.

Written with AI (Claude). The plotting is the code from the course labs
(https://github.com/klaverhenrik/Deep-Learning-for-Visual-Recognition-2026):
  - Lab 4a (lab_solutions/Lab4a_ImageClassification_Solution.ipynb): torchmetrics' .plot()
    for confusion matrices and precision-recall curves, and the matplotlib loss curves.
What we changed: several models share one figure (ax=...), there is one precision-recall curve
per model where the lab has one per class, AUROC replaces accuracy, and the test threshold
is the one from our protocol.

Run:  python report/make_report_figs.py <path to repo> <output folder>
The repo is all it needs, so no GPU and no images.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryPrecisionRecallCurve,
    MulticlassConfusionMatrix,
)

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

num_classes = 2
class_names = ["benign", "malignant"]


def held_out(folder, run):
    """The held-out predictions of all 5 folds in one table. In `run`, {f} marks where the fold goes."""
    preds = pd.concat([pd.read_csv(REPO / folder / run.format(f=f) / "preds.csv") for f in range(5)])
    return preds[preds.split == "held_fold"]


def to_lab_format(df):
    """Turns our preds.csv into the (y_score, y_true) tensors of Lab 4a. Malignant is column 1."""
    p = torch.tensor(df.prob.values, dtype=torch.float32)
    y_score = torch.stack([1 - p, p], dim=1)
    y_true = torch.tensor(df.target.values, dtype=torch.long)
    return y_score, y_true


def pr_curves(models, ax):
    """The precision-recall curve of Lab 4a, with one curve for each model on shared axes."""
    names = []
    for name, df in models.items():
        y_score, y_true = to_lab_format(df)
        pr_curve_metric = BinaryPrecisionRecallCurve()
        pr_curve_metric.update(y_score[:, 1], y_true)
        pr_curve_metric.plot(score=False, ax=ax)

        ap_metric = BinaryAveragePrecision()
        ap_metric.update(y_score[:, 1], y_true)
        names.append(f"{name} (AP {ap_metric.compute():.3f})")
    ax.legend(names, loc="upper right", fontsize=8)
    ax.set_title("")


def confusion_matrix(df, ax, title, threshold=None):
    """The confusion matrix of Lab 4a. With no threshold given, it uses the lab's argmax (= 0.5)."""
    y_score, y_true = to_lab_format(df)
    y_pred = y_score.argmax(dim=1) if threshold is None else (y_score[:, 1] >= threshold).long()
    confmat_metric = MulticlassConfusionMatrix(num_classes=num_classes, normalize="true")
    confmat_metric.update(y_pred, y_true)
    confmat_metric.plot(labels=class_names, ax=ax)
    ax.set_title(title, fontsize=10)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- Figure 1: the grid, full fine-tuning, the 5 held-out folds together ----
grid = {"Plain BCE (E0)": "E0", "Weighted BCE (E1)": "E1", "Balanced sampling (E2)": "E2",
        "Focal loss (E3)": "E3", "E2 + E3 (E5)": "E5", "AUC-margin (E6)": "E6"}
grid = {name: held_out("results/runs", code + "_full_f{f}_s0") for name, code in grid.items()}

fig, ax = plt.subplots(figsize=(6, 5))
pr_curves(grid, ax)
save(fig, "fig1a_grid_pr")

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, (name, df) in zip(axes.flat, grid.items()):
    confusion_matrix(df, ax, name)          # at threshold 0.5
plt.tight_layout()
save(fig, "fig1b_grid_confmat")


# ---- Figure 2: the main extension steps ----
steps = {"Plain BCE (E0)": ("results/runs", "E0_full_f{f}_s0"),
         "ResNet-18 + ISIC 2019": ("results/m12", "E0_full_f{f}_s0_isic2019"),
         "PanDerm, frozen": ("results/m12", "E0_frozen_f{f}_s0_panderm"),
         "PanDerm, fine-tuned": ("results/m12", "E0_full_f{f}_s0_panderm_ft"),
         "PanDerm, fine-tuned + ISIC 2019": ("results/m12", "E0_full_f{f}_s0_panderm_ft_isic2019")}
steps = {name: held_out(folder, run) for name, (folder, run) in steps.items()}
fig, ax = plt.subplots(figsize=(6, 5))
pr_curves(steps, ax)
save(fig, "fig2_extension_pr")


# ---- Figure 3: training length, loss curves from Lab 4a (fold 0) ----
runs = {"8 epochs (our setting)": "E0_full_f0_s0_panderm_ft_isic2019",
        "50-epoch schedule (stopped after 29)": "E0_full_f0_s0_panderm_ft_isic2019_e50"}
fig, all_axes = plt.subplots(2, 2, figsize=(10, 7.5))
for axes, (title, run) in zip(all_axes, runs.items()):
    h = pd.read_csv(REPO / "results/m12" / run / "history.csv")
    # train is the clean training sample, which means it has the same loss as validation
    history = {"train_loss": h.train_clean_loss, "val_loss": h.val_loss,
               "train_auroc": h.train_clean_auroc, "val_auroc": h.val_auroc}

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title(f"Loss curves: {title}")

    axes[1].plot(history["train_auroc"], label="train")
    axes[1].plot(history["val_auroc"], label="validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].legend()
    axes[1].set_title(f"AUROC curves: {title}")
plt.tight_layout()
save(fig, "fig3_training_curves")


# ---- Figure 4: locked test set ----
summary = json.load(open(REPO / "results/test/summary.json"))
test_runs = {"Plain BCE (E0)": ("E0", "E0_full_final_s0"),
             "AUC-margin (E6)": ("E6", "E6_full_final_s0"),
             "PanDerm FT + ISIC 2019": ("winner", "E0_full_final_s0_panderm_ft_isic2019")}
test = {name: pd.read_csv(REPO / "results/test" / run / "preds.csv") for name, (_, run) in test_runs.items()}

fig, ax = plt.subplots(figsize=(6, 5))
pr_curves(test, ax)
save(fig, "fig4a_test_pr")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (name, (key, _)) in zip(axes, test_runs.items()):
    # the cut that our protocol gives: it finds 80% of the melanomas in the held-out folds
    threshold = summary["protocol"]["models"][key]["threshold_80"]
    confusion_matrix(test[name], ax, name, threshold)
plt.tight_layout()
save(fig, "fig4b_test_confmat")

