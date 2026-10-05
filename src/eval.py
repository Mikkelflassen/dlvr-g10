"""All metrics, computed from saved preds.csv files. Nothing here retrains.

Metrics are scikit-learn functions:
  https://scikit-learn.org/stable/modules/model_evaluation.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             recall_score, precision_score, f1_score,
                             precision_recall_curve)


def ece(y, p, n_bins=15):
    """Expected calibration error — the E3 calibration claim lives or dies here.

    Method: Naeini et al. (2015); Guo et al. (2017), https://arxiv.org/abs/1706.04599
    Adapted from `_ECELoss` in Guo et al.'s code (G. Pleiss, MIT):
      https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Changes: NumPy instead of torch, and the binary form — in each bin the mean
    predicted probability of malignant is compared with the observed fraction of
    malignant, instead of top-class confidence with accuracy.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = (p > bin_lower) & (p <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y[in_bin].mean()
            avg_confidence_in_bin = p[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def best_threshold(y, p, metric="f1"):
    """Tuned on inner_val ONLY. Tuning it on the held fold would break the comparison.

    Library: sklearn.metrics.precision_recall_curve gives precision and recall at
    every candidate threshold; we take the one with the highest F1 (or recall).
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
      https://scikit-learn.org/stable/modules/classification_threshold.html
    """
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    if metric == "f1":
        scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    else:
        scores = recall
    return float(thresholds[int(np.argmax(scores))])


def score_run(run_dir, threshold=None):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    preds = pd.read_csv(run_dir / "preds.csv")

    inner = preds[preds.split == "inner_val"]
    held = preds[preds.split == "held_fold"]
    if threshold is None:
        threshold = best_threshold(inner.target.values, inner.prob.values)

    y, p = held.target.values, held.prob.values
    yhat = (p >= threshold).astype(int)
    return {
        "run": cfg["run"], "arm": cfg["arm"], "depth": cfg["depth"],
        "fold": cfg["fold"], "seed": cfg["seed"], "threshold": round(threshold, 4),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "recall": recall_score(y, yhat, zero_division=0),
        "precision": precision_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "ece": ece(y, p),
        "minutes": cfg.get("minutes"),
    }


def build_results(runs_dir, out_csv=None):
    """Every run folder -> one table. This is what the report's results section uses."""
    rows = [score_run(d) for d in sorted(Path(runs_dir).iterdir())
            if (d / "preds.csv").exists()]
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def summarise(results):
    """Mean +/- std across folds, per arm x depth — the headline table."""
    return (results.groupby(["arm", "depth"])[["auroc", "auprc", "recall", "f1", "ece"]]
            .agg(["mean", "std"]).round(4))


def e0b(runs_dir, depth, fold, seed=0, metric="recall"):
    """E0b: E0's model, thresholded for recall instead of 0.5. No training."""
    return score_run(Path(runs_dir) / f"E0_{depth}_f{fold}_s{seed}",
                     threshold=None) | {"arm": "E0b"}
