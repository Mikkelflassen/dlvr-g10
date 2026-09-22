"""All metrics, computed from saved preds.csv files. Nothing here retrains."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             recall_score, precision_score, f1_score)


def ece(y, p, n_bins=15):
    """Expected calibration error — the E3 calibration claim lives or dies here."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            out += m.mean() * abs(y[m].mean() - p[m].mean())
    return out


def best_threshold(y, p, metric="f1"):
    """Tuned on inner_val ONLY. Tuning it on the held fold would break the comparison."""
    grid = np.unique(np.round(p, 4))
    scorer = f1_score if metric == "f1" else recall_score
    scores = [scorer(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


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
