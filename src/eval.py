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


def best_threshold(y, p):
    """The threshold with the highest F1. Tuned on inner_val ONLY — tuning it on
    the held fold would break the comparison.

    Library: sklearn.metrics.precision_recall_curve gives precision and recall at
    every candidate threshold.
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
      https://scikit-learn.org/stable/modules/classification_threshold.html
    """
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def score_run(run_dir, threshold=0.5):
    """One run -> one row of metrics on the held-out fold.

    threshold: a number (default 0.5, used for every trained arm) or "tuned"
    (best F1 on inner_val; this is what E0b uses). AUROC, AUPRC and ECE do not
    depend on the threshold; recall, precision and F1 do.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    preds = pd.read_csv(run_dir / "preds.csv")

    inner = preds[preds.split == "inner_val"]
    held = preds[preds.split == "held_fold"]
    if threshold == "tuned":
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
    """Every run folder -> one table. This is what the report's results section uses.

    Each trained run gives one row at threshold 0.5. Each E0 run gives a second
    row, arm "E0b": the same predictions with the threshold tuned on inner_val.
    """
    rows = []
    for d in sorted(Path(runs_dir).iterdir()):
        if not (d / "preds.csv").exists():
            continue
        row = score_run(d)
        rows.append(row)
        if row["arm"] == "E0":
            rows.append(score_run(d, threshold="tuned") | {"arm": "E0b"})
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def summarise(results):
    """Mean +/- std across folds, per arm x depth — the headline table."""
    return (results.groupby(["arm", "depth"])[["auroc", "auprc", "recall", "f1", "ece"]]
            .agg(["mean", "std"]).round(4))


def e0b(runs_dir, depth, fold, seed=0):
    """E0b: E0's model with the threshold tuned on inner_val instead of 0.5. No training."""
    return score_run(Path(runs_dir) / f"E0_{depth}_f{fold}_s{seed}",
                     threshold="tuned") | {"arm": "E0b"}



def tuning_summary(folder, out_name="summary.md"):
    """One small table per tuning folder, written to <folder>/summary.md.

    Inner validation only (choices are never made on the held-out fold), and only
    the numbers needed to pick:
      augmentation tests  -> AUROC per fold and the mean (pick the highest mean);
      loss settings (pos_weight, gamma) -> F1 at threshold 0.5 (pick the highest),
      with melanomas found and benign false alarms at 0.5 to show the trade-off.
    """
    folder = Path(folder)
    runs = []
    for d in sorted(folder.iterdir()):
        if not (d / "preds.csv").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        cfg.setdefault("aug", "base")  # runs from before the ladder existed used base
        preds = pd.read_csv(d / "preds.csv")
        inner = preds[preds.split == "inner_val"]
        y, p = inner.target.values, inner.prob.values
        yhat = (p >= 0.5).astype(int)
        runs.append({"cfg": cfg, "fold": cfg["fold"],
                     "f1": f1_score(y, yhat, zero_division=0),
                     "found": f"{int(yhat[y == 1].sum())} of {int(y.sum())}",
                     "false_alarms": int(yhat[y == 0].sum()),
                     "auroc": roc_auc_score(y, p)})
    if not runs:
        return None

    def num(v):  # sort 7 < 15 < 37, and text alphabetically
        try:
            return (0, float(v), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(v))

    keys = [k for k in ("arm", "aug", "pos_weight", "gamma", "alpha", "epochs")
            if len({str(r["cfg"].get(k)) for r in runs}) > 1] or ["tag"]
    table = [f"# {folder.name}", ""]

    if keys == ["aug"]:
        order = ["none", "base", "geo", "light", "colour", "cutout", "randaug"]
        folds = sorted({r["fold"] for r in runs})
        auc = {(r["cfg"]["aug"], r["fold"]): r["auroc"] for r in runs}
        steps = sorted({a for a, _ in auc}, key=lambda a: order.index(a) if a in order else 99)
        table += ["Inner-validation AUROC. Pick the highest mean.", "",
                  "| aug | " + " | ".join(f"fold {f}" for f in folds) + " | mean |",
                  "|" + "---|" * (len(folds) + 2)]
        for a in steps:
            vals = [auc.get((a, f)) for f in folds]
            got = [v for v in vals if v is not None]
            mean = f"{np.mean(got):.3f}" if len(got) == len(folds) else "–"
            table.append(f"| {a} | " + " | ".join(f"{v:.3f}" if v is not None else "–" for v in vals)
                         + f" | {mean} |")
    else:
        rows = sorted(runs, key=lambda r: (r["fold"], [num(r["cfg"].get(k)) for k in keys]))
        head = keys + ["fold", "F1 at 0.5", "melanomas found", "false alarms", "AUROC"]
        table += ["Inner validation, threshold 0.5. Pick the highest F1.", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in rows:
            table.append("| " + " | ".join([str(r["cfg"].get(k)) for k in keys] + [
                str(r["fold"]), f"{r['f1']:.3f}", r["found"], str(r["false_alarms"]),
                f"{r['auroc']:.3f}"]) + " |")
    (folder / out_name).write_text("\n".join(table) + "\n")
    return folder / out_name
