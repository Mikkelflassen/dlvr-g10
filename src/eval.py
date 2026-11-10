"""All our metrics, calculated from the saved preds.csv files.

scikit-learn: https://scikit-learn.org/stable/modules/model_evaluation.html
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             recall_score, precision_score, f1_score,
                             precision_recall_curve, confusion_matrix)


def ece(y, p, n_bins=15):
    """Expected calibration error: how far the network's percentages are from the real rates.

    Naeini et al. (2015); Guo et al. (2017), https://arxiv.org/abs/1706.04599
    Adapted from `_ECELoss` (G. Pleiss, MIT):
      https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Changes: NumPy instead of torch, and the binary form.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    total = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # |mean score - real share of melanomas| in each bin, weighted by the bin size
        in_bin = (p > bin_lower) & (p <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y[in_bin].mean()
            avg_confidence_in_bin = p[in_bin].mean()
            total += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(total)


# F1 and the precision-recall trade-off: see the bonus lecture (week 4), sections 5 and 7.
def best_threshold(y, p):
    """The threshold with the highest F1. We only use it on inner validation.

      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
      https://scikit-learn.org/stable/modules/classification_threshold.html
    """
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[int(np.argmax(f1))])


# Written with AI (Claude).
def false_alarms_to_find(y, p, share):
    """Lowers the threshold until `share` of the melanomas are found, and counts the false alarms."""
    y, p = np.asarray(y), np.asarray(p)
    melanoma_scores = np.sort(p[y == 1])[::-1]                     # highest first
    cut = melanoma_scores[int(np.ceil(share * len(melanoma_scores))) - 1]
    return int(((p >= cut) & (y == 0)).sum())                       # benign images above the cut


# Based on lab 2 (confusion_matrix) and the bonus lecture (week 4), section 4.
def counts(y, yhat):
    """Melanomas found and missed, false alarms, and the percentages.

      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, yhat, labels=[0, 1]).ravel())
    return {
        "found": tp, "missed": fn, "false_alarms": fp, "benign_correct": tn,
        "malignant_accuracy": tp / max(tp + fn, 1),
        "benign_accuracy": tn / max(tn + fp, 1),
        "malignant_calls_right": tp / max(tp + fp, 1),
        "benign_calls_right": tn / max(tn + fn, 1),
    }


# AP (we call it AUPRC): see the bonus lecture (week 4), sections 6 and 7.
def score_run(run_dir, threshold=0.5):
    """Scores one run on the held-out fold.

    threshold is a number or "tuned" (best F1 on inner validation).
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
        "fa_find50": false_alarms_to_find(y, p, 0.5),
        "fa_find80": false_alarms_to_find(y, p, 0.8),
        **counts(y, yhat),
        "minutes": cfg.get("minutes"),
    }


# one row per run, plus a row with the tuned cut
def build_results(runs_dir, out_csv=None, tuned_all=False):
    """Makes one table from all run folders. E0 also gets a tuned row. tuned_all=True does this for every arm."""
    rows = []
    for run_dir in sorted(Path(runs_dir).iterdir()):
        if not (run_dir / "preds.csv").exists():
            continue

        row = score_run(run_dir) | {"cut": "0.5"}
        rows.append(row)

        if row["arm"] == "E0" or tuned_all:
            rows.append(
                score_run(run_dir, threshold="tuned") | {"arm": row["arm"] + "b", "cut": "tuned"}
            )

    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def summarise(results):
    """Mean and std over the folds."""
    return (
        results
        .groupby(["arm", "depth"])[["auroc", "auprc", "recall", "f1", "ece"]]
        .agg(["mean", "std"])
        .round(4)
    )


# E0 again but with the tuned threshold
def e0b(runs_dir, depth, fold, seed=0):
    """E0 with the threshold tuned on inner validation."""
    return (
        score_run(Path(runs_dir) / f"E0_{depth}_f{fold}_s{seed}", threshold="tuned")
        | {"arm": "E0b"}
    )


# Written with AI (Claude).
def tuning_summary(folder, out_name="summary.md"):
    """Writes a small table for a tuning folder to summary.md (inner validation only)."""
    folder = Path(folder)
    runs = _tuning_runs(folder)
    if not runs:
        return None

    # the settings that differ between the runs become the first columns
    keys = [key for key in ("arm", "aug", "pos_weight", "gamma", "alpha", "mal_frac", "sampling",
                            "epochs", "seed")
            if len({str(run["cfg"].get(key)) for run in runs}) > 1] or ["tag"]

    table = [f"# {folder.name}", ""]
    if keys == ["aug"]:
        table += _augmentation_table(runs)
    else:
        table += _counts_table(runs, keys)
    (folder / out_name).write_text("\n".join(table) + "\n")
    return folder / out_name


# reads every run in a tuning folder and scores it on inner validation at 0.5
def _tuning_runs(folder):
    runs = []
    for run_dir in sorted(folder.iterdir()):
        if not (run_dir / "preds.csv").exists():
            continue
        cfg = json.loads((run_dir / "config.json").read_text())
        cfg.setdefault("aug", "base")  # runs from before the ladder existed used base
        preds = pd.read_csv(run_dir / "preds.csv")
        inner = preds[preds.split == "inner_val"]
        y, p = inner.target.values, inner.prob.values
        yhat = (p >= 0.5).astype(int)
        runs.append({"cfg": cfg, "fold": cfg["fold"], "c": counts(y, yhat),
                     "auroc": roc_auc_score(y, p)})
    return runs


# augmentation runs: one row per step, one column per fold
def _augmentation_table(runs):
    order = ["none", "base", "geo", "light", "colour", "cutout", "randaug"]
    folds = sorted({run["fold"] for run in runs})
    auc = {(run["cfg"]["aug"], run["fold"]): run["auroc"] for run in runs}
    steps = sorted({step for step, _ in auc}, key=lambda step: order.index(step) if step in order else 99)

    lines = ["Inner-validation AUROC. Pick the highest mean.", "",
             "| aug | " + " | ".join(f"fold {fold}" for fold in folds) + " | mean |",
             "|" + "---|" * (len(folds) + 2)]
    for step in steps:
        values = [auc.get((step, fold)) for fold in folds]
        found = [value for value in values if value is not None]
        mean = f"{np.mean(found):.3f}" if len(found) == len(folds) else "–"
        cells = [f"{value:.3f}" if value is not None else "–" for value in values]
        lines.append(f"| {step} | " + " | ".join(cells) + f" | {mean} |")
    return lines


# other runs: melanomas found, false alarms and so on, one row per run
def _counts_table(runs, keys):
    def as_number(value):  # sort 7 < 15 < 37, and text alphabetically
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(value))

    rows = sorted(runs, key=lambda run: (run["fold"], [as_number(run["cfg"].get(key)) for key in keys]))
    head = keys + ["fold", "melanomas found", "malignant accuracy", "false alarms",
                   "benign accuracy", "malignant guesses that are right",
                   "benign guesses that are right"]
    lines = ["Inner validation, threshold 0.5 (an image is called malignant when "
             "the network's output is 0.5 or more).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run in rows:
        c = run["c"]
        n_mal = c["found"] + c["missed"]
        said_mal = c["found"] + c["false_alarms"]
        said_ben = c["benign_correct"] + c["missed"]
        cells = [str(run["cfg"].get(key)) for key in keys] + [
            str(run["fold"]),
            f"{c['found']} of {n_mal}",
            f"{c['malignant_accuracy']:.1%}",
            str(c["false_alarms"]),
            f"{c['benign_accuracy']:.1%}",
            f"{c['found']} of {said_mal} ({c['malignant_calls_right']:.1%})",
            f"{c['benign_correct']:,} of {said_ben:,} ({c['benign_calls_right']:.1%})",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


# ---------------------------------------------------------------------------
# Our decision rules. We wrote them before we ran the grid.

# same folds for both arms so the comparison is fair
def _paired(results, depth, a, b, metric="auprc"):
    """The scores of two arms on the folds they both have."""
    rows = results[(results.cut == "0.5") & (results.depth == depth)]
    scores_a = rows[rows.arm == a].set_index("fold")[metric]
    scores_b = rows[rows.arm == b].set_index("fold")[metric]
    folds = sorted(set(scores_a.index) & set(scores_b.index))
    return scores_a.loc[folds], scores_b.loc[folds]


# our rule: higher mean and better on 4 of 5 folds
def trick_helps(results, arm, depth, min_wins=4):
    """A trick helps if its mean AUPRC is above E0 and it wins on at least `min_wins` folds."""
    trick, baseline = _paired(results, depth, arm, "E0")
    wins = int((trick > baseline).sum())

    if len(trick) == 0:
        return {"arm": arm, "depth": depth, "folds": 0, "helps": "no shared folds yet"}

    return {
        "arm": arm,
        "depth": depth,
        "folds": len(trick),
        "auprc": trick.mean(),
        "auprc_E0": baseline.mean(),
        "gap": (trick - baseline).mean(),
        "wins": wins,
        "helps": bool(trick.mean() > baseline.mean() and wins >= min_wins),
    }


# if they are almost equal the lower ece wins
def e1_vs_e2(results, depth, tie=0.01):
    """Higher mean AUPRC wins. A difference under 0.01 is a tie, then the lower ECE wins."""
    auprc_e1, auprc_e2 = _paired(results, depth, "E1", "E2")
    ece_e1, ece_e2 = _paired(results, depth, "E1", "E2", "ece")
    diff = auprc_e1.mean() - auprc_e2.mean()

    if len(auprc_e1) == 0:
        return {"depth": depth, "folds": 0, "winner": "no shared folds yet"}

    if abs(diff) < tie:
        winner, why = ("E1" if ece_e1.mean() < ece_e2.mean() else "E2"), "near tie, lower ECE"
    else:
        winner, why = ("E1" if diff > 0 else "E2"), "higher AUPRC"

    return {
        "depth": depth,
        "folds": len(auprc_e1),
        "auprc_E1": auprc_e1.mean(),
        "auprc_E2": auprc_e2.mean(),
        "ece_E1": ece_e1.mean(),
        "ece_E2": ece_e2.mean(),
        "winner": winner,
        "why": why,
    }


# if the gap is smaller than the fold spread, frozen wins
def depth_choice(results, arm):
    """Higher mean AUPRC wins. If the gap is within the fold spread, frozen wins."""
    rows = results[(results.cut == "0.5") & (results.arm == arm)]
    full, frozen = rows[rows.depth == "full"].auprc, rows[rows.depth == "frozen"].auprc
    spread = max(full.std(), frozen.std())
    gap = full.mean() - frozen.mean()

    if len(full) < 2 or len(frozen) < 2:
        winner = "need 2+ folds per depth"
    else:
        winner = "full" if gap > spread else "frozen"

    return {
        "arm": arm,
        "auprc_full": full.mean(),
        "auprc_frozen": frozen.mean(),
        "gap": gap,
        "spread": spread,
        "winner": winner,
    }


# Kang et al. (2020): https://arxiv.org/abs/1910.09217
def trick_gap_by_depth(results, arm):
    """Kang et al. (2020): is a trick's gain over E0 about the same frozen and fully fine-tuned?"""
    out = {"arm": arm}
    for depth in ("frozen", "full"):
        trick, baseline = _paired(results, depth, arm, "E0")
        out[f"gap_{depth}"] = (trick - baseline).mean()
        out[f"gap_{depth}_std"] = (trick - baseline).std()
    return out
