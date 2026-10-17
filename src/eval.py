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
                             precision_recall_curve, confusion_matrix)


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


def false_alarms_to_find(y, p, share):
    """Lower the threshold until `share` of the malignant images are called
    malignant, and count the benign images called malignant at that point.
    The same for any monotone shift of the scores, so it measures separation,
    not where the 0.5 cut happens to fall."""
    y, p = np.asarray(y), np.asarray(p)
    pos = np.sort(p[y == 1])[::-1]
    t = pos[int(np.ceil(share * len(pos))) - 1]
    return int(((p >= t) & (y == 0)).sum())


def counts(y, yhat):
    """The four outcome counts and the percentages read off them.

    Library: sklearn.metrics.confusion_matrix; for labels [0, 1] it is laid out
    [[tn, fp], [fn, tp]].
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    found / missed: malignant images called malignant / benign.
    false_alarms / benign_correct: benign images called malignant / benign.
    malignant_accuracy = found / all malignant (the same number as recall).
    benign_accuracy = benign_correct / all benign.
    malignant_calls_right = found / all images called malignant (same as precision).
    benign_calls_right = benign_correct / all images called benign.
    """
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, yhat, labels=[0, 1]).ravel())
    return {
        "found": tp, "missed": fn, "false_alarms": fp, "benign_correct": tn,
        "malignant_accuracy": tp / max(tp + fn, 1),
        "benign_accuracy": tn / max(tn + fp, 1),
        "malignant_calls_right": tp / max(tp + fp, 1),
        "benign_calls_right": tn / max(tn + fn, 1),
    }


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
        "fa_find50": false_alarms_to_find(y, p, 0.5),
        "fa_find80": false_alarms_to_find(y, p, 0.8),
        **counts(y, yhat),
        "minutes": cfg.get("minutes"),
    }


def build_results(runs_dir, out_csv=None, tuned_all=False):
    """Every run folder -> one table. This is what the report's results section uses.

    Each trained run gives one row at threshold 0.5 (cut "0.5"). Each E0 run gives
    a second row, arm "E0b": the same predictions with the threshold tuned on
    inner_val (cut "tuned"). With tuned_all=True every arm gets such a row
    (E1b, E2b, E3b), so all arms can also be compared at their own best cut.
    """
    rows = []
    for d in sorted(Path(runs_dir).iterdir()):
        if not (d / "preds.csv").exists():
            continue
        row = score_run(d) | {"cut": "0.5"}
        rows.append(row)
        if row["arm"] == "E0" or tuned_all:
            rows.append(score_run(d, threshold="tuned") | {"arm": row["arm"] + "b", "cut": "tuned"})
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
      loss / sampling settings (pos_weight, gamma, mal_frac) -> one row per run at threshold 0.5:
      melanomas found, benign false alarms, accuracy on each class, and how many
      of the malignant and benign calls were right (see `counts`).
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
        runs.append({"cfg": cfg, "fold": cfg["fold"], "c": counts(y, yhat),
                     "auroc": roc_auc_score(y, p)})
    if not runs:
        return None

    def num(v):  # sort 7 < 15 < 37, and text alphabetically
        try:
            return (0, float(v), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(v))

    keys = [k for k in ("arm", "aug", "pos_weight", "gamma", "alpha", "mal_frac", "sampling", "epochs", "seed")
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
        head = keys + ["fold", "melanomas found", "malignant accuracy", "false alarms",
                       "benign accuracy", "malignant guesses that are right",
                       "benign guesses that are right"]
        table += ["Inner validation, threshold 0.5 (an image is called malignant when "
                  "the network's output is 0.5 or more).", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in rows:
            c = r["c"]
            n_mal, n_ben = c["found"] + c["missed"], c["benign_correct"] + c["false_alarms"]
            said_mal, said_ben = c["found"] + c["false_alarms"], c["benign_correct"] + c["missed"]
            table.append("| " + " | ".join([str(r["cfg"].get(k)) for k in keys] + [
                str(r["fold"]),
                f"{c['found']} of {n_mal}", f"{c['malignant_accuracy']:.1%}",
                str(c["false_alarms"]), f"{c['benign_accuracy']:.1%}",
                f"{c['found']} of {said_mal} ({c['malignant_calls_right']:.1%})",
                f"{c['benign_correct']:,} of {said_ben:,} ({c['benign_calls_right']:.1%})",
            ]) + " |")
    (folder / out_name).write_text("\n".join(table) + "\n")
    return folder / out_name


# ---------------------------------------------------------------------------
# Phase 3 decision rules (group workflow document, "DeepLearning 5.pdf").
# They were fixed before the grid was run, so they cannot be bent to fit the
# results. Only rows at cut "0.5" are used: AUPRC and ECE do not depend on the cut.

def _paired(results, depth, a, b, metric="auprc"):
    """Per-fold metric of arm a and arm b at one depth, folds both have."""
    r = results[(results.cut == "0.5") & (results.depth == depth)]
    x = r[r.arm == a].set_index("fold")[metric]
    y = r[r.arm == b].set_index("fold")[metric]
    folds = sorted(set(x.index) & set(y.index))
    return x.loc[folds], y.loc[folds]


def trick_helps(results, arm, depth, min_wins=4):
    """A trick "helps" if its mean AUPRC is above E0's AND it is higher than E0
    in at least `min_wins` of the folds (4 of 5 in the workflow)."""
    x, e0 = _paired(results, depth, arm, "E0")
    wins = int((x > e0).sum())
    if len(x) == 0:
        return {"arm": arm, "depth": depth, "folds": 0, "helps": "no shared folds yet"}
    return {"arm": arm, "depth": depth, "folds": len(x),
            "auprc": x.mean(), "auprc_E0": e0.mean(), "gap": (x - e0).mean(),
            "wins": wins, "helps": bool(x.mean() > e0.mean() and wins >= min_wins)}


def e1_vs_e2(results, depth, tie=0.01):
    """E1 vs E2 by mean AUPRC; within `tie` of each other -> lower mean ECE wins.
    The workflow says "near tie" without a number; 0.01 AUPRC is our choice."""
    a, b = _paired(results, depth, "E1", "E2")
    ea, eb = _paired(results, depth, "E1", "E2", "ece")
    diff = a.mean() - b.mean()
    if len(a) == 0:
        return {"depth": depth, "folds": 0, "winner": "no shared folds yet"}
    if abs(diff) < tie:
        winner, why = ("E1" if ea.mean() < eb.mean() else "E2"), "near tie, lower ECE"
    else:
        winner, why = ("E1" if diff > 0 else "E2"), "higher AUPRC"
    return {"depth": depth, "folds": len(a), "auprc_E1": a.mean(), "auprc_E2": b.mean(),
            "ece_E1": ea.mean(), "ece_E2": eb.mean(), "winner": winner, "why": why}


def depth_choice(results, arm):
    """Higher mean AUPRC wins; if the gap is within the fold-to-fold spread
    (the larger of the two standard deviations), frozen wins as the simpler one."""
    r = results[(results.cut == "0.5") & (results.arm == arm)]
    full, frozen = r[r.depth == "full"].auprc, r[r.depth == "frozen"].auprc
    spread = max(full.std(), frozen.std())
    gap = full.mean() - frozen.mean()
    if len(full) < 2 or len(frozen) < 2:
        winner = "need 2+ folds per depth"
    else:
        winner = "full" if gap > spread else "frozen"
    return {"arm": arm, "auprc_full": full.mean(), "auprc_frozen": frozen.mean(),
            "gap": gap, "spread": spread, "winner": winner}


def trick_gap_by_depth(results, arm):
    """Kang et al. (2020) test: gap = AUPRC(trick) - AUPRC(E0), per fold, at each
    depth. If imbalance tricks mainly act on the last layer, the gap should be
    about as large with the backbone frozen as with full fine-tuning."""
    out = {"arm": arm}
    for depth in ("frozen", "full"):
        x, e0 = _paired(results, depth, arm, "E0")
        out[f"gap_{depth}"] = (x - e0).mean()
        out[f"gap_{depth}_std"] = (x - e0).std()
    return out
