"""Scores the locked test set. We ran this once.

Everything was fixed before we opened the test set (results/test/protocol.md).
The 95% intervals come from a bootstrap over patients (2,000 resamples).
  Efron & Tibshirani (1993), An Introduction to the Bootstrap.
  Field & Welsh (2007), https://doi.org/10.1111/j.1467-9868.2007.00593.x
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from eval import ece, counts

N_BOOT = 2000


# all the numbers for one model
def _metrics(y, p, t):
    outcome = counts(y, (p >= t).astype(int))
    return {
        "auprc": average_precision_score(y, p),
        "auroc": roc_auc_score(y, p),
        "ece": ece(y, p),
        "found": outcome["found"],
        "malignant": outcome["found"] + outcome["missed"],
        "false_alarms": outcome["false_alarms"],
        "malignant_accuracy": outcome["malignant_accuracy"],
        "benign_accuracy": outcome["benign_accuracy"],
        "malignant_calls_right": outcome["malignant_calls_right"],
    }


def evaluate(repo, out_dir=None, seed=0):
    repo = Path(repo)
    proto = json.loads((repo / "configs" / "test.json").read_text())
    test = pd.read_csv(repo / "splits" / "test_split.csv")[["image_name", "patient_id", "target"]]
    attr = pd.read_csv(repo / "splits" / "isic2020_attribution.csv")[["image_name", "attribution"]]
    test = test.merge(attr, on="image_name", how="left")
    probs, thr = {}, {}
    for name, m in proto["models"].items():
        c = m["config"]
        run = f"{c['arm']}_{c['depth']}_final_s{c['seed']}" + (f"_{c['tag']}" if c.get("tag") else "")
        p = pd.read_csv(repo / "results" / "test" / run / "preds.csv")
        assert len(p) == len(test) and set(p.image_name) == set(test.image_name), f"{run}: not the whole test set"
        probs[name] = test[["image_name"]].merge(p[["image_name", "prob"]], on="image_name").prob.values
        thr[name] = m["threshold_80"]
    y = test.target.values

    rows = {name: _metrics(y, probs[name], thr[name]) for name in probs}
    aus = test.attribution.fillna("").str.contains("Queensland|Sydney", case=False).values
    for name in probs:
        rows[name]["auprc_without_qld_sydney"] = average_precision_score(y[~aus], probs[name][~aus])

    # The bootstrap was written with AI (Claude), following the references at the top.
    # patient bootstrap
    rng = np.random.default_rng(seed)
    patients = test.patient_id.values
    groups = {pid: np.flatnonzero(patients == pid) for pid in np.unique(patients)}
    pids = np.array(list(groups))
    boot = {name: {"auprc": [], "auroc": []} for name in probs}
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[pid] for pid in rng.choice(pids, len(pids), replace=True)])
        if y[idx].sum() == 0:
            continue
        for name in probs:
            boot[name]["auprc"].append(average_precision_score(y[idx], probs[name][idx]))
            boot[name]["auroc"].append(roc_auc_score(y[idx], probs[name][idx]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    for name in probs:
        for k in ("auprc", "auroc"):
            rows[name][f"{k}_ci95"] = ci(boot[name][k])
    diffs = {}
    for name in probs:
        if name == "E0":
            continue
        for k in ("auprc", "auroc"):
            d = np.array(boot[name][k]) - np.array(boot["E0"][k])
            diffs[f"{name} - E0 {k}"] = {"diff": rows[name][k] - rows["E0"][k], "ci95": ci(d),
                                         "share_of_resamples_above_0": float((d > 0).mean())}
    result = {"protocol": proto, "n_images": int(len(y)), "n_melanomas": int(y.sum()),
              "n_patients": int(len(pids)), "n_bootstrap": len(boot["E0"]["auprc"]),
              "models": rows, "differences": diffs}
    if out_dir:
        out_dir = Path(out_dir)
        (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=float))
        (out_dir / "summary.md").write_text(_markdown(result))
    return result


# writes the tables for summary.md
def _markdown(result):
    lines = [
        f"# Test set (opened once) — {result['n_images']} images, {result['n_melanomas']} melanomas, "
        f"{result['n_patients']} patients\n",
        "| Model | AUPRC [95% CI] | AUROC [95% CI] | Found / malignant | False alarms | ECE | "
        "AUPRC w/o Queensland+Sydney |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, model in result["models"].items():
        lines.append(
            f"| {name} | {model['auprc']:.3f} [{model['auprc_ci95'][0]:.3f}, {model['auprc_ci95'][1]:.3f}] | "
            f"{model['auroc']:.3f} [{model['auroc_ci95'][0]:.3f}, {model['auroc_ci95'][1]:.3f}] | "
            f"{model['found']} / {model['malignant']} | {model['false_alarms']} | {model['ece']:.3f} | "
            f"{model['auprc_without_qld_sydney']:.3f} |"
        )

    lines += ["", "| Difference | Value [95% CI] | Resamples above 0 |", "|---|---|---|"]

    for name, diff in result["differences"].items():
        lines.append(
            f"| {name} | {diff['diff']:+.3f} [{diff['ci95'][0]:+.3f}, {diff['ci95'][1]:+.3f}] | "
            f"{100 * diff['share_of_resamples_above_0']:.0f}% |"
        )

    return "\n".join(lines) + "\n"
