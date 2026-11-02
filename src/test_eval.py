"""M13: scoring the locked test set. Run once, after the three final runs exist.

Everything here was fixed before the test set was opened (docs/test_protocol.md):
models, thresholds (configs/test.json), metrics, and the bootstrap.

Uncertainty: the test set has ~80 melanomas, so every number gets a 95% confidence
interval from a bootstrap that resamples *patients* (a patient's images stay
together, as in our splits), 2,000 resamples. Differences between models are
paired: the same resample scores both models.
  Efron & Tibshirani (1993), An Introduction to the Bootstrap.
  Patient-level (cluster) resampling: Field & Welsh (2007), https://doi.org/10.1111/j.1467-9868.2007.00593.x
Metrics: scikit-learn average_precision_score (AUPRC) and roc_auc_score;
ECE from eval.ece.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from eval import ece, counts

N_BOOT = 2000


def _metrics(y, p, t):
    c = counts(y, (p >= t).astype(int))
    return {"auprc": average_precision_score(y, p), "auroc": roc_auc_score(y, p), "ece": ece(y, p),
            "found": c["found"], "malignant": c["found"] + c["missed"], "false_alarms": c["false_alarms"],
            "malignant_accuracy": c["malignant_accuracy"], "benign_accuracy": c["benign_accuracy"],
            "malignant_calls_right": c["malignant_calls_right"]}


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


def _markdown(r):
    lines = [f"# Test set (opened once) — {r['n_images']} images, {r['n_melanomas']} melanomas, "
             f"{r['n_patients']} patients\n",
             "| Model | AUPRC [95% CI] | AUROC [95% CI] | Found / malignant | False alarms | ECE | "
             "AUPRC w/o Queensland+Sydney |", "|---|---|---|---|---|---|---|"]
    for n, m in r["models"].items():
        lines.append(f"| {n} | {m['auprc']:.3f} [{m['auprc_ci95'][0]:.3f}, {m['auprc_ci95'][1]:.3f}] | "
                     f"{m['auroc']:.3f} [{m['auroc_ci95'][0]:.3f}, {m['auroc_ci95'][1]:.3f}] | "
                     f"{m['found']} / {m['malignant']} | {m['false_alarms']} | {m['ece']:.3f} | "
                     f"{m['auprc_without_qld_sydney']:.3f} |")
    lines += ["", "| Difference | Value [95% CI] | Resamples above 0 |", "|---|---|---|"]
    for n, d in r["differences"].items():
        lines.append(f"| {n} | {d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | "
                     f"{100 * d['share_of_resamples_above_0']:.0f}% |")
    return "\n".join(lines) + "\n"
