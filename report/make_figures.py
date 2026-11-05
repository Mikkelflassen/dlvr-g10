"""Figures and tables for the report. Run from the repo root:  python report/make_figures.py

Reads only files in results/ (preds.csv, history.csv, results/test/), so every
number in the report can be traced back to a run folder. Writes
report/figures/*.pdf and report/numbers.tex (LaTeX macros with the numbers the
text uses).
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from eval import score_run  # noqa: E402

OUT = REPO / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RUNS, M12, TEST = REPO / "results" / "runs", REPO / "results" / "m12", REPO / "results" / "test"

# Colours: categorical slots 1-4 of the validated reference palette (dataviz skill),
# in fixed order; line style is the second encoding so print/greyscale still works.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = ["-", "--", "-.", ":"]
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
ARMS = [("E0", "Plain BCE (E0)"), ("E1", "Weighted BCE (E1)"),
        ("E2", "Balanced sampling (E2)"), ("E3", "Focal loss (E3)")]


def held(run_dir):
    p = pd.read_csv(Path(run_dir) / "preds.csv")
    return p[p.split == "held_fold"]


# ---------------------------------------------------------------- Figure 1
def fig_tradeoff_reliability():
    """(a) false alarms against share of melanomas found, with where 0.5 lands;
    (b) reliability diagrams. Full fine-tuning, 5 held-out folds."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.45), gridspec_kw={"width_ratios": [1.25, 1]})
    grid = np.linspace(0.05, 0.95, 37)
    for i, (arm, label) in enumerate(ARMS):
        fa = []
        marks = []
        for f in range(5):
            h = held(RUNS / f"{arm}_full_f{f}_s0")
            y, p = h.target.values, h.prob.values
            pos = np.sort(p[y == 1])[::-1]
            fa.append([((p >= pos[int(np.ceil(s * len(pos))) - 1]) & (y == 0)).sum() for s in grid])
            marks.append(((p >= 0.5) & (y == 1)).sum() / (y == 1).sum())
            marks.append(((p >= 0.5) & (y == 0)).sum())
        a.plot(grid * 100, np.mean(fa, axis=0), color=C[i], ls=STYLE[i], label=label)
        rec, fal = np.mean(marks[0::2]) * 100, np.mean(marks[1::2])
        a.plot(rec, fal, "o", ms=6, color=C[i], mec="white", mew=1.2, zorder=5)
    a.set_xlabel("Melanomas found (%)")
    a.set_ylabel("False alarms per fold (of ~5,500 benign)")
    a.set_title("(a) Cost of finding melanomas; dots = threshold 0.5", loc="left")
    a.legend(loc="upper left")
    a.set_xlim(0, 100)
    a.set_ylim(0, None)

    bins = np.linspace(0, 1, 11)
    b.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--")
    for i, (arm, label) in enumerate(ARMS):
        h = pd.concat([held(RUNS / f"{arm}_full_f{f}_s0") for f in range(5)])
        idx = np.clip(np.digitize(h.prob, bins) - 1, 0, 9)
        xs, ys = [], []
        for k in range(10):
            m = idx == k
            if m.sum() >= 20:
                xs.append(h.prob[m].mean()); ys.append(h.target[m].mean())
        b.plot(xs, ys, color=C[i], ls=STYLE[i], marker="o", ms=3.5)
    b.set_xlabel("Predicted probability of melanoma")
    b.set_ylabel("Observed share of melanomas")
    b.set_title("(b) Calibration (reliability diagram)", loc="left")
    b.set_xlim(0, 1); b.set_ylim(0, 1)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig1_tradeoff_reliability.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
STEPS = [  # (label, run pattern) in the order of the text
    ("Plain BCE (base)", RUNS / "E0_full_f{f}_s0"),
    ("E2 + E3 combined (E5)", RUNS / "E5_full_f{f}_s0"),
    ("AUC-margin loss (E6)", RUNS / "E6_full_f{f}_s0"),
    ("384 px images", M12 / "E0_full_f{f}_s0_384"),
    ("EfficientNet-B0", M12 / "E0_full_f{f}_s0_effb0"),
    ("PanDerm, frozen", M12 / "E0_frozen_f{f}_s0_panderm"),
    ("+ ISIC 2019 (ResNet-18)", M12 / "E0_full_f{f}_s0_isic2019"),
    ("PanDerm frozen + ISIC 2019", M12 / "E0_frozen_f{f}_s0_panderm_isic2019"),
    ("PanDerm fine-tuned", M12 / "E0_full_f{f}_s0_panderm_ft"),
    ("PanDerm fine-tuned + ISIC 2019", M12 / "E0_full_f{f}_s0_panderm_ft_isic2019"),
    ("  + age, sex, body site", M12 / "E0_full_f{f}_s0_panderm_ft_isic2019_meta"),
]


def step_scores():
    rows = []
    for label, pat in STEPS:
        for f in range(5):
            r = score_run(str(pat).format(f=f), threshold="tuned")
            rows.append({"step": label, "fold": f, **{k: r[k] for k in
                         ("auprc", "auroc", "ece", "fa_find50", "fa_find80")}})
    return pd.DataFrame(rows)


def fig_steps(df):
    base = df[df.step == STEPS[0][0]].set_index("fold").auprc
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    labels = [s for s, _ in STEPS]
    for y, s in enumerate(labels):
        d = df[df.step == s].set_index("fold").auprc
        wins = int((d > base).sum()) if y else None
        passed = y and d.mean() > base.mean() and wins >= 4
        col = C[0] if passed else MUTED
        ax.scatter(d.values, [y] * 5, s=14, color=col, alpha=0.55, lw=0, zorder=3)
        ax.plot([d.mean()] * 2, [y - 0.32, y + 0.32], color=col if y else INK, lw=2.2, zorder=4)
        txt = f"{d.mean():.3f}" + (f"   {wins}/5" if y else "")
        ax.text(0.405, y, txt, va="center", ha="left", fontsize=7.5, color=INK)
    ax.axvline(base.mean(), color=INK, lw=0.8, ls="--", zorder=2)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0.05, 0.40)
    ax.set_xlabel("AUPRC on the held-out fold (dots: 5 folds; bar: mean)")
    ax.text(0.405, -1.0, "mean   folds > base", fontsize=7, color=MUTED, va="center")
    ax.grid(axis="y", visible=False)
    for y in (2.5, 4.5):
        ax.axhline(y, color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_steps.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig_long_training():
    h = pd.read_csv(M12 / "E0_full_f0_s0_panderm_ft_isic2019_e50" / "history.csv")
    short = score_run(M12 / "E0_full_f0_s0_panderm_ft_isic2019", threshold="tuned")
    best = int(h.val_auroc.idxmax())
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.2))
    a.plot(h.epoch, h.train_clean_auroc, color=C[1], ls="--", label="Training images (clean)")
    a.plot(h.epoch, h.val_auroc, color=C[0], ls="-", label="Inner validation (selects the epoch)")
    a.plot(h.epoch, h.held_auroc, color=C[2], ls="-.", label="Held-out fold (recorded only)")
    a.set_ylabel("AUROC"); a.set_xlabel("Epoch")
    a.set_title("(a) AUROC per epoch", loc="left")
    a.legend(loc="lower right"); a.set_ylim(0.62, 1.005)
    b.plot(h.epoch, h.held_auprc, color=C[2], ls="-.")
    b.axhline(short["auprc"], color=INK, lw=0.8, ls="--")
    b.text(28.5, short["auprc"] + 0.008, f"8-epoch run: {short['auprc']:.3f}", ha="right", fontsize=7.5)
    b.set_ylabel("Held-out AUPRC"); b.set_xlabel("Epoch")
    b.set_title("(b) Held-out AUPRC per epoch", loc="left")
    for ax in (a, b):
        ax.axvline(best, color=MUTED, lw=0.8, ls=":")
    b.text(best + 0.5, 0.15, f"best inner validation\n(epoch {best}): {h.held_auprc[best]:.3f}", fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig3_long_training.pdf")
    plt.close(fig)
    return {"best": best, "best_val": h.val_auroc[best], "held_at_best": h.held_auprc[best],
            "epochs_done": int(h.epoch.max()) + 1, "short": short["auprc"],
            "first_train_1": int(h.epoch[h.train_clean_auroc >= 0.9995].min())}


# ---------------------------------------------------------------- Figure 4
def fig_test(summary):
    proto = json.loads((REPO / "configs" / "test.json").read_text())
    test = pd.read_csv(REPO / "splits" / "test_split.csv")[["image_name", "target"]]
    names = {"E0": "Plain BCE (E0)", "E6": "AUC-margin loss (E6)", "winner": "PanDerm FT + ISIC 2019"}
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for i, (key, label) in enumerate(names.items()):
        c = proto["models"][key]["config"]
        run = f"{c['arm']}_{c['depth']}_final_s{c['seed']}" + (f"_{c['tag']}" if c.get("tag") else "")
        p = test.merge(pd.read_csv(TEST / run / "preds.csv")[["image_name", "prob"]], on="image_name")
        pr, rc, _ = precision_recall_curve(p.target, p.prob)
        ap = average_precision_score(p.target, p.prob)
        ax.plot(rc * 100, pr * 100, color=C[i], ls=STYLE[i], label=f"{label}: {ap:.3f}", drawstyle="steps-post")
        t = proto["models"][key]["threshold_80"]
        call = p.prob >= t
        r = (call & (p.target == 1)).sum() / p.target.sum() * 100
        q = (call & (p.target == 1)).sum() / call.sum() * 100
        ax.plot(r, q, "o", ms=6, color=C[i], mec="white", mew=1.2, zorder=5)
    ax.axhline(test.target.mean() * 100, color=MUTED, lw=0.8, ls="--")
    ax.text(2, test.target.mean() * 100 + 1.5, "chance", fontsize=7, color=MUTED)
    ax.set_xlabel("Recall: melanomas found (%)"); ax.set_ylabel("Precision (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper right", title="AUPRC on the test set", title_fontsize=7.5, borderaxespad=0.2)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_test_pr.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_tradeoff_reliability()
    df = step_scores()
    df.to_csv(OUT / "steps_per_fold.csv", index=False)
    fig_steps(df)
    long = fig_long_training()
    summary = json.loads((TEST / "summary.json").read_text())
    fig_test(summary)
    print(df.groupby("step", sort=False)[["auprc", "auroc", "ece", "fa_find50", "fa_find80"]].mean().round(3))
    print(long)
