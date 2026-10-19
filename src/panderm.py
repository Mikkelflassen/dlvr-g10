"""M12 step `panderm`: a dermatology foundation model as a frozen encoder.

PanDerm (Yan et al., 2025, Nature Medicine, https://arxiv.org/abs/2410.15038) is a
ViT pretrained with self-supervised learning on ~2 million skin images. We use the
smaller PanDerm_Base (ViT-B/16, 768 features). The encoder is frozen: each image is
passed through it once, and only a linear classifier is trained on the features
(a "linear probe", the evaluation the PanDerm authors report as close to full
fine-tuning). This is the same depth as our `frozen` arms, so it can be compared
with E0_frozen (ImageNet ResNet-18) as well as with the M12 base (E0_full, tuned threshold).

Code and weights: https://github.com/SiyuanYan1/PanDerm (cloned at run time).
  - model definition: classification/models/modeling_finetune.py,
    panderm_base_patch16_224 (used unchanged, loaded from the clone)
  - loading and image transform: classification/models/builder.py, get_encoder /
    get_eval_transforms: Resize(256), CenterCrop(224), ImageNet mean with
    std (0.228, 0.224, 0.225), CLS token as the feature (head = Identity)
  - their linear probe is an L2-regularised logistic regression on the features
    (panderm_model/downstream/eval_features/linear_probe.py); we use scikit-learn's
    LogisticRegression and pick its strength C on inner validation.
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Leakage check (see check_institutions.py): PanDerm's pretraining includes in-house
Australian cohorts (MYM, HOP) that are likely the same studies as ISIC 2020's
University of Queensland images. Results are therefore also reported without
Queensland images (`score_without`).

Output: one folder per fold, results/m12/E0_frozen_f<fold>_s0_panderm/, with the
same config.json / history.csv / preds.csv as every other run, so eval.score_run
works on it unchanged.
"""
from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from data import inner_split

# PanDerm's eval transform (builder.get_eval_transforms). Note std 0.228, as in their code.
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
])
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def load_panderm(panderm_repo, checkpoint, device):
    """PanDerm_Base with its pretrained weights, CLS-token features, eval mode.

    Their builder loads with strict=False, which would also accept a file with
    none of the right keys; we check that every encoder weight was found.
    """
    path = Path(panderm_repo) / "classification" / "models" / "modeling_finetune.py"
    spec = importlib.util.spec_from_file_location("panderm_modeling", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod     # timm's @register_model looks the module up here
    spec.loader.exec_module(mod)
    model = mod.panderm_base_patch16_224()

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    state = {k.replace("module.", "").replace("encoder.", ""): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    # head: replaced below. pos_embed: fixed sin-cos values built in the code, not learned.
    missing = [k for k in result.missing_keys if not k.startswith("head.") and k != "pos_embed"]
    n_found = len(model.state_dict()) - len(result.missing_keys)
    print(f"PanDerm weights: {n_found} tensors loaded, {len(missing)} encoder tensors missing, "
          f"{len(result.unexpected_keys)} extra in file")
    assert not missing, f"encoder weights missing from the checkpoint, e.g. {missing[:5]}"

    model.head = torch.nn.Identity()
    return model.eval().to(device)


class _Images(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TRANSFORM(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def extract(model, df, device, batch_size=128, workers=2):
    """One 768-number feature per image (CLS token), in the order of df."""
    loader = DataLoader(_Images(df.path), batch_size=batch_size, num_workers=workers)
    feats, start = [], time.time()
    for i, x in enumerate(loader):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            feats.append(model(x.to(device)).float().cpu())
        if i % 50 == 0:
            print(f"  {i * batch_size}/{len(df)} images, {(time.time() - start) / 60:.1f} min")
    return torch.cat(feats).numpy()


def probe_fold(feats, dev, fold, splits_dir, runs_dir, extra_cfg=None):
    """Train the linear probe for one held-out fold and write the run folder.

    The same splits as every other run: train / inner_val from inner_split.csv,
    the held-out fold is only predicted. C is chosen by inner-validation AUROC,
    the same criterion that picks the best epoch in train.run.
    """
    start = time.time()
    index = pd.Series(np.arange(len(dev)), index=dev.image_name)
    tr, val = inner_split(dev, fold, splits_dir)
    held = dev[dev.fold == fold]
    X = {k: feats[index[d.image_name].values] for k, d in (("tr", tr), ("val", val), ("held", held))}

    scaler = StandardScaler().fit(X["tr"])
    X = {k: scaler.transform(v) for k, v in X.items()}

    history, best = [], None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000).fit(X["tr"], tr.target)
        p_val = clf.predict_proba(X["val"])[:, 1]
        row = {"C": C, "val_auroc": roc_auc_score(val.target, p_val),
               "val_auprc": average_precision_score(val.target, p_val)}
        history.append(row)
        print(f"  fold {fold}  C={C:g}  inner AUROC {row['val_auroc']:.3f}  AUPRC {row['val_auprc']:.3f}")
        if best is None or row["val_auroc"] > best[0]["val_auroc"]:
            best = (row, clf)
    row, clf = best

    cfg = {"arm": "E0", "depth": "frozen", "backbone": "panderm_base", "img_size": 224,
           "fold": fold, "seed": 0, "tag": "panderm", "probe": "logistic_regression",
           "C": row["C"], "C_grid": C_GRID, "augment": False, **(extra_cfg or {})}
    name = f"E0_frozen_f{fold}_s0_panderm"
    out = Path(runs_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.concat([
        pd.DataFrame({"image_name": val.image_name, "target": val.target,
                      "prob": clf.predict_proba(X["val"])[:, 1], "split": "inner_val"}),
        pd.DataFrame({"image_name": held.image_name, "target": held.target,
                      "prob": clf.predict_proba(X["held"])[:, 1], "split": "held_fold"}),
    ])
    preds.to_csv(out / "preds.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    cfg.update({"run": name, "best_inner_auroc": row["val_auroc"],
                "minutes": round((time.time() - start) / 60, 2)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return out


def score_without(run_dir, attribution_csv, institution="Queensland"):
    """AUPRC / AUROC on the held-out fold, with all images and without one institution."""
    preds = pd.read_csv(Path(run_dir) / "preds.csv")
    held = preds[preds.split == "held_fold"].merge(
        pd.read_csv(attribution_csv)[["image_name", "attribution"]], on="image_name", how="left")
    drop = held.attribution.fillna("").str.contains(institution, case=False)
    rest = held[~drop]
    return {"auprc_all": average_precision_score(held.target, held.prob),
            "auprc_without": average_precision_score(rest.target, rest.prob),
            "auroc_without": roc_auc_score(rest.target, rest.prob),
            "dropped_images": int(drop.sum()), "dropped_melanomas": int(held.target[drop].sum())}
