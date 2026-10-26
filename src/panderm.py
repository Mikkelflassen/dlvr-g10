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


def _modeling(panderm_repo):
    """PanDerm's model definitions, loaded once from the cloned repo."""
    if "panderm_modeling" in sys.modules:
        return sys.modules["panderm_modeling"]
    path = Path(panderm_repo) / "classification" / "models" / "modeling_finetune.py"
    spec = importlib.util.spec_from_file_location("panderm_modeling", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod     # timm's @register_model looks the module up here
    spec.loader.exec_module(mod)
    return mod


def _load_weights(model, checkpoint):
    """Copy the pretrained weights into `model` by name.

    Their builder loads with strict=False, which would also accept a file with
    none of the right keys; we check that every encoder weight was found.
    The pretraining head (if any) is dropped first: it never fits our one-logit head.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    state = {k.replace("module.", "").replace("encoder.", ""): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("head.")}
    result = model.load_state_dict(state, strict=False)
    # head: ours, new. pos_embed: fixed sin-cos values built in the code, not learned.
    missing = [k for k in result.missing_keys if not k.startswith("head.") and k != "pos_embed"]
    n_found = len(model.state_dict()) - len(result.missing_keys)
    print(f"PanDerm weights: {n_found} tensors loaded, {len(missing)} encoder tensors missing, "
          f"{len(result.unexpected_keys)} extra in file")
    assert not missing, f"encoder weights missing from the checkpoint, e.g. {missing[:5]}"


def load_panderm(panderm_repo, checkpoint, device):
    """PanDerm_Base with its pretrained weights, CLS-token features, eval mode (frozen use)."""
    model = _modeling(panderm_repo).panderm_base_patch16_224()
    _load_weights(model, checkpoint)
    model.head = torch.nn.Identity()
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Full fine-tuning (M12 steps panderm_ft, panderm_ft_isic2019), used by train.run.
# The recipe follows PanDerm's own fine-tuning script (classification/
# run_class_finetuning.py, README "Fine-tuning"): AdamW, weight decay 0.05,
# layer-wise learning-rate decay 0.65, stochastic depth (drop_path) 0.2, cosine
# schedule with linear warm-up, CLS token + linear head (init_scale 0.001,
# layer scale 0.1). Changed to fit our protocol and budget: 8 epochs instead of
# 50 (warm-up 1 instead of 10), batch 64 instead of 128 with the learning rate
# scaled linearly (5e-4 -> 2.5e-4, Goyal et al. 2017, https://arxiv.org/abs/1706.02677),
# plain BCE with one logit and no sampler (arm E0), our "light" augmentation
# instead of mixup/cutmix, best epoch by inner-validation AUROC as in every run.
# Layer-wise decay: Bao et al. 2022 (BEiT), https://arxiv.org/abs/2106.08254
# ---------------------------------------------------------------------------
def build_finetune(panderm_repo, checkpoint, drop_path=0.2):
    """PanDerm_Base for full fine-tuning with one output logit."""
    model = _modeling(panderm_repo).panderm_base_patch16_224_finetune(
        pretrained=False, num_classes=1, drop_rate=0.0, drop_path_rate=drop_path,
        attn_drop_rate=0.0, drop_block_rate=None, use_mean_pooling=False,
        init_scale=0.001, use_rel_pos_bias=False, init_values=0.1, lin_probe=False)
    _load_weights(model, checkpoint)
    return model


def _layer_id(name, n_layers):
    """Adapted from get_num_layer_for_vit (PanDerm classification/furnace/optim_factory.py):
    embeddings are layer 0, block i is layer i+1, everything after the blocks is the top."""
    if name in ("cls_token", "mask_token", "pos_embed") or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return n_layers + 1


def layer_decay_groups(model, weight_decay=0.05, layer_decay=0.65):
    """Parameter groups with a learning-rate scale per layer, adapted from
    get_parameter_groups + LayerDecayValueAssigner (same file): the head learns at
    the full rate, each block below it at layer_decay times the one above. No weight
    decay on biases, norms and tokens."""
    n_layers = len(model.blocks)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # inside meta.WithMeta the ViT is `backbone`; its metadata network and head count as the top
        name = name[len("backbone."):] if name.startswith("backbone.") else name
        no_wd = param.ndim == 1 or name.endswith(".bias") or name in ("cls_token", "pos_embed")
        lid = _layer_id(name, n_layers)
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if no_wd else weight_decay,
                           "lr_scale": layer_decay ** (n_layers + 1 - lid)}
        groups[key]["params"].append(param)
    return list(groups.values())


def cosine_lr(base, final, epochs, steps_per_epoch, warmup_epochs=1):
    """Learning rate per step: linear warm-up from 0, then cosine down to `final`.
    Adapted from cosine_scheduler (PanDerm classification/furnace/utils.py)."""
    warm = int(warmup_epochs * steps_per_epoch)
    total = epochs * steps_per_epoch
    rest = np.arange(total - warm)
    cos = final + 0.5 * (base - final) * (1 + np.cos(np.pi * rest / max(len(rest), 1)))
    return np.concatenate([np.linspace(0, base, warm, endpoint=False), cos])


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


def probe_fold(feats, dev, fold, splits_dir, runs_dir, extra_cfg=None, extra=None):
    """Train the linear probe for one held-out fold and write the run folder.

    The same splits as every other run: train / inner_val from inner_split.csv,
    the held-out fold is only predicted. C is chosen by inner-validation AUROC,
    the same criterion that picks the best epoch in train.run.

    extra: optional (features, targets) of ISIC 2019 images (isic2019.load_extra),
    added to the training part only, as in train.run with extra_train="isic2019".
    The run is then tagged panderm_isic2019.
    """
    start = time.time()
    index = pd.Series(np.arange(len(dev)), index=dev.image_name)
    tr, val = inner_split(dev, fold, splits_dir)
    held = dev[dev.fold == fold]
    X = {k: feats[index[d.image_name].values] for k, d in (("tr", tr), ("val", val), ("held", held))}
    y_tr = tr.target.values
    tag = "panderm"
    if extra is not None:
        X["tr"] = np.concatenate([X["tr"], extra[0]])
        y_tr = np.concatenate([y_tr, extra[1]])
        tag = "panderm_isic2019"

    scaler = StandardScaler().fit(X["tr"])
    X = {k: scaler.transform(v) for k, v in X.items()}

    history, best = [], None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000).fit(X["tr"], y_tr)
        p_val = clf.predict_proba(X["val"])[:, 1]
        row = {"C": C, "val_auroc": roc_auc_score(val.target, p_val),
               "val_auprc": average_precision_score(val.target, p_val)}
        history.append(row)
        print(f"  fold {fold}  C={C:g}  inner AUROC {row['val_auroc']:.3f}  AUPRC {row['val_auprc']:.3f}")
        if best is None or row["val_auroc"] > best[0]["val_auroc"]:
            best = (row, clf)
    row, clf = best

    cfg = {"arm": "E0", "depth": "frozen", "backbone": "panderm_base", "img_size": 224,
           "fold": fold, "seed": 0, "tag": tag, "probe": "logistic_regression",
           "C": row["C"], "C_grid": C_GRID, "augment": False, **(extra_cfg or {})}
    if extra is not None:
        cfg.update({"extra_train": "isic2019", "n_extra": int(len(extra[1])),
                    "n_extra_melanomas": int(np.sum(extra[1]))})
    name = f"E0_frozen_f{fold}_s0_{tag}"
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
