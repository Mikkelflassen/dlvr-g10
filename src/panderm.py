"""PanDerm, frozen and fully fine-tuned.

PanDerm (Yan et al., 2025, Nature Medicine, https://arxiv.org/abs/2410.15038) is a ViT
pretrained on about 2 million skin images. We use PanDerm_Base (ViT-B/16).
Code and weights: https://github.com/SiyuanYan1/PanDerm
  - model: classification/models/modeling_finetune.py
  - loading and transform: classification/models/builder.py
  - linear probe: panderm_model/downstream/eval_features/linear_probe.py
    We use scikit-learn's LogisticRegression:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

PanDerm may have seen the Queensland images in ISIC 2020 (see check_institutions.py),
so we also score without them (`score_without`).
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

# PanDerm code and weights: CC-BY-NC-ND 4.0 licence (see their README).
from data import inner_split

# PanDerm's eval transform (builder.get_eval_transforms). Note std 0.228, as in their code.
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)),
])
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


# Adapted from the Python docs: https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
def _modeling(panderm_repo):
    """Loads PanDerm's model code from the clone."""
    if "panderm_modeling" in sys.modules:
        return sys.modules["panderm_modeling"]
    path = Path(panderm_repo) / "classification" / "models" / "modeling_finetune.py"
    spec = importlib.util.spec_from_file_location("panderm_modeling", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module     # timm's @register_model looks the module up here
    spec.loader.exec_module(module)
    return module


# The key clean-up and the missing-weights check were written with AI (Claude).
def _load_weights(model, checkpoint):
    """Copies the pretrained weights into the model and checks that none are missing."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    # the file may name the weights "module.x" or "encoder.x"; our model calls them "x"
    state = {key.replace("module.", "").replace("encoder.", ""): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if not key.startswith("head.")}
    result = model.load_state_dict(state, strict=False)
    # head: ours, new. pos_embed: fixed sin-cos values built in the code, not learned.
    missing = [key for key in result.missing_keys if not key.startswith("head.") and key != "pos_embed"]
    n_found = len(model.state_dict()) - len(result.missing_keys)
    print(f"PanDerm weights: {n_found} tensors loaded, {len(missing)} encoder tensors missing, "
          f"{len(result.unexpected_keys)} extra in file")
    assert not missing, f"encoder weights missing from the checkpoint, e.g. {missing[:5]}"


def load_panderm(panderm_repo, checkpoint, device):
    """PanDerm_Base, frozen, giving features."""
    model = _modeling(panderm_repo).panderm_base_patch16_224()
    _load_weights(model, checkpoint)
    # Based on lab 3 (nn.Identity to get the features).
    model.head = torch.nn.Identity()
    return model.eval().to(device)


# ---------------------------------------------------------------------------
# Full fine-tuning. The recipe is from PanDerm's own script (classification/run_class_finetuning.py):
# AdamW, weight decay 0.05, layer decay 0.65, drop_path 0.2, cosine schedule with warm-up.
# We changed: 8 epochs (not 50), warm-up 1, batch 64 with lr 2.5e-4 (Goyal et al. 2017,
# https://arxiv.org/abs/1706.02677), plain BCE, and our "light" augmentation.
# Layer-wise decay: Bao et al. 2022 (BEiT), https://arxiv.org/abs/2106.08254
# ---------------------------------------------------------------------------
def build_finetune(panderm_repo, checkpoint, drop_path=0.2):
    """PanDerm_Base for fine-tuning, with one output."""
    model = _modeling(panderm_repo).panderm_base_patch16_224_finetune(
        pretrained=False, num_classes=1, drop_rate=0.0, drop_path_rate=drop_path,
        attn_drop_rate=0.0, drop_block_rate=None, use_mean_pooling=False,
        init_scale=0.001, use_rel_pos_bias=False, init_values=0.1, lin_probe=False)
    _load_weights(model, checkpoint)
    return model


# PanDerm's version comes from BEiT (MIT licence): https://github.com/microsoft/unilm/tree/master/beit
def _layer_id(name, n_layers):
    """Adapted from get_num_layer_for_vit (PanDerm classification/furnace/optim_factory.py)."""
    if name in ("cls_token", "mask_token", "pos_embed") or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return n_layers + 1


# See also lecture 7, Code 1.
def layer_decay_groups(model, weight_decay=0.05, layer_decay=0.65):
    """Lower learning rates for earlier layers. Adapted from get_parameter_groups and LayerDecayValueAssigner (same file)."""
    n_layers = len(model.blocks)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # inside meta.WithMeta the ViT is `backbone`; its metadata network and head count as the top
        name = name[len("backbone."):] if name.startswith("backbone.") else name
        no_decay = param.ndim == 1 or name.endswith(".bias") or name in ("cls_token", "pos_embed")
        layer = _layer_id(name, n_layers)
        key = (layer, no_decay)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if no_decay else weight_decay,
                           "lr_scale": layer_decay ** (n_layers + 1 - layer)}   # earlier layer = smaller
        groups[key]["params"].append(param)
    return list(groups.values())


# See also lecture 6, Code 2. PanDerm's version comes from BEiT.
def cosine_lr(base, final, epochs, steps_per_epoch, warmup_epochs=1):
    """Warm-up, then cosine decay. Adapted from cosine_scheduler (PanDerm classification/furnace/utils.py)."""
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_steps = epochs * steps_per_epoch
    warmup = np.linspace(0, base, warmup_steps, endpoint=False)     # 0 up to base
    after = np.arange(total_steps - warmup_steps)
    cosine = final + 0.5 * (base - final) * (1 + np.cos(np.pi * after / max(len(after), 1)))
    return np.concatenate([warmup, cosine])


# only images, we don't need labels for the features
class _Images(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return TRANSFORM(Image.open(self.paths[index]).convert("RGB"))


# runs every image through panderm once and keeps the features
@torch.no_grad()
def extract(model, df, device, batch_size=128, workers=2):
    """One 768-number feature per image."""
    loader = DataLoader(_Images(df.path), batch_size=batch_size, num_workers=workers)
    features, start = [], time.time()
    for batch, images in enumerate(loader):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            features.append(model(images.to(device)).float().cpu())

        if batch % 50 == 0:
            print(f"  {batch * batch_size}/{len(df)} images, {(time.time() - start) / 60:.1f} min")

    return torch.cat(features).numpy()


# Written with AI (Claude). See also lecture 4, Code 13.
def probe_fold(feats, dev, fold, splits_dir, runs_dir, extra_cfg=None, extra=None):
    """Trains the linear classifier for one fold. C is chosen on inner validation."""
    start = time.time()
    row_of = pd.Series(np.arange(len(dev)), index=dev.image_name)   # image name -> row in feats
    train, val = inner_split(dev, fold, splits_dir)
    held = dev[dev.fold == fold]
    X = {part: feats[row_of[df.image_name].values]
         for part, df in (("train", train), ("val", val), ("held", held))}
    y_train = train.target.values
    tag = "panderm"
    if extra is not None:                    # extra = (features, targets) of ISIC 2019
        X["train"] = np.concatenate([X["train"], extra[0]])
        y_train = np.concatenate([y_train, extra[1]])
        tag = "panderm_isic2019"

    # scale the features with the training mean and std
    scaler = StandardScaler().fit(X["train"])
    X = {part: scaler.transform(features) for part, features in X.items()}

    # try each C and keep the one with the best inner validation AUROC
    history, best = [], None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=3000).fit(X["train"], y_train)
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


# same score but without one hospital, for the leakage check
def score_without(run_dir, attribution_csv, institution="Queensland"):
    """AUPRC and AUROC with and without one institution."""
    preds = pd.read_csv(Path(run_dir) / "preds.csv")
    held = preds[preds.split == "held_fold"].merge(
        pd.read_csv(attribution_csv)[["image_name", "attribution"]],
        on="image_name",
        how="left",
    )

    drop = held.attribution.fillna("").str.contains(institution, case=False)
    rest = held[~drop]

    return {
        "auprc_all": average_precision_score(held.target, held.prob),
        "auprc_without": average_precision_score(rest.target, rest.prob),
        "auroc_without": roc_auc_score(rest.target, rest.prob),
        "dropped_images": int(drop.sum()),
        "dropped_melanomas": int(held.target[drop].sum()),
    }
