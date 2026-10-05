"""One run = one config + one fold. Writes a run folder; computes no metrics.

Everything the report needs is derived later from preds.csv by eval.py.

The training loop is adapted from two sources:
  [T] PyTorch "Transfer Learning for Computer Vision" tutorial, `train_model`
      (S. Chilamkurthy, BSD-3-Clause)
      https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
  [H] ISIC 2020 winning solution, `train.py` (Ha et al., 2020, MIT)
      https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/train.py
The workflow around it (fixed seed, one change at a time, evaluate on the whole
validation set) follows A. Karpathy, "A Recipe for Training Neural Networks":
  https://karpathy.github.io/2019/04/25/recipe/
"""
import json
import os
import random
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from data import load_splits, inner_split, make_loader
from models import build_model, default_lr
from losses import build_loss


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def set_seed(seed=0):
    """From [H] `set_seed`, unchanged."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def val_epoch(model, loader, criterion, device):
    """Adapted from [H] `val_epoch`: loss and probabilities over the whole loader.

    Changes: sigmoid on one logit instead of softmax, no test-time augmentation,
    and the loss is averaged per image instead of per batch.
    """
    model.eval()
    val_loss = 0.0
    PROBS = []
    with torch.no_grad():
        for (data, target) in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data).squeeze(1)
            probs = logits.sigmoid()
            PROBS.append(probs.detach().cpu())

            loss = criterion(logits, target)
            val_loss += loss.item() * data.size(0)

    PROBS = torch.cat(PROBS).numpy()
    return val_loss / len(PROBS), PROBS


def predict(model, loader, device):
    """Probabilities only — the same loop as `val_epoch` without the loss."""
    model.eval()
    PROBS = []
    with torch.no_grad():
        for (data, target) in loader:
            data = data.to(device)
            logits = model(data).squeeze(1)
            probs = logits.sigmoid()
            PROBS.append(probs.detach().cpu())
    return torch.cat(PROBS).numpy()


def run(cfg, splits_dir, data_root, runs_dir, device=None):
    """cfg keys: arm, depth, fold, seed, epochs, batch_size,
                 pos_weight (E1), gamma / alpha (E3), balanced (E2), augment, aug, lr, tag

    `tag` appends to the run folder name, so the same arm/fold/seed can be run
    more than once with different settings without overwriting itself.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(cfg)
    cfg.setdefault("seed", 0)
    cfg.setdefault("epochs", 5)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("lr", default_lr(cfg["depth"]))
    cfg.setdefault("augment", True)
    cfg.setdefault("aug", "base")   # step on the augmentation ladder, see data.make_transform
    cfg.setdefault("balanced", cfg["arm"] == "E2")

    set_seed(cfg["seed"])

    name = f"{cfg['arm']}_{cfg['depth']}_f{cfg['fold']}_s{cfg['seed']}"
    if cfg.get("tag"):
        name += "_" + cfg["tag"]
    out = Path(runs_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    # Folds as in [H] `run`: df[df['fold'] != fold] trains, df[df['fold'] == fold] is held out.
    dev, _ = load_splits(splits_dir, data_root)
    tr, inner_val = inner_split(dev, cfg["fold"], seed=cfg["seed"])
    held = dev[dev.fold == cfg["fold"]].reset_index(drop=True)

    train_loader = make_loader(tr, cfg["batch_size"], train=True,
                               augment=cfg["augment"], balanced=cfg["balanced"],
                               aug=cfg["aug"])
    val_loader = make_loader(inner_val, cfg["batch_size"])
    held_loader = make_loader(held, cfg["batch_size"])

    model = build_model(cfg["depth"]).to(device)
    criterion = build_loss(cfg["arm"], cfg.get("pos_weight"), cfg.get("gamma", 2.0),
                           device, cfg.get("alpha", -1))
    # Only parameters that require gradients are optimised, as in [T].
    # Adam: Kingma & Ba (2015), https://arxiv.org/abs/1412.6980
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])

    history = []
    since = time.time()

    # Training loop adapted from [T] `train_model`: a temporary checkpoint holds the
    # best epoch and is reloaded at the end. The best epoch is chosen by validation
    # AUROC as in [H] (`if auc > auc_max: torch.save(...)`), not by accuracy as in [T]
    # — and not by loss, which is not comparable across arms.
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_auc = -1.0

        for epoch in range(cfg["epochs"]):
            model.train()  # Set model to training mode
            running_loss = 0.0
            n_seen = 0

            # Iterate over data.
            for inputs, labels in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, labels)

                # backward + optimize
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                n_seen += inputs.size(0)

            epoch_loss = running_loss / n_seen
            # val_loss uses this arm's own loss, so compare it only within an arm.
            # train_loss is measured on augmented images, val_loss on clean ones.
            val_loss, val_probs = val_epoch(model, val_loader, criterion, device)
            val_auc = roc_auc_score(inner_val.target, val_probs)

            history.append({"epoch": epoch,
                            "train_loss": float(epoch_loss),
                            "val_loss": float(val_loss),
                            "val_auroc": float(val_auc),
                            "minutes": round((time.time() - since) / 60, 2)})
            print(f"  epoch {epoch}  loss {epoch_loss:.4f}  val loss {val_loss:.4f}  val AUROC {val_auc:.4f}")

            # keep the best epoch
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_params_path)

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, map_location=device))

    # Per-image predictions are saved so that every metric can be computed later,
    # like the out-of-fold prediction files written by [H] `evaluate.py`.
    preds = pd.concat([
        pd.DataFrame({"image_name": inner_val.image_name, "target": inner_val.target,
                      "prob": predict(model, val_loader, device), "split": "inner_val"}),
        pd.DataFrame({"image_name": held.image_name, "target": held.target,
                      "prob": predict(model, held_loader, device), "split": "held_fold"}),
    ])
    preds.to_csv(out / "preds.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    cfg.update({"run": name, "git": _git_hash(), "best_inner_auroc": best_auc,
                "minutes": round((time.time() - since) / 60, 2)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{name}: best inner AUROC {best_auc:.4f}  ({cfg['minutes']} min)")
    return out
