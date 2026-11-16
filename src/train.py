"""Trains a single run (one setup on one fold) and stores it in a folder of its own.

Adapted from:
  [T] PyTorch "Transfer Learning for Computer Vision" tutorial, `train_model` (S. Chilamkurthy, BSD-3-Clause)
      https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
  [H] ISIC 2020 winning solution, `train.py` (Ha et al., 2020, MIT)
      https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/train.py
The way we work (a fixed seed, changing one thing at a time) comes from A. Karpathy:
  https://karpathy.github.io/2019/04/25/recipe/

`run` is the main function. Its steps are:
  1. settings     _add_defaults, _check_settings
  2. data         _load_data, _make_loaders
  3. network      _build_network, _build_optimizer
  4. training     _train_one_epoch, _score_epoch (and _resume / _save_state for long runs)
  5. results      _predictions, then preds.csv, history.csv and config.json
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
from sklearn.metrics import roc_auc_score, average_precision_score

from data import load_splits, inner_split, make_loader, sampling_weights
from models import build_model, default_lr
from losses import build_loss

SAMPLER_EXPERIMENTS = ("E2", "E5")   # these experiments use the balanced sampler
FOCAL_EXPERIMENTS = ("E3", "E5")     # these experiments use the focal loss


# records the code version that a run was made with
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


# runs with metadata take two inputs, all other runs take images only
def _forward(model, inputs, device):
    """A single forward pass."""
    if isinstance(inputs, (list, tuple)):
        return model(*[tensor.to(device, non_blocking=True) for tensor in inputs])
    return model(inputs.to(device, non_blocking=True))


# See also lab 2.
def val_epoch(model, loader, criterion, device, amp=False):
    """Adapted from [H] `val_epoch`. Gives back the loss and the probabilities.

    What we changed: sigmoid in place of softmax, no test-time augmentation, one loss value per image.
    """
    model.eval()
    val_loss = 0.0
    all_probs = []
    with torch.no_grad():
        for inputs, target in loader:
            target = target.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = _forward(model, inputs, device).squeeze(1)
            logits = logits.float()
            probs = logits.sigmoid()
            all_probs.append(probs.detach().cpu())

            loss = criterion(logits, target)
            val_loss += loss.item() * target.size(0)

    all_probs = torch.cat(all_probs).numpy()
    return val_loss / len(all_probs), all_probs


# the probabilities are all we need here
def predict(model, loader, device, amp=False):
    """Gives back the probabilities and nothing else."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = _forward(model, inputs, device).squeeze(1)
            probs = logits.float().sigmoid()
            all_probs.append(probs.detach().cpu())
    return torch.cat(all_probs).numpy()


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

# settings missing from the config get the defaults below (the grid ran with these defaults)
def _add_defaults(cfg, device):
    cfg.setdefault("seed", 0)
    cfg.setdefault("epochs", 5)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("lr", default_lr(cfg["depth"]))
    cfg.setdefault("augment", True)
    cfg.setdefault("aug", "light")                 # "light" came out best in our augmentation test (see data.py)

    # settings used by the extension
    cfg.setdefault("img_size", 256)
    cfg.setdefault("backbone", "resnet18")
    cfg.setdefault("extra_train", None)            # "isic2019": explained in isic2019.py

    # PanDerm with full fine-tuning. panderm.py has the recipe and its sources.
    if cfg["backbone"] == "panderm_base":
        cfg.setdefault("panderm_repo", "/content/PanDerm")
        cfg.setdefault("panderm_ckpt", "/content/panderm_bb_data6_checkpoint-499.pth")
        cfg.setdefault("weight_decay", 0.05)
        cfg.setdefault("layer_decay", 0.65)
        cfg.setdefault("drop_path", 0.2)
        cfg.setdefault("warmup_epochs", 1)
        cfg.setdefault("min_lr", 1e-6)

    # only panderm runs with mixed precision, since it is too slow without it
    cfg.setdefault("amp", cfg["backbone"] == "panderm_base" and device == "cuda")
    cfg.setdefault("extra_root", "/content/isic2019")

    if cfg["experiment"] == "E6":                         # the default settings of LibAUC, see auc_stage
        cfg.setdefault("auc_epochs", 4)
        cfg.setdefault("auc_lr", 0.1)
        cfg.setdefault("margin", 1.0)
        cfg.setdefault("epoch_decay", 0.002)
        cfg.setdefault("auc_weight_decay", 1e-5)

    cfg.setdefault("meta", False)                  # True: the image gets age, sex and body site as extra input (meta.py)
    cfg.setdefault("workers", min(8, os.cpu_count() or 2))   # extra cpus only speed up the loading
    cfg.setdefault("balanced", cfg["experiment"] in SAMPLER_EXPERIMENTS)    # E5: E2's sampler with E3's focal loss

    if cfg["experiment"] in SAMPLER_EXPERIMENTS:
        cfg.setdefault("mal_frac", 0.5)
        # "fixed" keeps one share for all epochs. "progressive" is progressively-balanced
        # sampling (Kang et al., 2020, Sec. 3, https://arxiv.org/abs/1910.09217), where the share
        # starts at the natural one in the first epoch and ends at mal_frac in the last.
        cfg.setdefault("sampling", "fixed")
    return cfg


# stop when a setting has no effect on this experiment, since the run name would then be wrong
def _check_settings(cfg):
    experiment = cfg["experiment"]
    if experiment == "E6" and cfg["backbone"] != "resnet18":
        raise ValueError("E6 is set up for ResNet-18 (model.fc)")
    if cfg.get("pos_weight") is not None and experiment != "E1":
        raise ValueError(f"pos_weight is set but the experiment is {experiment}; it only has an effect on E1")
    if cfg.get("mal_frac") is not None and experiment not in SAMPLER_EXPERIMENTS:
        raise ValueError(f"mal_frac is set but the experiment is {experiment}; it only has an effect on E2/E5")
    if cfg.get("sampling") is not None and experiment not in SAMPLER_EXPERIMENTS:
        raise ValueError(f"sampling is set but the experiment is {experiment}; it only has an effect on E2/E5")
    if experiment in SAMPLER_EXPERIMENTS and cfg["sampling"] not in ("fixed", "progressive"):
        raise ValueError(f"sampling must be 'fixed' or 'progressive', got {cfg['sampling']!r}")
    if (cfg.get("gamma") is not None or cfg.get("alpha") is not None) and experiment not in FOCAL_EXPERIMENTS:
        raise ValueError(f"gamma/alpha is set but the experiment is {experiment}; it only has an effect on E3/E5")


# examples: E0_full_f0_s0, and E0_full_final_s0 when it is the test set run
def _run_name(cfg, final):
    if final:
        name = f"{cfg['experiment']}_{cfg['depth']}_final_s{cfg['seed']}"
    else:
        name = f"{cfg['experiment']}_{cfg['depth']}_f{cfg['fold']}_s{cfg['seed']}"
    if cfg.get("tag"):
        name += "_" + cfg["tag"]
    return name


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------

def _load_data(cfg, splits_dir, data_root, final):
    """Gives back train_df, train_2020, inner_val, held and the metadata columns (None if unused).

    train_2020 is the same as train_df with the ISIC 2019 images taken out.
    """
    # Same folds as [H] `run`: train on df[df['fold'] != fold], hold out df[df['fold'] == fold].
    dev, test = load_splits(splits_dir, data_root)
    if final:
        # test set run: all dev images are used for training and there is no inner validation
        train_df, inner_val, held = dev.reset_index(drop=True), None, test.reset_index(drop=True)
    else:
        train_df, inner_val = inner_split(dev, cfg["fold"], splits_dir)
        held = dev[dev.fold == cfg["fold"]].reset_index(drop=True)
    train_2020 = train_df

    # the isic 2019 images are used for training only
    if cfg.get("extra_train") == "isic2019":
        from isic2019 import load_extra
        extra = load_extra(splits_dir, cfg["extra_root"])
        cfg.update({"n_extra": int(len(extra)), "n_extra_melanomas": int(extra.target.sum())})
        if cfg["meta"]:
            from meta import meta_2019
            extra = extra.merge(meta_2019(cfg["extra_root"]), on="image_name", how="left")
            train_df = train_df[["image_name", "target", "path",
                                 "sex", "age_approx", "anatom_site_general_challenge"]]
        else:
            train_df = train_df[["image_name", "target", "path"]]
        train_df = pd.concat([train_df, extra], ignore_index=True)
    elif cfg.get("extra_train"):
        raise ValueError(f"unknown extra_train {cfg['extra_train']!r}")

    meta_cols = None
    if cfg["meta"]:
        from meta import add_meta, META_COLS
        meta_cols = META_COLS
        train_df, held = add_meta(train_df, "train"), add_meta(held, "held")
        inner_val = None if inner_val is None else add_meta(inner_val, "inner_val")
        train_2020 = add_meta(train_2020, "train 2020")

    return train_df, train_2020, inner_val, held, meta_cols


def _make_loaders(cfg, train_df, train_2020, inner_val, held, meta_cols):
    """Builds the four data loaders and train_check (explained below)."""
    train_loader = make_loader(train_df, cfg["batch_size"], train=True, meta_cols=meta_cols,
                               augment=cfg["augment"], balanced=cfg["balanced"],
                               aug=cfg["aug"], workers=cfg["workers"],
                               mal_frac=cfg.get("mal_frac", 0.5), size=cfg["img_size"])

    # loaders for evaluation have no augmentation and no sampler
    def eval_loader(df):
        return make_loader(df, cfg["batch_size"], workers=cfg["workers"], size=cfg["img_size"],
                           meta_cols=meta_cols)

    val_loader = None if inner_val is None else eval_loader(inner_val)

    # a fixed sample of clean training images that we compare with inner validation
    n_check = 4500 if inner_val is None else len(inner_val)
    train_check = train_2020.sample(n=min(n_check, len(train_2020)),
                                    random_state=cfg["seed"]).reset_index(drop=True)
    check_loader = eval_loader(train_check)
    held_loader = eval_loader(held)
    return train_loader, val_loader, check_loader, held_loader, train_check


# ---------------------------------------------------------------------------
# 3. Network, loss and optimizer
# ---------------------------------------------------------------------------

def _build_network(cfg, device):
    if cfg["backbone"] != "panderm_base":
        if cfg["meta"]:
            raise ValueError("meta is only implemented for backbone panderm_base")
        return build_model(cfg["depth"], cfg["backbone"]).to(device)

    if cfg["depth"] != "full":
        raise ValueError("panderm_base in train.run is full fine-tuning; the frozen version is panderm.probe_fold")
    from panderm import build_finetune
    model = build_finetune(cfg["panderm_repo"], cfg["panderm_ckpt"], cfg["drop_path"])
    if cfg["meta"]:
        from meta import WithMeta
        model.head = torch.nn.Identity()       # WithMeta has its own head, which gets the 768 CLS features
        model = WithMeta(model, n_features=768)
    return model.to(device)


def _build_optimizer(model, cfg, steps_per_epoch):
    """Gives back the optimizer and one learning rate per step (None: the learning rate is fixed)."""
    if cfg["backbone"] == "panderm_base":
        # AdamW, with layer-wise decay and a per-step cosine schedule (panderm.py).
        # AdamW: Loshchilov & Hutter (2019), https://arxiv.org/abs/1711.05101
        from panderm import layer_decay_groups, cosine_lr
        groups = layer_decay_groups(model, cfg["weight_decay"], cfg["layer_decay"])
        optimizer = torch.optim.AdamW(groups, lr=cfg["lr"])
        lr_schedule = cosine_lr(cfg["lr"], cfg["min_lr"], cfg["epochs"], steps_per_epoch,
                                cfg["warmup_epochs"])
        return optimizer, lr_schedule

    # As in [T], we optimise only the parameters that require gradients.
    # See also lecture 6, Code 1.
    # Adam: Kingma & Ba (2015), https://arxiv.org/abs/1412.6980
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=cfg["lr"]), None


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

# See also lecture 2, Code 1 and lab 2 (training loop).
# Adapted from [T] `train_model`.
# The mixed precision part is adapted from the PyTorch examples:
#   https://docs.pytorch.org/docs/stable/notes/amp_examples.html
#   https://docs.pytorch.org/docs/stable/amp.html
# When amp=False the scaler is a no-op, so this is the ordinary step.
def _train_one_epoch(model, loader, criterion, optimizer, scaler, lr_schedule, step, device, amp):
    """Goes through the training images once. Gives back the mean loss and the step counter."""
    model.train()
    running_loss = 0.0
    n_seen = 0

    for inputs, labels in loader:
        labels = labels.to(device, non_blocking=True)

        # Adapted from PanDerm (classification/furnace/engine_for_finetuning.py).
        if lr_schedule is not None:
            for group in optimizer.param_groups:
                group["lr"] = lr_schedule[step] * group["lr_scale"]
        step += 1

        # reset the gradients
        optimizer.zero_grad()

        # forward pass
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            outputs = _forward(model, inputs, device).squeeze(1)
        loss = criterion(outputs.float(), labels)

        # backward pass and optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # running totals
        running_loss += loss.item() * labels.size(0)
        n_seen += labels.size(0)

    return running_loss / n_seen, step


# Same idea as lab 7: the training loss is measured on augmented images. We therefore measure
# it once more on clean training images, so it can be compared with inner validation.
def _score_epoch(model, criterion, cfg, device, loaders, frames):
    """Scores the clean training sample and inner validation. For the long training check it
    also scores the held-out fold, which is recorded but never used to pick the epoch.

    Each experiment uses its own loss here, so losses can only be compared within one experiment.
    """
    check_loader, val_loader, held_loader = loaders
    train_check, inner_val, held = frames

    clean_loss, clean_probs = val_epoch(model, check_loader, criterion, device, cfg["amp"])
    scores = {"train_clean_loss": float(clean_loss),
              "train_clean_auroc": float(roc_auc_score(train_check.target, clean_probs))}

    if inner_val is None:        # final run: there is no validation set
        scores.update({"val_loss": float("nan"), "val_auroc": float("nan")})
    else:
        val_loss, val_probs = val_epoch(model, val_loader, criterion, device, cfg["amp"])
        scores.update({"val_loss": float(val_loss),
                       "val_auroc": float(roc_auc_score(inner_val.target, val_probs))})

    held_scores = {}
    if cfg.get("track_held"):
        _, held_probs = val_epoch(model, held_loader, criterion, device, cfg["amp"])
        held_scores = {"held_auroc": float(roc_auc_score(held.target, held_probs)),
                       "held_auprc": float(average_precision_score(held.target, held_probs))}
    return scores, held_scores


# The resume code was written with AI (Claude). Based on lab 6 (save_checkpoint).
# resume_dir is meant for the long runs. Everything is saved after each epoch,
# so a colab stop costs us one epoch at most.
#   https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
def _save_state(path, model, optimizer, scaler, step, epoch, best_auc, history, minutes):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "step": step, "epoch": epoch,
             "best_auc": best_auc, "history": history, "minutes": minutes,
             "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
             "rng_python": random.getstate()}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)     # a half-written file can never be left behind


def _resume(path, model, optimizer, scaler, device):
    """Reads a saved state: step, best_auc, history, the next epoch and the minutes used so far."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    # map_location put the RNG state on the GPU, but set_rng_state wants it on the CPU
    torch.set_rng_state(state["rng_torch"].cpu().to(torch.uint8))
    np.random.set_state(state["rng_numpy"])
    random.setstate(state["rng_python"])
    return state["step"], state["best_auc"], state["history"], state["epoch"] + 1, state["minutes"]


# Written with AI (Claude), using LibAUC.
# The two stages are adapted from the LibAUC tutorial:
# https://github.com/Optimization-AI/LibAUC/blob/main/examples/02_v140_Optimizing_AUROC_with_ResNet20_on_Imbalanced_CIFAR10.ipynb
def auc_stage(model, cfg, train_df, train_loader, val_loader, inner_val, check_loader, train_check,
              device, history, since):
    """E6, stage 2: the plain BCE network is fine-tuned with the AUC-margin loss.

    Yuan et al. (ICCV 2021), https://arxiv.org/abs/2012.03173
    LibAUC: https://libauc.org, https://github.com/Optimization-AI/LibAUC
    The last layer is reset, and training uses AUCMLoss with PESG.
    Default settings of LibAUC: lr 0.1, margin 1.0, epoch_decay 0.002, weight_decay 1e-5.
    Halfway through, the learning rate is divided by 10.
    """
    import warnings
    from libauc.losses import AUCMLoss
    from libauc.optimizers import PESG

    model.fc.reset_parameters()                       # "re-initialise the classifier layer"
    aucm = AUCMLoss(margin=cfg["margin"], imratio=float(train_df.target.mean()))

    def criterion(logits, target):
        return aucm(torch.sigmoid(logits), target, auto=False)

    optimizer = PESG(model.parameters(), loss_fn=aucm, lr=cfg["auc_lr"], margin=cfg["margin"],
                     epoch_decay=cfg["epoch_decay"], weight_decay=cfg["auc_weight_decay"],
                     verbose=False)

    # Based on lecture 6, Code 5 (keep the best weights).
    best_auc, best_state = -1.0, None
    for epoch in range(cfg["auc_epochs"]):
        if epoch == cfg["auc_epochs"] // 2:
            optimizer.update_regularizer(decay_factor=10)

        model.train()
        running_loss, n_seen = 0.0, 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)   # a batch with no melanoma is normal here
            for inputs, labels in train_loader:
                labels = labels.to(device, non_blocking=True)
                loss = criterion(_forward(model, inputs, device).squeeze(1), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
                n_seen += labels.size(0)
            clean_loss, clean_probs = val_epoch(model, check_loader, criterion, device)
            if inner_val is not None:
                val_loss, val_probs = val_epoch(model, val_loader, criterion, device)
        clean_auc = roc_auc_score(train_check.target, clean_probs)

        if inner_val is None:            # final run: with no validation set we keep the last epoch
            val_loss, val_auc, select = float("nan"), float("nan"), float(epoch)
        else:
            val_auc = roc_auc_score(inner_val.target, val_probs)
            select = val_auc

        history.append({"epoch": cfg["epochs"] + epoch, "stage": "auc",
                        "train_loss": running_loss / n_seen, "train_clean_loss": float(clean_loss),
                        "train_clean_auroc": float(clean_auc), "val_loss": float(val_loss),
                        "val_auroc": float(val_auc), "minutes": round((time.time() - since) / 60, 2)})
        print(f"  AUC epoch {epoch}  loss {running_loss / n_seen:.4f}  "
              f"train-clean AUROC {clean_auc:.4f}  val AUROC {val_auc:.4f}")

        if select > best_auc:
            best_auc = select
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_auc


# ---------------------------------------------------------------------------
# 5. Results
# ---------------------------------------------------------------------------

# each image gets one prediction, as in [H] `evaluate.py`
def _predictions(model, final, inner_val, held, val_loader, held_loader, device, amp):
    def table(df, loader, split):
        return pd.DataFrame({"image_name": df.image_name, "target": df.target,
                             "prob": predict(model, loader, device, amp), "split": split})

    if final:
        return table(held, held_loader, "test")
    return pd.concat([table(inner_val, val_loader, "inner_val"),
                      table(held, held_loader, "held_fold")])


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------

# The setup code here was written with AI (Claude). Borrowed parts name their source above.
def run(cfg, splits_dir, data_root, runs_dir, device=None, resume_dir=None, epoch_hook=None):
    """Trains a single run. The settings are in cfg (experiment, depth, fold, seed, epochs and so on).

    `tag` goes into the folder name, which keeps test runs from overwriting each other.
    final=True: train on all dev images and predict the test set (results/test/protocol.md).
    resume_dir: the training state is saved after each epoch, so a long run can be continued.
    epoch_hook(out, epoch): runs after each epoch, for example to push history.csv.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. settings
    cfg = _add_defaults(dict(cfg), device)
    _check_settings(cfg)
    set_seed(cfg["seed"])
    final = bool(cfg.get("final"))
    name = _run_name(cfg, final)
    out = Path(runs_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    # 2. data
    train_df, train_2020, inner_val, held, meta_cols = _load_data(cfg, splits_dir, data_root, final)
    train_loader, val_loader, check_loader, held_loader, train_check = _make_loaders(
        cfg, train_df, train_2020, inner_val, held, meta_cols)

    # 3. network, loss and optimizer
    model = _build_network(cfg, device)
    criterion = build_loss(cfg["experiment"], cfg.get("pos_weight"), cfg.get("gamma", 2.0),
                           device, cfg.get("alpha", -1))
    optimizer, lr_schedule = _build_optimizer(model, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    # 4. training. As in [H], the epoch with the best inner validation AUROC is kept.
    # See also lecture 6, Code 7 (saving the best epoch).
    step = 0
    history = []
    since = time.time()
    with TemporaryDirectory() as tempdir:
        state_path = None
        if resume_dir:
            Path(resume_dir).mkdir(parents=True, exist_ok=True)
            best_model_params_path = os.path.join(resume_dir, f"{name}_best.pt")
            state_path = os.path.join(resume_dir, f"{name}_state.pt")
        else:
            best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        best_auc = -1.0
        start_epoch = 0
        if state_path and os.path.exists(state_path):
            step, best_auc, history, start_epoch, minutes = _resume(
                state_path, model, optimizer, scaler, device)
            since -= minutes * 60
            cfg["resumed_at_epoch"] = cfg.get("resumed_at_epoch", []) + [start_epoch]
            print(f"  resumed from {state_path}: epoch {start_epoch} of {cfg['epochs']}, "
                  f"best inner AUROC so far {best_auc:.4f}")
        else:
            torch.save(model.state_dict(), best_model_params_path)

        natural = float(train_df.target.mean())    # the true share of melanomas
        for epoch in range(start_epoch, cfg["epochs"]):
            share = None
            if cfg.get("sampling") == "progressive":
                share = natural + epoch / max(cfg["epochs"] - 1, 1) * (cfg["mal_frac"] - natural)
                train_loader.sampler.weights = sampling_weights(train_df.target.values, share)

            epoch_loss, step = _train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                                lr_schedule, step, device, cfg["amp"])
            scores, held_scores = _score_epoch(model, criterion, cfg, device,
                                               (check_loader, val_loader, held_loader),
                                               (train_check, inner_val, held))

            history.append({"epoch": epoch,
                            "train_loss": float(epoch_loss),    # during training, on augmented images
                            **scores,                           # on clean images and on inner validation
                            "minutes": round((time.time() - since) / 60, 2),
                            **({"mal_frac": round(share, 4)} if share is not None else {}),
                            **held_scores})
            print(f"  epoch {epoch}  loss {epoch_loss:.4f}  "
                  f"train-clean AUROC {scores['train_clean_auroc']:.4f}  "
                  f"val loss {scores['val_loss']:.4f}  val AUROC {scores['val_auroc']:.4f}")

            # remember the best epoch (in the final run that is always the last one)
            val_auc = scores["val_auroc"]
            if final or val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_params_path)

            pd.DataFrame(history).to_csv(out / "history.csv", index=False)   # what we have so far
            if state_path:
                _save_state(state_path, model, optimizer, scaler, step, epoch, best_auc, history,
                            (time.time() - since) / 60)
            if epoch_hook:
                epoch_hook(out, epoch)

        # go back to the best weights
        model.load_state_dict(torch.load(best_model_params_path, map_location=device))

    # E6: the stage above was plain BCE, and stage 2 now trains with the AUC-margin loss
    if cfg["experiment"] == "E6":
        cfg["stage1_best_inner_auroc"] = best_auc
        best_auc = auc_stage(model, cfg, train_df, train_loader, val_loader, inner_val, check_loader,
                             train_check, device, history, since)

    # 5. results
    preds = _predictions(model, final, inner_val, held, val_loader, held_loader, device, cfg["amp"])
    preds.to_csv(out / "preds.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    cfg.update({"run": name, "git": _git_hash(), "best_inner_auroc": None if final else best_auc,
                "minutes": round((time.time() - since) / 60, 2)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{name}: best inner AUROC {best_auc:.4f}  ({cfg['minutes']} min)")

    if state_path:                         # the run is done, so the resume files can go
        for path in (state_path, best_model_params_path):
            if os.path.exists(path):
                os.remove(path)
    return out
