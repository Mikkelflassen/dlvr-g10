"""Trains one run (one setup, one fold) and saves it in its own folder.

Adapted from:
  [T] PyTorch "Transfer Learning for Computer Vision" tutorial, `train_model` (S. Chilamkurthy, BSD-3-Clause)
      https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
  [H] ISIC 2020 winning solution, `train.py` (Ha et al., 2020, MIT)
      https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/train.py
Our way of working (fixed seed, one change at a time) follows A. Karpathy:
  https://karpathy.github.io/2019/04/25/recipe/

The main function is `run`. It goes through these steps:
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

SAMPLER_ARMS = ("E2", "E5")   # arms that use the balanced sampler
FOCAL_ARMS = ("E3", "E5")     # arms that use the focal loss


# saves which code version a run used
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


# metadata runs have two inputs, the normal runs only images
def _forward(model, inputs, device):
    """One forward pass."""
    if isinstance(inputs, (list, tuple)):
        return model(*[tensor.to(device, non_blocking=True) for tensor in inputs])
    return model(inputs.to(device, non_blocking=True))


# See also lab 2.
def val_epoch(model, loader, criterion, device, amp=False):
    """Adapted from [H] `val_epoch`. Returns the loss and the probabilities.

    Changes: sigmoid instead of softmax, no test-time augmentation, loss per image.
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


# we only need the probabilities here
def predict(model, loader, device, amp=False):
    """Only the probabilities."""
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

# anything not in the config gets these defaults (the defaults are what the grid used)
def _add_defaults(cfg, device):
    cfg.setdefault("seed", 0)
    cfg.setdefault("epochs", 5)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("lr", default_lr(cfg["depth"]))
    cfg.setdefault("augment", True)
    cfg.setdefault("aug", "light")                 # "light" won our augmentation test (see data.py)

    # settings for the extension
    cfg.setdefault("img_size", 256)
    cfg.setdefault("backbone", "resnet18")
    cfg.setdefault("extra_train", None)            # "isic2019": see isic2019.py

    # full fine-tuning of PanDerm. The recipe and the sources are in panderm.py.
    if cfg["backbone"] == "panderm_base":
        cfg.setdefault("panderm_repo", "/content/PanDerm")
        cfg.setdefault("panderm_ckpt", "/content/panderm_bb_data6_checkpoint-499.pth")
        cfg.setdefault("weight_decay", 0.05)
        cfg.setdefault("layer_decay", 0.65)
        cfg.setdefault("drop_path", 0.2)
        cfg.setdefault("warmup_epochs", 1)
        cfg.setdefault("min_lr", 1e-6)

    # mixed precision only for panderm, otherwise it takes too long
    cfg.setdefault("amp", cfg["backbone"] == "panderm_base" and device == "cuda")
    cfg.setdefault("extra_root", "/content/isic2019")

    if cfg["arm"] == "E6":                         # LibAUC's default settings, see auc_stage
        cfg.setdefault("auc_epochs", 4)
        cfg.setdefault("auc_lr", 0.1)
        cfg.setdefault("margin", 1.0)
        cfg.setdefault("epoch_decay", 0.002)
        cfg.setdefault("auc_weight_decay", 1e-5)

    cfg.setdefault("meta", False)                  # True: age, sex, body site next to the image (meta.py)
    cfg.setdefault("workers", min(8, os.cpu_count() or 2))   # more cpus just makes loading faster
    cfg.setdefault("balanced", cfg["arm"] in SAMPLER_ARMS)    # E5 = E2's sampler + E3's focal loss

    if cfg["arm"] in SAMPLER_ARMS:
        cfg.setdefault("mal_frac", 0.5)
        # "fixed" uses the same share every epoch. "progressive" is progressively-balanced
        # sampling (Kang et al., 2020, Sec. 3, https://arxiv.org/abs/1910.09217): the share
        # moves from the natural one (first epoch) to mal_frac (last epoch).
        cfg.setdefault("sampling", "fixed")
    return cfg


# stop if a setting does nothing on this arm, else the run gets a wrong name
def _check_settings(cfg):
    arm = cfg["arm"]
    if arm == "E6" and cfg["backbone"] != "resnet18":
        raise ValueError("E6 is set up for ResNet-18 (model.fc)")
    if cfg.get("pos_weight") is not None and arm != "E1":
        raise ValueError(f"pos_weight is set but the arm is {arm}; it only has an effect on E1")
    if cfg.get("mal_frac") is not None and arm not in SAMPLER_ARMS:
        raise ValueError(f"mal_frac is set but the arm is {arm}; it only has an effect on E2/E5")
    if cfg.get("sampling") is not None and arm not in SAMPLER_ARMS:
        raise ValueError(f"sampling is set but the arm is {arm}; it only has an effect on E2/E5")
    if arm in SAMPLER_ARMS and cfg["sampling"] not in ("fixed", "progressive"):
        raise ValueError(f"sampling must be 'fixed' or 'progressive', got {cfg['sampling']!r}")
    if (cfg.get("gamma") is not None or cfg.get("alpha") is not None) and arm not in FOCAL_ARMS:
        raise ValueError(f"gamma/alpha is set but the arm is {arm}; it only has an effect on E3/E5")


# for example E0_full_f0_s0, or E0_full_final_s0 for the test set
def _run_name(cfg, final):
    if final:
        name = f"{cfg['arm']}_{cfg['depth']}_final_s{cfg['seed']}"
    else:
        name = f"{cfg['arm']}_{cfg['depth']}_f{cfg['fold']}_s{cfg['seed']}"
    if cfg.get("tag"):
        name += "_" + cfg["tag"]
    return name


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------

def _load_data(cfg, splits_dir, data_root, final):
    """Returns train_df, train_2020, inner_val, held and the metadata columns (or None).

    train_2020 is train_df without the ISIC 2019 images.
    """
    # Folds as in [H] `run`: df[df['fold'] != fold] trains, df[df['fold'] == fold] is held out.
    dev, test = load_splits(splits_dir, data_root)
    if final:
        # the test set run: train on all dev images, no inner validation
        train_df, inner_val, held = dev.reset_index(drop=True), None, test.reset_index(drop=True)
    else:
        train_df, inner_val = inner_split(dev, cfg["fold"], splits_dir)
        held = dev[dev.fold == cfg["fold"]].reset_index(drop=True)
    train_2020 = train_df

    # isic 2019 images only go into training
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
    """Returns the four data loaders and train_check (see below)."""
    train_loader = make_loader(train_df, cfg["batch_size"], train=True, meta_cols=meta_cols,
                               augment=cfg["augment"], balanced=cfg["balanced"],
                               aug=cfg["aug"], workers=cfg["workers"],
                               mal_frac=cfg.get("mal_frac", 0.5), size=cfg["img_size"])

    # evaluation loaders: no augmentation, no sampler
    def eval_loader(df):
        return make_loader(df, cfg["batch_size"], workers=cfg["workers"], size=cfg["img_size"],
                           meta_cols=meta_cols)

    val_loader = None if inner_val is None else eval_loader(inner_val)

    # a fixed sample of clean training images, to compare with inner validation
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
        model.head = torch.nn.Identity()       # the 768 CLS features go to WithMeta's own head
        model = WithMeta(model, n_features=768)
    return model.to(device)


def _build_optimizer(model, cfg, steps_per_epoch):
    """Returns the optimizer and the learning rate for every step (None = fixed learning rate)."""
    if cfg["backbone"] == "panderm_base":
        # AdamW with layer-wise decay and a cosine schedule per step (panderm.py).
        # AdamW: Loshchilov & Hutter (2019), https://arxiv.org/abs/1711.05101
        from panderm import layer_decay_groups, cosine_lr
        groups = layer_decay_groups(model, cfg["weight_decay"], cfg["layer_decay"])
        optimizer = torch.optim.AdamW(groups, lr=cfg["lr"])
        lr_schedule = cosine_lr(cfg["lr"], cfg["min_lr"], cfg["epochs"], steps_per_epoch,
                                cfg["warmup_epochs"])
        return optimizer, lr_schedule

    # Only parameters that require gradients are optimised, as in [T].
    # See also lecture 6, Code 1.
    # Adam: Kingma & Ba (2015), https://arxiv.org/abs/1412.6980
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=cfg["lr"]), None


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

# See also lecture 2, Code 1 and lab 2 (training loop).
# Adapted from [T] `train_model`.
# Mixed precision: adapted from the PyTorch mixed precision examples:
#   https://docs.pytorch.org/docs/stable/notes/amp_examples.html
#   https://docs.pytorch.org/docs/stable/amp.html
# With amp=False the scaler does nothing and the step is the plain one.
def _train_one_epoch(model, loader, criterion, optimizer, scaler, lr_schedule, step, device, amp):
    """One pass over the training images. Returns the mean loss and the step counter."""
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

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            outputs = _forward(model, inputs, device).squeeze(1)
        loss = criterion(outputs.float(), labels)

        # backward + optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # statistics
        running_loss += loss.item() * labels.size(0)
        n_seen += labels.size(0)

    return running_loss / n_seen, step


def _score_epoch(model, criterion, cfg, device, loaders, frames):
    """Scores the clean training sample, inner validation and (only for the long training
    check) the held-out fold. The held-out fold is only recorded, never used to pick the epoch.

    The losses use this arm's own loss, so only compare them within an arm.
    """
    check_loader, val_loader, held_loader = loaders
    train_check, inner_val, held = frames

    clean_loss, clean_probs = val_epoch(model, check_loader, criterion, device, cfg["amp"])
    scores = {"train_clean_loss": float(clean_loss),
              "train_clean_auroc": float(roc_auc_score(train_check.target, clean_probs))}

    if inner_val is None:        # final run: no validation set
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
# resume_dir is for the long runs: we save everything after each epoch,
# so if colab stops we only lose that epoch.
#   https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
def _save_state(path, model, optimizer, scaler, step, epoch, best_auc, history, minutes):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "step": step, "epoch": epoch,
             "best_auc": best_auc, "history": history, "minutes": minutes,
             "rng_torch": torch.get_rng_state(), "rng_numpy": np.random.get_state(),
             "rng_python": random.getstate()}
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)     # never a half-written file


def _resume(path, model, optimizer, scaler, device):
    """Loads a saved state. Returns step, best_auc, history, the next epoch and the minutes so far."""
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    # map_location moved the RNG state to the GPU; set_rng_state needs it on the CPU
    torch.set_rng_state(state["rng_torch"].cpu().to(torch.uint8))
    np.random.set_state(state["rng_numpy"])
    random.setstate(state["rng_python"])
    return state["step"], state["best_auc"], state["history"], state["epoch"] + 1, state["minutes"]


# Written with AI (Claude), using LibAUC.
# The two stages are adapted from the LibAUC tutorial:
# https://github.com/Optimization-AI/LibAUC/blob/main/examples/02_v140_Optimizing_AUROC_with_ResNet20_on_Imbalanced_CIFAR10.ipynb
def auc_stage(model, cfg, train_df, train_loader, val_loader, inner_val, check_loader, train_check,
              device, history, since):
    """E6, stage 2: fine-tune the plain BCE network with the AUC-margin loss.

    Yuan et al. (ICCV 2021), https://arxiv.org/abs/2012.03173
    LibAUC: https://libauc.org, https://github.com/Optimization-AI/LibAUC
    We reset the last layer and train with AUCMLoss and PESG.
    LibAUC's default settings: lr 0.1, margin 1.0, epoch_decay 0.002, weight_decay 1e-5.
    The learning rate is divided by 10 halfway.
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
            warnings.simplefilter("ignore", UserWarning)   # batches without a melanoma are expected
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

        if inner_val is None:            # final run: no validation set, the last epoch is kept
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

# one prediction per image, as in [H] `evaluate.py`
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
    """Trains one run. cfg holds the settings (arm, depth, fold, seed, epochs and so on).

    `tag` is added to the folder name, so test runs do not overwrite each other.
    final=True trains on all dev images and predicts the test set (results/test/protocol.md).
    resume_dir: save the training state after every epoch, so a long run can continue.
    epoch_hook(out, epoch): called after every epoch, for example to push history.csv.
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
    criterion = build_loss(cfg["arm"], cfg.get("pos_weight"), cfg.get("gamma", 2.0),
                           device, cfg.get("alpha", -1))
    optimizer, lr_schedule = _build_optimizer(model, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    # 4. training. We keep the epoch with the best inner validation AUROC, as in [H].
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

        natural = float(train_df.target.mean())    # the real share of melanomas
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
                            "train_loss": float(epoch_loss),    # augmented, during training
                            **scores,                           # clean images, inner validation
                            "minutes": round((time.time() - since) / 60, 2),
                            **({"mal_frac": round(share, 4)} if share is not None else {}),
                            **held_scores})
            print(f"  epoch {epoch}  loss {epoch_loss:.4f}  "
                  f"train-clean AUROC {scores['train_clean_auroc']:.4f}  "
                  f"val loss {scores['val_loss']:.4f}  val AUROC {scores['val_auroc']:.4f}")

            # keep the best epoch (final run: always the last epoch)
            val_auc = scores["val_auroc"]
            if final or val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_params_path)

            pd.DataFrame(history).to_csv(out / "history.csv", index=False)   # progress so far
            if state_path:
                _save_state(state_path, model, optimizer, scaler, step, epoch, best_auc, history,
                            (time.time() - since) / 60)
            if epoch_hook:
                epoch_hook(out, epoch)

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, map_location=device))

    # E6: stage 1 above was plain BCE; stage 2 trains with the AUC-margin loss
    if cfg["arm"] == "E6":
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

    if state_path:                         # run finished: the resume files are not needed
        for path in (state_path, best_model_params_path):
            if os.path.exists(path):
                os.remove(path)
    return out
