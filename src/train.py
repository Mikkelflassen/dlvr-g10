"""One run = one config + one fold. Writes a run folder; computes no metrics.

Everything the report needs is derived later from preds.csv by eval.py.
"""
import json
import subprocess
import time
from pathlib import Path

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


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for x, _ in loader:
        logits = model(x.to(device, non_blocking=True)).squeeze(1)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def run(cfg, splits_dir, data_root, runs_dir, device=None):
    """cfg keys: arm, depth, fold, seed, epochs, batch_size,
                 pos_weight (E1), gamma (E3), balanced (E2), augment, lr"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(cfg)
    cfg.setdefault("seed", 0)
    cfg.setdefault("epochs", 5)
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("lr", default_lr(cfg["depth"]))
    cfg.setdefault("augment", cfg["depth"] == "full")
    cfg.setdefault("balanced", cfg["arm"] == "E2")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    name = f"{cfg['arm']}_{cfg['depth']}_f{cfg['fold']}_s{cfg['seed']}"
    out = Path(runs_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    dev, _ = load_splits(splits_dir, data_root)
    tr, inner_val = inner_split(dev, cfg["fold"], seed=cfg["seed"])
    held = dev[dev.fold == cfg["fold"]].reset_index(drop=True)

    tr_loader = make_loader(tr, cfg["batch_size"], train=True,
                            augment=cfg["augment"], balanced=cfg["balanced"])
    val_loader = make_loader(inner_val, cfg["batch_size"])
    held_loader = make_loader(held, cfg["batch_size"])

    model = build_model(cfg["depth"]).to(device)
    criterion = build_loss(cfg["arm"], cfg.get("pos_weight"),
                           cfg.get("gamma", 2.0), device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    history, best_auc, t0 = [], -1.0, time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        losses = []
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_probs = predict(model, val_loader, device)
        val_auc = roc_auc_score(inner_val.target, val_probs)   # epoch choice: AUROC,
        history.append({"epoch": epoch,                        # not loss (not comparable
                        "train_loss": float(np.mean(losses)),  # across arms)
                        "val_auroc": float(val_auc),
                        "minutes": round((time.time() - t0) / 60, 2)})
        print(f"  epoch {epoch}  loss {np.mean(losses):.4f}  val AUROC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    preds = pd.concat([
        pd.DataFrame({"image_name": inner_val.image_name, "target": inner_val.target,
                      "prob": predict(model, val_loader, device), "split": "inner_val"}),
        pd.DataFrame({"image_name": held.image_name, "target": held.target,
                      "prob": predict(model, held_loader, device), "split": "held_fold"}),
    ])
    preds.to_csv(out / "preds.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    cfg.update({"run": name, "git": _git_hash(), "best_inner_auroc": best_auc,
                "minutes": round((time.time() - t0) / 60, 2)})
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "best.pt").unlink()  # answers are in preds.csv; model not needed after
    print(f"{name}: best inner AUROC {best_auc:.4f}  ({cfg['minutes']} min)")
    return out
