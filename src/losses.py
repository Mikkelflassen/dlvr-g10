"""The loss for each setup."""
from functools import partial

import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


def build_loss(arm, pos_weight=None, gamma=2.0, device="cpu", alpha=-1):
    """
    E0 / E0b : plain BCE            (E0b is E0 re-thresholded, no retraining)
    E1       : BCE with pos_weight
    E2       : plain BCE + balanced sampler (see data.make_loader)
    E3       : focal loss
    E5       : focal loss + E2's sampler (E2 and E3 combined)
    E6       : plain BCE in stage 1; stage 2 uses the AUC-margin loss (train.auc_stage)
    """
    if arm == "E1":
        # Library: torch.nn.BCEWithLogitsLoss, pos_weight argument.
        #   https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        if pos_weight is None:
            raise ValueError("E1 needs pos_weight")
        w = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=w)
    # See also lecture 8, Code 5 (focal loss). Here we use the torchvision version.
    if arm in ("E3", "E5"):
        # Library: torchvision.ops.sigmoid_focal_loss, the focal loss of Lin et al. (2017).
        #   https://arxiv.org/abs/1708.02002
        #   https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py
        # alpha -1 means off, so E3 does not overlap with E1
        # Calibration motivation: Mukhoti et al. (2020), https://arxiv.org/abs/2002.09437
        return partial(sigmoid_focal_loss, alpha=alpha, gamma=gamma, reduction="mean")
    # Based on lecture 2, Code 5 and lecture 3, Code 4.
    return nn.BCEWithLogitsLoss()
