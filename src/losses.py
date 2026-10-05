"""The four arms differ only here (E2 differs in the sampler instead)."""
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
    """
    if arm == "E1":
        # Library: torch.nn.BCEWithLogitsLoss, pos_weight argument.
        #   https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        if pos_weight is None:
            raise ValueError("E1 needs pos_weight")
        w = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=w)
    if arm == "E3":
        # Library: torchvision.ops.sigmoid_focal_loss — focal loss of Lin et al. (2017).
        #   https://arxiv.org/abs/1708.02002
        #   https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py
        # gamma is the focusing parameter (gamma=0 is plain BCE). alpha is the
        # class-balance weight from the paper; -1 switches it off, so E3 tests
        # focusing alone and does not overlap with E1's class weighting.
        # Calibration motivation: Mukhoti et al. (2020), https://arxiv.org/abs/2002.09437
        return partial(sigmoid_focal_loss, alpha=alpha, gamma=gamma, reduction="mean")
    return nn.BCEWithLogitsLoss()
