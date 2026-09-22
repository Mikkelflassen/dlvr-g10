"""The four arms differ only here (E2 differs in the sampler instead)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on logits. gamma=0 reduces to plain BCE."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # probability assigned to the true class
        return ((1 - p_t) ** self.gamma * bce).mean()


def build_loss(arm, pos_weight=None, gamma=2.0, device="cpu"):
    """
    E0 / E0b : plain BCE            (E0b is E0 re-thresholded, no retraining)
    E1       : BCE with pos_weight
    E2       : plain BCE + balanced sampler (see data.make_loader)
    E3       : focal loss
    """
    if arm == "E1":
        if pos_weight is None:
            raise ValueError("E1 needs pos_weight")
        w = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=w)
    if arm == "E3":
        return FocalLoss(gamma)
    return nn.BCEWithLogitsLoss()
