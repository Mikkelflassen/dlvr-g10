"""ResNet-18, one logit. Two fine-tuning depths, fixed across all arms."""
import torch.nn as nn
from torchvision import models


def build_model(depth="full"):
    """depth: 'frozen' = final layer only, 'full' = whole network."""
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if depth == "frozen":
        for p in m.parameters():
            p.requires_grad = False
    elif depth != "full":
        raise ValueError(f"depth must be 'frozen' or 'full', got {depth!r}")
    m.fc = nn.Linear(m.fc.in_features, 1)  # always trainable
    return m


def default_lr(depth):
    return 1e-3 if depth == "frozen" else 1e-4
