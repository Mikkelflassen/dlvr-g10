"""Our ResNet-18 with one output.

Adapted from the PyTorch "Transfer Learning for Computer Vision" tutorial by S. Chilamkurthy (BSD-3-Clause):
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
  https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
ResNet: He et al. (2016), https://arxiv.org/abs/1512.03385
"""
import torch.nn as nn
from torchvision import models


BACKBONES = ("resnet18", "efficientnet_b0")


def build_model(depth="full", backbone="resnet18"):
    """depth: 'frozen' trains the last layer, 'full' trains the whole network."""
    if depth not in ("frozen", "full"):
        raise ValueError(f"depth must be 'frozen' or 'full', got {depth!r}")
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
    if backbone == "efficientnet_b0":
        return _efficientnet_b0(depth)

    # See also lecture 4, Code 13 and lecture 7, Code 1.
    model = models.resnet18(weights="IMAGENET1K_V1")

    # frozen and full are the tutorial's two transfer learning scenarios
    if depth == "frozen":
        for param in model.parameters():
            param.requires_grad = False

    # one output with BCE, the tutorial uses 2 with cross-entropy
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model


def _efficientnet_b0(depth):
    """EfficientNet-B0 (Tan & Le, 2019, https://arxiv.org/abs/1905.11946) with one output.

    The ISIC 2020 winners used this family (Ha et al., 2020, https://arxiv.org/abs/2010.05351).
      https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    if depth == "frozen":
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    return model


# Lower learning rate for fine-tuning: see lecture 7, section 1.4.
def default_lr(depth):
    return 1e-3 if depth == "frozen" else 1e-4
