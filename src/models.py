"""ResNet-18, one logit. Two fine-tuning depths, fixed across all arms.

Adapted from the PyTorch "Transfer Learning for Computer Vision" tutorial by
Sasank Chilamkurthy (BSD-3-Clause):
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
  https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
ResNet: He et al. (2016), https://arxiv.org/abs/1512.03385
"""
import torch.nn as nn
from torchvision import models


def build_model(depth="full"):
    """depth: 'frozen' = final layer only, 'full' = whole network."""
    if depth not in ("frozen", "full"):
        raise ValueError(f"depth must be 'frozen' or 'full', got {depth!r}")

    model = models.resnet18(weights="IMAGENET1K_V1")

    # 'frozen' is the tutorial's "ConvNet as fixed feature extractor";
    # 'full' is its "Finetuning the ConvNet" (nothing frozen).
    if depth == "frozen":
        for param in model.parameters():
            param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default.
    # The tutorial uses 2 outputs with cross-entropy; we use 1 logit with BCE.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model


def default_lr(depth):
    return 1e-3 if depth == "frozen" else 1e-4
