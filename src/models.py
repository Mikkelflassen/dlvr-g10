"""ResNet-18, one logit. Two fine-tuning depths, fixed across all arms.

Adapted from the PyTorch "Transfer Learning for Computer Vision" tutorial by
Sasank Chilamkurthy (BSD-3-Clause):
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
  https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
ResNet: He et al. (2016), https://arxiv.org/abs/1512.03385
"""
import torch.nn as nn
from torchvision import models


BACKBONES = ("resnet18", "efficientnet_b0")


def build_model(depth="full", backbone="resnet18"):
    """depth: 'frozen' = final layer only, 'full' = whole network.
    backbone: 'resnet18' (every run up to M12) or 'efficientnet_b0' (M12)."""
    if depth not in ("frozen", "full"):
        raise ValueError(f"depth must be 'frozen' or 'full', got {depth!r}")
    if backbone not in BACKBONES:
        raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
    if backbone == "efficientnet_b0":
        return _efficientnet_b0(depth)

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


def _efficientnet_b0(depth):
    """EfficientNet-B0 (Tan & Le, 2019, https://arxiv.org/abs/1905.11946), ImageNet
    weights from torchvision, one logit. The family used by the ISIC 2020 winners
    (Ha et al., 2020, https://arxiv.org/abs/2010.05351), who used B3-B7 at larger
    sizes; B0 is the smallest member and fits our GPU budget.
    torchvision's classifier is Sequential(Dropout(0.2), Linear(1280, 1000)); only
    the Linear layer is replaced, the same change as for ResNet-18 above.
      https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    if depth == "frozen":
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    return model


def default_lr(depth):
    return 1e-3 if depth == "frozen" else 1e-4
