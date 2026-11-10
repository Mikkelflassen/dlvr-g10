"""Our dataset and data loaders. The splits are read from files.

Data: ISIC 2020 (Rotemberg et al., 2021, https://doi.org/10.1038/s41597-021-00815-z),
256x256 version by Chris Deotte: https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# See also lab 2.
# ImageNet statistics, as in the PyTorch transfer learning tutorial (see make_transform).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# reads our fixed splits and adds where each image is
def load_splits(splits_dir, data_root):
    """Reads the dev and test sets and adds the path of each image."""
    splits_dir, data_root = Path(splits_dir), Path(data_root)
    out = []
    for name in ("dev_split.csv", "test_split.csv"):
        df = pd.read_csv(splits_dir / name)
        df["path"] = df.image_name.map(lambda n: str(data_root / "train" / f"{n}.jpg"))
        out.append(df)
    return out


# Train, validation and test sets: see lecture 2, section 3.
def inner_split(dev, test_fold, splits_dir):
    """Splits the training folds into train and inner validation (splits/inner_split.csv).

    Inner validation chooses the best epoch, the threshold and the settings.
    Why these choices need their own split: Cawley & Talbot (2010),
      https://jmlr.org/papers/v11/cawley10a.html
    """
    train_folds = dev[dev.fold != test_fold].reset_index(drop=True)
    inner = pd.read_csv(Path(splits_dir) / "inner_split.csv")
    val_names = inner.loc[inner.test_fold == test_fold, "image_name"]
    is_val = train_folds.image_name.isin(val_names)
    train = train_folds[~is_val].reset_index(drop=True)
    val = train_folds[is_val].reset_index(drop=True)

    # checks: nothing missing, and no patient on both sides
    assert is_val.sum() == len(val_names), "inner_split.csv does not match dev_split.csv"
    assert not set(train.patient_id) & set(val.patient_id), "a patient is in both train and inner_val"
    return train, val


# The augmentation ladder: each step keeps everything from the steps before it.
AUG_STEPS = ["none", "base", "geo", "light", "colour", "cutout", "randaug"]


def _transpose(img):
    """Flips the image along the diagonal."""
    return img.transpose(Image.TRANSPOSE)


# Converted from albumentations to torchvision with AI (Claude).
# See also lecture 6, Code 3.
def make_transform(train, augment=False, aug="base", size=256):
    """Training augmentation. Validation and test images are not augmented.

    `aug` picks a step. Each step adds to the one before:
      none    nothing
      base    crop + flips
      geo     + transpose, shift / scale / rotate
      light   + brightness, contrast (the one we used)
      colour  + hue, saturation
      cutout  + one erased square. DeVries & Taylor (2017), https://arxiv.org/abs/1708.04552
      randaug base + RandAugment. Cubuk et al. (2020), https://arxiv.org/abs/1909.13719
              https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html

    base is adapted from `data_transforms` in the PyTorch transfer learning tutorial (BSD-3-Clause):
      https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
    The other steps use the operations and strengths of `get_transforms` in Ha et al. (2020, MIT):
      https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
      https://arxiv.org/abs/2010.05351
    They use albumentations and we use torchvision, so it is close but not exact.
      https://docs.pytorch.org/vision/stable/transforms.html
    """
    if aug not in AUG_STEPS:
        raise ValueError(f"aug must be one of {AUG_STEPS}, got {aug!r}")
    level = AUG_STEPS.index(aug)
    if aug == "randaug":
        level = 1  # base only; RandAugment replaces the hand-picked steps

    on_image, on_tensor = [], []   # steps before and after the image becomes a tensor
    if train and augment and level >= 1:
        # base
        on_image += [
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),   # Ha et al.: VerticalFlip(p=0.5)
        ]
        if aug == "randaug":
            # torchvision defaults, which are the paper's ImageNet ResNet-50 setting
            # (N=2 operations, magnitude M=9 out of 30).
            on_image += [transforms.RandAugment(num_ops=2, magnitude=9)]
        if level >= 2:  # geo
            on_image += [
                # Ha et al.: Transpose(p=0.5)
                transforms.RandomApply([transforms.Lambda(_transpose)], p=0.5),
                # Ha et al.: ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                #                             rotate_limit=15, border_mode=0, p=0.85)
                transforms.RandomApply([transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.85),
            ]
        if level >= 3:  # light
            on_image += [
                # Ha et al.: RandomBrightness(limit=0.2, p=0.75)
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=0.75),
                # Ha et al.: RandomContrast(limit=0.2, p=0.75)
                transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=0.75),
            ]
        if level >= 4:  # colour
            on_image += [
                # Ha et al.: HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                #                               val_shift_limit=10, p=0.5)
                # Their limits are on OpenCV's scale (hue 0-180, the others 0-255):
                # 10/180 = 0.056 of the hue circle, 20/255 = 0.08, 10/255 = 0.04.
                # torchvision scales saturation and value instead of shifting them.
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.04, saturation=0.08, hue=0.056)], p=0.5),
            ]
        if level >= 5:  # cutout
            on_tensor += [
                # Ha et al.: Cutout(max_h_size=0.375 * size, max_w_size=0.375 * size,
                #                   num_holes=1, p=0.7)
                # One black square with side 0.375 of the image = 14% of its area.
                transforms.RandomErasing(p=0.7, scale=(0.14, 0.14), ratio=(1.0, 1.0), value=0),
            ]
    # panderm needs 224 px from our 256 px images, so we crop the centre, as PanDerm does
    if not on_image:
        on_image = [transforms.CenterCrop(size)]
    return transforms.Compose(on_image
                              + [transforms.ToTensor()]
                              + on_tensor
                              + [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# See also lab 3 (ImageDatasetFromCSV).
class ISICDataset(Dataset):
    """Adapted from `MelanomaDataset` in Ha et al. (MIT):
      https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
    Changes: PIL and torchvision instead of cv2 and albumentations, float target.
    """

    def __init__(self, csv, transform=None, meta_cols=None):
        self.csv = csv.reset_index(drop=True)
        self.transform = transform
        # Metadata step: ((image, metadata), target) instead of (image, target).
        self.meta = None if meta_cols is None else self.csv[list(meta_cols)].to_numpy(np.float32)

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = Image.open(row.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        target = torch.tensor(row.target).float()
        if self.meta is not None:
            return (image, torch.from_numpy(self.meta[index])), target
        return image, target


def sampling_weights(target, mal_frac):
    """Weights so that a share `mal_frac` of the images drawn is malignant. Adapted from ptrblck (see make_loader)."""
    target = np.asarray(target).astype(int)
    # images per class, then a weight per class, then a weight per image
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = np.array([1. - mal_frac, mal_frac]) / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    return torch.from_numpy(samples_weight).double()


# E2 and E5 get the sampler, everything else just shuffles
def make_loader(df, batch_size=64, train=False, augment=False,
                balanced=False, workers=2, aug="base", mal_frac=0.5, size=256, meta_cols=None):
    """balanced=True is E2: we draw more malignant images."""
    ds = ISICDataset(df, make_transform(train, augment, aug, size), meta_cols)
    if train and balanced:
        # Adapted from ptrblck's answer in the PyTorch forum thread
        # "How to handle imbalanced classes" (post 2):
        #   https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        # Library: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        samples_weight = sampling_weights(df.target.values, mal_frac)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=workers, pin_memory=True)
    # own random generator, so evaluating does not change the training
    # https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader
    # batch norm fails on a last batch of 1 image
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True,
                      drop_last=train and meta_cols is not None,
                      generator=None if train else torch.Generator().manual_seed(0))
