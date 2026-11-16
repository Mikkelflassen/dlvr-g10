"""The dataset and the data loaders. Splits come from files.

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
# The ImageNet statistics, taken from the PyTorch transfer learning tutorial (see make_transform).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# loads our fixed splits and adds the location of every image
def load_splits(splits_dir, data_root):
    """Loads the dev set and the test set, with the path of every image added."""
    splits_dir, data_root = Path(splits_dir), Path(data_root)
    out = []
    for name in ("dev_split.csv", "test_split.csv"):
        df = pd.read_csv(splits_dir / name)
        df["path"] = df.image_name.map(lambda n: str(data_root / "train" / f"{n}.jpg"))
        out.append(df)
    return out


# Train, validation and test sets: see lecture 2, section 3.
def inner_split(dev, test_fold, splits_dir):
    """Divides the training folds into train and inner validation (splits/inner_split.csv).

    The best epoch, the threshold and the settings are all chosen on inner validation.
    For why these choices need a split of their own, see Cawley & Talbot (2010),
      https://jmlr.org/papers/v11/cawley10a.html
    """
    train_folds = dev[dev.fold != test_fold].reset_index(drop=True)
    inner = pd.read_csv(Path(splits_dir) / "inner_split.csv")
    val_names = inner.loc[inner.test_fold == test_fold, "image_name"]
    is_val = train_folds.image_name.isin(val_names)
    train = train_folds[~is_val].reset_index(drop=True)
    val = train_folds[is_val].reset_index(drop=True)

    # two checks: no image is missing, and no patient is on both sides
    assert is_val.sum() == len(val_names), "inner_split.csv does not match dev_split.csv"
    assert not set(train.patient_id) & set(val.patient_id), "a patient is in both train and inner_val"
    return train, val


# The augmentation ladder. Every step includes all the steps before it.
AUG_STEPS = ["none", "base", "geo", "light", "colour", "cutout", "randaug"]


def _transpose(img):
    """Mirrors the image along its diagonal."""
    return img.transpose(Image.TRANSPOSE)


# Converted from albumentations to torchvision with AI (Claude).
# See also lecture 6, Code 3.
def make_transform(train, augment=False, aug="base", size=256):
    """Augmentation for training. Images for validation and test are left as they are.

    `aug` chooses a step, and every step builds on the previous one:
      none    nothing
      base    crop + flips
      geo     + transpose, shift / scale / rotate
      light   + brightness, contrast (the one we used)
      colour  + hue, saturation
      cutout  + one erased square. DeVries & Taylor (2017), https://arxiv.org/abs/1708.04552
      randaug base + RandAugment. Cubuk et al. (2020), https://arxiv.org/abs/1909.13719
              https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html

    base follows `data_transforms` in the PyTorch transfer learning tutorial (BSD-3-Clause):
      https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
    For the other steps we took the operations and strengths from `get_transforms` in Ha et al. (2020, MIT):
      https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
      https://arxiv.org/abs/2010.05351
    Their code uses albumentations and ours uses torchvision, so the result is close but not identical.
      https://docs.pytorch.org/vision/stable/transforms.html
    """
    if aug not in AUG_STEPS:
        raise ValueError(f"aug must be one of {AUG_STEPS}, got {aug!r}")
    level = AUG_STEPS.index(aug)
    if aug == "randaug":
        level = 1  # base alone, since RandAugment takes the place of the hand-picked steps

    on_image, on_tensor = [], []   # what runs before, and what runs after, the image is turned into a tensor
    if train and augment and level >= 1:
        # base
        on_image += [
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),   # Ha et al.: VerticalFlip(p=0.5)
        ]
        if aug == "randaug":
            # the torchvision defaults, the same as the paper's setting for ResNet-50 on ImageNet
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
                # These limits are given on the OpenCV scale (0-180 for hue, 0-255 for the others):
                # 10/180 = 0.056 of the hue circle, 20/255 = 0.08, 10/255 = 0.04.
                # In torchvision, saturation and value are scaled, not shifted.
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.04, saturation=0.08, hue=0.056)], p=0.5),
            ]
        if level >= 5:  # cutout
            on_tensor += [
                # Ha et al.: Cutout(max_h_size=0.375 * size, max_w_size=0.375 * size,
                #                   num_holes=1, p=0.7)
                # A single black square whose side is 0.375 of the image, which is 14% of the area.
                transforms.RandomErasing(p=0.7, scale=(0.14, 0.14), ratio=(1.0, 1.0), value=0),
            ]
    # panderm takes 224 px and our images are 256 px, so we cut out the centre like PanDerm does
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
    What we changed: PIL and torchvision in place of cv2 and albumentations, and a float target.
    """

    def __init__(self, csv, transform=None, meta_cols=None):
        self.csv = csv.reset_index(drop=True)
        self.transform = transform
        # Metadata step: gives ((image, metadata), target) where the others give (image, target).
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
    """Weights that make a share `mal_frac` of the drawn images malignant. Adapted from ptrblck (see make_loader)."""
    target = np.asarray(target).astype(int)
    # count the images in each class, turn that into a class weight, then into a weight per image
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = np.array([1. - mal_frac, mal_frac]) / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    return torch.from_numpy(samples_weight).double()


# the sampler is for E2 and E5 only, all other runs simply shuffle
def make_loader(df, batch_size=64, train=False, augment=False,
                balanced=False, workers=2, aug="base", mal_frac=0.5, size=256, meta_cols=None):
    """balanced=True means E2, where malignant images are drawn more often."""
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
    # the loader has its own random generator, so that evaluating leaves the training unchanged
    # https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader
    # batch norm breaks when the last batch has only 1 image
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True,
                      drop_last=train and meta_cols is not None,
                      generator=None if train else torch.Generator().manual_seed(0))
