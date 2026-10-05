"""Dataset and loaders. Reads the frozen split CSVs — never recomputes a split.

Data: ISIC 2020 (Rotemberg et al., 2021, https://doi.org/10.1038/s41597-021-00815-z)
in Chris Deotte's 256x256 JPEG version,
https://www.kaggle.com/datasets/cdeotte/jpeg-melanoma-256x256
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
from PIL import Image

# ImageNet statistics, as in the PyTorch transfer learning tutorial (see make_transform).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_splits(splits_dir, data_root):
    """dev (with fold 0-4) and test, with an absolute `path` column."""
    splits_dir, data_root = Path(splits_dir), Path(data_root)
    out = []
    for name in ("dev_split.csv", "test_split.csv"):
        df = pd.read_csv(splits_dir / name)
        df["path"] = df.image_name.map(lambda n: str(data_root / "train" / f"{n}.jpg"))
        out.append(df)
    return out


def inner_split(dev, test_fold, val_frac=0.2, seed=42):
    """Training folds -> (train, inner_val), grouped by patient.

    inner_val is used for epoch selection, threshold tuning and arm settings.
    The held-out fold is never touched for any choice.

    Library: sklearn.model_selection.StratifiedGroupKFold — every patient stays on
    one side and the malignant rate is kept equal on both. One of round(1/val_frac)
    parts becomes inner_val (5 parts, so 20%, for the default).
      https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
    Why patients must not be split (ISIC 2020): C. Deotte, "Triple Stratified KFold",
      https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords
    Why selection needs its own split: Cawley & Talbot (2010),
      https://jmlr.org/papers/v11/cawley10a.html
    """
    trn = dev[dev.fold != test_fold].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=round(1 / val_frac), shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(trn, trn.target, groups=trn.patient_id))
    return (trn.iloc[train_idx].reset_index(drop=True),
            trn.iloc[val_idx].reset_index(drop=True))


def make_transform(train, augment=False):
    """EDIT THE TRAINING AUGMENTATION HERE — this is the only place it lives.

    Applies to training images only; validation and test are never augmented.
    Colour is deliberately left alone: colour variegation is diagnostic for
    melanoma, so a hue or saturation shift can contradict the label.

    Adapted from `data_transforms` in the PyTorch transfer learning tutorial
    (BSD-3-Clause): RandomResizedCrop + RandomHorizontalFlip + ToTensor + Normalize.
      https://github.com/pytorch/tutorials/blob/main/beginner_source/transfer_learning_tutorial.py
    The vertical flip is from the ISIC 2020 winning solution (Ha et al., 2020, MIT):
      https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
    """
    if train and augment:
        t = [
            transforms.RandomResizedCrop(256, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    else:
        t = []  # images are already 256x256
    return transforms.Compose(t + [transforms.ToTensor(),
                                   transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ISICDataset(Dataset):
    """Adapted from `MelanomaDataset` in the ISIC 2020 winning solution (Ha et al., MIT):
      https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
    Changes: PIL + torchvision transforms instead of cv2 + albumentations, no
    metadata branch, float target for a single-logit BCE output.
    """

    def __init__(self, csv, transform=None):
        self.csv = csv.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = Image.open(row.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(row.target).float()


def make_loader(df, batch_size=64, train=False, augment=False,
                balanced=False, workers=2):
    """balanced=True is E2: oversample malignant to a 50/50 effective prior."""
    ds = ISICDataset(df, make_transform(train, augment))
    if train and balanced:
        # Adapted from ptrblck's answer in the PyTorch forum thread
        # "How to handle imbalanced classes" (post 2):
        #   https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264
        # Library: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
        target = df.target.values.astype(int)
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=workers, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True)
