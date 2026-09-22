"""Dataset and loaders. Reads the frozen split CSVs — never recomputes a split."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

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


def inner_split(dev, test_fold, val_frac=0.15, seed=42):
    """Training folds -> (train, inner_val), grouped by patient.

    inner_val is used for epoch selection, threshold tuning and arm settings.
    The held-out fold is never touched for any choice.
    """
    trn = dev[dev.fold != test_fold]
    pats = trn.patient_id.unique()
    rng = np.random.RandomState(seed)
    val_pats = set(rng.choice(pats, int(len(pats) * val_frac), replace=False))
    return (trn[~trn.patient_id.isin(val_pats)].reset_index(drop=True),
            trn[trn.patient_id.isin(val_pats)].reset_index(drop=True))


def make_transform(train, augment=False):
    if train and augment:
        t = [transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip()]
    else:
        t = []  # images are already 256x256
    return transforms.Compose(t + [transforms.ToTensor(),
                                   transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ISICDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df.path.tolist()
        self.labels = df.target.astype(np.float32).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def make_loader(df, batch_size=64, train=False, augment=False,
                balanced=False, workers=2):
    """balanced=True is E2: oversample malignant to a 50/50 effective prior."""
    ds = ISICDataset(df, make_transform(train, augment))
    if train and balanced:
        counts = df.target.value_counts()
        w = df.target.map(lambda y: 1.0 / counts[y]).values
        sampler = WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=workers, pin_memory=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=workers, pin_memory=True)
