"""This is how we made splits/inner_split.csv. We ran it once.

For each held-out fold (0-4), we split the other four folds 80/20 into train and inner
validation, grouped by patient and stratified by target. The file lists the inner
validation images for each held-out fold. All other images in the training folds are train.
data.inner_split reads this file, so the images do not depend on the seed or the
scikit-learn version.

Why we saved it in a file: StratifiedGroupKFold(shuffle=True) gives a different split in
scikit-learn 1.8 than in 1.6/1.7 with the same random_state. We ran this script with
scikit-learn 1.7.2 and checked that it gives the same inner validation images as the runs
we had already done (folds 0 and 1, seed 0).

Library: sklearn.model_selection.StratifiedGroupKFold
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
Why patients must not be split (ISIC 2020): C. Deotte, "Triple Stratified KFold",
  https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords

Do not run it again, the split must stay the same.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

VAL_FRAC = 0.2
SEED = 0

root = Path(__file__).resolve().parents[1]
dev = pd.read_csv(root / "splits" / "dev_split.csv")

# 80/20 split for each held-out fold, patients stay together
parts = []
for test_fold in range(5):
    train_folds = dev[dev.fold != test_fold].reset_index(drop=True)

    splitter = StratifiedGroupKFold(n_splits=round(1 / VAL_FRAC), shuffle=True, random_state=SEED)
    _, val_idx = next(splitter.split(train_folds, train_folds.target, groups=train_folds.patient_id))

    inner_val = train_folds.iloc[val_idx][["image_name", "patient_id", "target"]].copy()
    inner_val.insert(0, "test_fold", test_fold)
    parts.append(inner_val)

out = pd.concat(parts).sort_values(["test_fold", "image_name"])
out.to_csv(root / "splits" / "inner_split.csv", index=False)
print(out.groupby("test_fold").target.agg(images="size", malignant="sum"))
