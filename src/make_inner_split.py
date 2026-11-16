"""This is how splits/inner_split.csv was made. It should not be run again.

For every held-out fold, we split the four other folds 80/20, grouped by patient.
The result is saved, since scikit-learn 1.8 and 1.7 give different splits for the same seed.

Library: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
Why patients must not be split: C. Deotte, https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

VAL_FRAC = 0.2
SEED = 0

root = Path(__file__).resolve().parents[1]
dev = pd.read_csv(root / "splits" / "dev_split.csv")

# an 80/20 split per held-out fold, where a patient is never divided
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
