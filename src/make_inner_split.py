"""Writes splits/inner_split.csv once. Kept only as provenance — do not rerun.

For each held-out fold (0-4), the remaining four folds are split 80/20 into
train / inner_val, grouped by patient and stratified by target. The file lists
the inner_val images per held-out fold; every other image of the training folds
is train. data.inner_split reads this file, so the inner validation images no
longer depend on the seed or on the scikit-learn version.

Why it was frozen (17 Oct 2026): StratifiedGroupKFold(shuffle=True) gives a
different split in scikit-learn 1.8 than in 1.6/1.7 for the same random_state.
All runs up to now used the 1.6/1.7 split with random_state=0 (fold 0: 4,355
images, 79 malignant). This script was run with scikit-learn 1.7.2 and checked
against the inner_val images recorded in the existing runs' preds.csv
(folds 0 and 1, seed 0): identical.

Library: sklearn.model_selection.StratifiedGroupKFold
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
Why patients must not be split (ISIC 2020): C. Deotte, "Triple Stratified KFold",
  https://www.kaggle.com/code/cdeotte/triple-stratified-kfold-with-tfrecords

Usage (from the repo root):  python src/make_inner_split.py
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

VAL_FRAC = 0.2
SEED = 0

root = Path(__file__).resolve().parents[1]
dev = pd.read_csv(root / "splits" / "dev_split.csv")

parts = []
for test_fold in range(5):
    trn = dev[dev.fold != test_fold].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=round(1 / VAL_FRAC), shuffle=True, random_state=SEED)
    _, val_idx = next(sgkf.split(trn, trn.target, groups=trn.patient_id))
    val = trn.iloc[val_idx][["image_name", "patient_id", "target"]].copy()
    val.insert(0, "test_fold", test_fold)
    parts.append(val)

out = pd.concat(parts).sort_values(["test_fold", "image_name"])
out.to_csv(root / "splits" / "inner_split.csv", index=False)
print(out.groupby("test_fold").target.agg(images="size", malignant="sum"))
