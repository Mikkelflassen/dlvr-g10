# dlvr-g10

Group 10, Deep Learning for Visual Recognition, Aarhus University, autumn 2026.

Benign vs malignant skin-lesion classification on ISIC 2020, comparing four
treatments for a ~56:1 class imbalance at two fine-tuning depths.

| Arm | Treatment |
|-----|-----------|
| E0  | Plain BCE (baseline) |
| E0b | E0 re-thresholded — no training, the control |
| E1  | BCE with `pos_weight` |
| E2  | Balanced sampling (`WeightedRandomSampler`) |
| E3  | Focal loss |

## Rules

1. **`splits/dev_split.csv` and `splits/test_split.csv` are canonical.** Never
   regenerate them. `make_splits.py` is kept as provenance, not to be rerun.
2. **All training goes through `notebooks/G10_run.ipynb`.** No training loop in
   any other notebook — a second copy is a second explanation for any gap
   between arms.
3. **All metrics come from `eval.py`**, computed from saved `preds.csv`.
4. **`test_split.csv` stays closed until December.** Model selection,
   thresholds and arm settings use the inner validation split only.

## Data

Images are not in this repo. Each session:

```python
!kaggle datasets download -d cdeotte/jpeg-melanoma-256x256 -p /content -q
!unzip -q /content/jpeg-melanoma-256x256.zip -d /content/isic
```

792 MB, 256×256 JPEGs, centre-cropped then resized by the dataset author.
Not the ~108 GB competition archive. Needs `KAGGLE_USERNAME` / `KAGGLE_KEY`
in Colab Secrets.

## One run

```python
import json
from src.train import run

cfg = json.load(open("configs/arms.json"))["E3_full"]
run(cfg | {"fold": 0, "seed": 0}, "splits", "/content/isic", "results/runs")
```

Writes `results/runs/E3_full_f0_s0/` containing `config.json` (settings + git
hash), `history.csv` (per-epoch loss and AUROC) and `preds.csv` (per-image
probabilities). Every reported metric is derived from `preds.csv`, so the
analysis can change without retraining.

Checkpoints are not committed.

## The grid

4 arms × 2 depths × 5 folds = 40 runs. Split **by fold**, not by arm — each
person runs all eight combinations on their own folds, so machine differences
hit every arm equally instead of flattering one.

## Layout

```
src/        data.py models.py losses.py train.py eval.py
configs/    arms.json
splits/     dev_split.csv test_split.csv
results/    runs/<arm>_<depth>_f<fold>_s<seed>/ , results.csv
notebooks/  G10_run.ipynb G10_analysis.ipynb G10_data.ipynb
```
