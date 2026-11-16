"""Step `isic2019`: ISIC 2019 images used as extra training data.

Our rules:
- They are for training only. Validation, the held-out folds and the test set remain ISIC 2020.
- Melanoma is 1 and all other diagnoses are 0.
- Before anything else, ISIC 2019 images that also exist in ISIC 2020 are taken out.

Data: C. Deotte's resized ISIC 2019: https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-256x256
The reading code follows `get_df` by Ha et al. (dataset.py, MIT):
  https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
ISIC 2019: Tschandl et al. 2018, Codella et al. 2018, Combalia et al. 2019.

Duplicate check: every image gets a pHash, and two images are duplicates when at most T bits differ.
T is set from the official ISIC 2020 list of 425 duplicates, with 6 bits as the minimum.
We went for the loose side, since the only cost is a few training images.
  https://github.com/JohannesBuchner/imagehash
  https://challenge2020.isic-archive.com/
"""
from pathlib import Path
from multiprocessing import Pool
import json

import numpy as np
import pandas as pd
from PIL import Image

KAGGLE_SLUG = "cdeotte/jpeg-isic2019-256x256"
MIN_T = 6
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def load_meta(root2019):
    """The ISIC 2019 rows, read as in Ha et al.'s get_df and given our target."""
    root2019 = Path(root2019)
    df = pd.read_csv(root2019 / "train.csv")
    df = df[df["tfrecord"] >= 0].reset_index(drop=True)
    df["target"] = (df["diagnosis"] == "MEL").astype(int)
    df["path"] = df.image_name.map(lambda n: str(root2019 / "train" / f"{n}.jpg"))
    return df


def _phash(path):
    import imagehash
    bits = imagehash.phash(Image.open(path).convert("RGB")).hash.flatten()
    return np.packbits(bits).view(">u8")[0]


# all cpus are used because hashing is slow
def phash_all(paths, workers=2):
    with Pool(workers) as pool:
        return np.array(pool.map(_phash, list(paths), chunksize=256), dtype=np.uint64)


# _POP, _hamming and nearest were written with AI (Claude).
def _hamming(a, b):
    """How many bits differ, for every pair of hashes from a and b."""
    different_bits = np.bitwise_xor(a[:, None], b[None, :])
    # _POP gives the number of 1-bits per byte, and each hash is 8 bytes
    return _POP[different_bits.view(np.uint8)].reshape(len(a), len(b), 8).sum(axis=2, dtype=np.uint8)


def _distance(a, b):
    """The Hamming distance from a[i] to b[i], for each i."""
    return _POP[np.bitwise_xor(a, b).view(np.uint8)].reshape(-1, 8).sum(axis=1)


def nearest(a, b, same_set=False, chunk=400):
    """For every hash in a, the closest hash in b. Works in chunks to stay within memory."""
    dist = np.empty(len(a), dtype=np.uint8)
    idx = np.empty(len(a), dtype=np.int64)
    for start in range(0, len(a), chunk):
        end = start + chunk
        d = _hamming(a[start:end], b)
        rows = np.arange(d.shape[0])
        if same_set:                     # an image cannot be its own nearest neighbour
            d[rows, np.arange(start, start + d.shape[0])] = 255
        idx[start:end] = d.argmin(axis=1)
        dist[start:end] = d[rows, idx[start:end]]
    return dist, idx


# Written with AI (Claude). Idea from lab 3 (find_train_val_duplicates); we use pHash instead of ResNet features.
def build(splits_dir, root2020, root2019, out_figure=None, workers=2):
    """Run once. Saves the allowed and the removed ISIC 2019 images, and near-identical pairs in our splits."""
    splits_dir, root2020 = Path(splits_dir), Path(root2020)
    ours = _our_images(splits_dir, root2020)
    ext = load_meta(root2019)
    summary = {"isic2019_rows": int(len(ext)), "isic2019_melanomas": int(ext.target.sum())}

    # 1. Choose T from the known duplicate pairs of ISIC 2020 (we have both copies on disk).
    T = _threshold(splits_dir, root2020, workers, summary)

    # 2. Compute the hashes of our ISIC 2020 images and of ISIC 2019.
    print(f"hashing {len(ours)} ISIC 2020 and {len(ext)} ISIC 2019 images ...")
    h2020 = phash_all(ours.path, workers)
    h2019 = phash_all(ext.path, workers)

    # How often are two *unrelated* ISIC 2020 images within T of each other (false alarms)?
    _random_pairs(ours, h2020, T, summary)

    # 3. ISIC 2019 against ISIC 2020: the image name is the same, or the hash is within T.
    dup = _isic2019_duplicates(ext, ours, h2019, h2020, T, splits_dir, summary)

    # 4. Within ISIC 2020: near-identical images that sit on different sides of a split.
    _cross_split_pairs(ours, h2020, T, splits_dir, summary)

    if out_figure:
        _figure(dup.sort_values("distance", ascending=False), ours, out_figure, T)
    (splits_dir / "isic2019_check.json").write_text(json.dumps(summary, indent=2))
    return summary


# every ISIC 2020 image we use (dev and test) with its place: fold0..fold4 or test
def _our_images(splits_dir, root2020):
    dev = pd.read_csv(splits_dir / "dev_split.csv")
    test = pd.read_csv(splits_dir / "test_split.csv")
    dev["where"] = "fold" + dev.fold.astype(str)
    test["where"] = "test"
    ours = pd.concat([dev, test], ignore_index=True)[["image_name", "patient_id", "target", "where"]]
    ours["path"] = ours.image_name.map(lambda n: str(root2020 / "train" / f"{n}.jpg"))
    return ours


# T is set to cover 99% of the known duplicate pairs, with MIN_T bits as the minimum
def _threshold(splits_dir, root2020, workers, summary):
    official = pd.read_csv(splits_dir / "isic2020_official_duplicates.csv")
    print("hashing the 425 official ISIC 2020 duplicate pairs ...")
    first = phash_all([root2020 / "train" / f"{n}.jpg" for n in official.image_name_1], workers)
    second = phash_all([root2020 / "train" / f"{n}.jpg" for n in official.image_name_2], workers)
    known = _distance(first, second)
    p99 = int(np.percentile(known, 99))
    T = max(p99, MIN_T)
    summary["known_pair_distance"] = {"median": float(np.median(known)), "p99": p99, "max": int(known.max())}
    print(f"known duplicates: median distance {np.median(known):.0f}, 99% within {T}, max {known.max()}")
    return T


# 200,000 random image pairs, each from two different patients
def _random_pairs(ours, h2020, T, summary):
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(h2020), 200_000), rng.integers(0, len(h2020), 200_000)
    keep = ours.patient_id.values[i] != ours.patient_id.values[j]
    random_distance = _distance(h2020[i[keep]], h2020[j[keep]])
    summary["random_pairs_within_T"] = float((random_distance <= T).mean())
    print(f"random pairs of different patients within {T}: {100 * (random_distance <= T).mean():.3f}%")


# writes the removed and the kept ISIC 2019 images to files, and returns the removed ones
def _isic2019_duplicates(ext, ours, h2019, h2020, T, splits_dir, summary):
    distance, nearest_row = nearest(h2019, h2020)
    ext["nearest_2020"] = ours.image_name.values[nearest_row]
    ext["nearest_2020_where"] = ours["where"].values[nearest_row]
    ext["distance"] = distance
    ext["same_name"] = ext.image_name.isin(ours.image_name)

    hist = np.bincount(distance, minlength=33)[:33]
    summary["isic2019_nearest_distance_histogram_0_32"] = hist.tolist()
    print("ISIC 2019 -> nearest ISIC 2020, images per distance 0..16:", hist[:17].tolist())

    dup = ext[(ext.distance <= T) | ext.same_name]
    kept = ext.drop(dup.index)
    summary.update({"threshold_T": T, "isic2019_removed": int(len(dup)),
                    "isic2019_removed_melanomas": int(dup.target.sum()),
                    "isic2019_removed_matching_test": int((dup.nearest_2020_where == "test").sum()),
                    "isic2019_kept": int(len(kept)), "isic2019_kept_melanomas": int(kept.target.sum())})
    cols = ["image_name", "diagnosis", "target", "nearest_2020", "nearest_2020_where", "distance", "same_name"]
    dup[cols].to_csv(splits_dir / "isic2019_duplicates.csv", index=False)
    kept[["image_name", "diagnosis", "target"]].to_csv(splits_dir / "isic2019_train.csv", index=False)
    return dup


# writes the near-identical ISIC 2020 pairs that are split over two folds or over dev and test
def _cross_split_pairs(ours, h2020, T, splits_dir, summary):
    distance, nearest_row = nearest(h2020, h2020, same_set=True)
    pairs = ours.assign(other=ours.image_name.values[nearest_row],
                        other_where=ours["where"].values[nearest_row],
                        other_patient=ours.patient_id.values[nearest_row],
                        distance=distance)
    cross = pairs[(pairs.distance <= T) & (pairs["where"] != pairs.other_where)]
    cross = cross[cross.image_name < cross.other]           # count each pair only once
    cross[["image_name", "where", "other", "other_where", "distance", "target"]].to_csv(
        splits_dir / "isic2020_near_duplicates.csv", index=False)
    touching_test = (cross["where"] == "test") | (cross.other_where == "test")
    summary.update({"isic2020_cross_split_pairs": int(len(cross)),
                    "isic2020_cross_split_pairs_touching_test": int(touching_test.sum())})


# shows the pairs side by side, so that we can check them by eye
def _figure(dup, ours, path, T, n=8):
    """The removed pairs that are nearest to the threshold, shown next to each other."""
    import matplotlib.pyplot as plt
    rows = dup.head(n)
    if rows.empty:
        return
    where = dict(zip(ours.image_name, ours.path))
    fig, axes = plt.subplots(2, len(rows), figsize=(2 * len(rows), 4.4))
    axes = np.atleast_2d(axes).reshape(2, -1)

    for column, (_, row) in enumerate(rows.iterrows()):
        axes[0, column].imshow(Image.open(row.path))
        axes[0, column].set_title(f"2019 {row.image_name[5:]}", fontsize=7)

        axes[1, column].imshow(Image.open(where[row.nearest_2020]))
        axes[1, column].set_title(f"2020 d={row.distance}", fontsize=7)

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(f"Removed ISIC 2019 images closest to the threshold (T = {T} bits)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pairs_figure(splits_dir, root2020, root2019, path, per_distance=6):
    """A few example pairs for every hash distance."""
    import matplotlib.pyplot as plt
    splits_dir, root2020, root2019 = Path(splits_dir), Path(root2020), Path(root2019)
    dup = pd.read_csv(splits_dir / "isic2019_duplicates.csv")
    near = pd.read_csv(splits_dir / "isic2020_near_duplicates.csv")
    rows = [
        (
            f"2019 vs 2020, d={distance}",
            dup[dup.distance == distance].head(per_distance),
            lambda row: (
                root2019 / "train" / f"{row.image_name}.jpg",
                root2020 / "train" / f"{row.nearest_2020}.jpg",
            ),
        )
        for distance in (0, 2, 4, 6)
    ]
    rows.append((
        "2020 vs 2020 across splits, d=0",
        near[near.distance == 0].head(per_distance),
        lambda row: (
            root2020 / "train" / f"{row.image_name}.jpg",
            root2020 / "train" / f"{row.other}.jpg",
        ),
    ))

    fig, axes = plt.subplots(
        len(rows),
        2 * per_distance,
        figsize=(2 * per_distance * 1.1, len(rows) * 1.35),
    )

    for row_nr, (title, df, paths) in enumerate(rows):
        axes[row_nr, 0].set_title(title, fontsize=8, loc="left")

        for column, (_, row) in enumerate(df.iterrows()):
            a, b = paths(row)
            axes[row_nr, 2 * column].imshow(Image.open(a))
            axes[row_nr, 2 * column + 1].imshow(Image.open(b))

    for a in axes.flat:
        a.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# just the isic 2019 images that got through the duplicate check
def load_extra(splits_dir, root2019):
    """The ISIC 2019 images for training, after the duplicate check."""
    kept_csv = Path(splits_dir) / "isic2019_train.csv"
    if not kept_csv.exists():
        raise FileNotFoundError("splits/isic2019_train.csv is missing: run notebooks/"
                                "G10_isic2019_check.ipynb first (duplicate check)")
    kept = pd.read_csv(kept_csv)
    kept["path"] = kept.image_name.map(lambda n: str(Path(root2019) / "train" / f"{n}.jpg"))
    return kept[["image_name", "target", "path"]]
