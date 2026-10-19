"""M12 step `isic2019`: ISIC 2019 images as extra *training* data.

Question for the report: do real extra melanomas help where resampling the same
~400 melanomas (E2) did not?

Rules (decided 18 Oct 2026, before any run):
  - ISIC 2019 is used for training only. Inner validation, the held-out folds and
    the test set stay pure ISIC 2020, so every number stays comparable with the grid.
  - Only melanoma (diagnosis MEL) is 1. Everything else, BCC and SCC included, is 0,
    because the ISIC 2020 target is melanoma vs not melanoma.
  - Every ISIC 2019 image that duplicates an ISIC 2020 image (dev OR test) is removed
    before any training run (`build`, run once from notebooks/G10_isic2019_check.ipynb).

Data: Chris Deotte's resized ISIC 2019 on Kaggle (the external data of the ISIC 2020
winners), same 256 px format as our ISIC 2020 images.
  https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-256x256
Reading it follows Ha et al.'s `get_df` (dataset.py, MIT licence): train.csv,
images in train/<image_name>.jpg, rows with tfrecord < 0 dropped (the dataset
author's own duplicate marking), diagnosis "MEL" = melanoma.
  https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution/blob/master/dataset.py
ISIC 2019 challenge data: Tschandl et al. 2018 (HAM10000), Codella et al. 2018,
Combalia et al. 2019 (BCN20000).

Duplicate check: a perceptual hash (pHash, 64 bits) per image, from the
`imagehash` library; two images whose hashes differ in at most T bits count as
duplicates. pHash survives resizing and JPEG re-compression, which a file checksum
does not. T is calibrated on ISIC 2020's own official duplicate list
(ISIC_2020_Training_Duplicates.csv, 425 pairs, both copies are on disk):
T = the distance that covers 99% of those known pairs, but at least 6 bits: the
ISIC 2019 copy of a photo may come from a differently cropped original, so it can
sit a few bits further away than ISIC 2020's own copies. A too-strict T could miss
real duplicates; a too-loose one only removes a few extra training images, so the
check errs on the loose side. The full distance histogram is saved for the report.
  https://github.com/JohannesBuchner/imagehash
  https://challenge2020.isic-archive.com/  (duplicate list under "Data")
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
    """ISIC 2019 rows as in Ha et al.'s get_df, with our binary target."""
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


def phash_all(paths, workers=2):
    with Pool(workers) as pool:
        return np.array(pool.map(_phash, list(paths), chunksize=256), dtype=np.uint64)


def _hamming(a, b):
    """Bits that differ between every hash in a (n) and every hash in b (m) -> n x m."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POP[x.view(np.uint8)].reshape(len(a), len(b), 8).sum(axis=2, dtype=np.uint8)


def nearest(a, b, same_set=False, chunk=400):
    """For each hash in a: the distance to, and index of, its closest hash in b.
    same_set=True ignores each image's distance to itself (a and b are the same list)."""
    dist = np.empty(len(a), dtype=np.uint8)
    idx = np.empty(len(a), dtype=np.int64)
    for s in range(0, len(a), chunk):
        d = _hamming(a[s:s + chunk], b)
        if same_set:
            d[np.arange(d.shape[0]), np.arange(s, s + d.shape[0])] = 255
        idx[s:s + chunk] = d.argmin(axis=1)
        dist[s:s + chunk] = d[np.arange(d.shape[0]), idx[s:s + chunk]]
    return dist, idx


def build(splits_dir, root2020, root2019, out_figure=None, workers=2):
    """Run once. Writes splits/isic2019_train.csv (the ISIC 2019 images allowed in
    training), splits/isic2019_duplicates.csv (the ones removed, with their ISIC 2020
    match) and splits/isic2020_near_duplicates.csv (pairs inside our own splits that
    cross a fold or the test set), and returns a summary."""
    splits_dir, root2020 = Path(splits_dir), Path(root2020)
    dev = pd.read_csv(splits_dir / "dev_split.csv")
    test = pd.read_csv(splits_dir / "test_split.csv")
    dev["where"] = "fold" + dev.fold.astype(str)
    test["where"] = "test"
    ours = pd.concat([dev, test], ignore_index=True)[["image_name", "patient_id", "target", "where"]]
    ours["path"] = ours.image_name.map(lambda n: str(root2020 / "train" / f"{n}.jpg"))
    ext = load_meta(root2019)
    official = pd.read_csv(splits_dir / "isic2020_official_duplicates.csv")
    summary = {"isic2019_rows": int(len(ext)), "isic2019_melanomas": int(ext.target.sum())}

    # 1. Calibrate T on ISIC 2020's known duplicate pairs (both copies are on disk).
    print("hashing the 425 official ISIC 2020 duplicate pairs ...")
    p1 = phash_all([root2020 / "train" / f"{n}.jpg" for n in official.image_name_1], workers)
    p2 = phash_all([root2020 / "train" / f"{n}.jpg" for n in official.image_name_2], workers)
    known = _POP[np.bitwise_xor(p1, p2).view(np.uint8)].reshape(-1, 8).sum(axis=1)
    p99 = int(np.percentile(known, 99))
    T = max(p99, MIN_T)
    summary["known_pair_distance"] = {"median": float(np.median(known)), "p99": p99, "max": int(known.max())}
    print(f"known duplicates: median distance {np.median(known):.0f}, 99% within {T}, max {known.max()}")

    # 2. Hash our ISIC 2020 images and ISIC 2019.
    print(f"hashing {len(ours)} ISIC 2020 and {len(ext)} ISIC 2019 images ...")
    h20 = phash_all(ours.path, workers)
    h19 = phash_all(ext.path, workers)

    # How often do two *unrelated* ISIC 2020 images fall within T (false alarms)?
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(h20), 200_000), rng.integers(0, len(h20), 200_000)
    keep = ours.patient_id.values[i] != ours.patient_id.values[j]
    rand = _POP[np.bitwise_xor(h20[i[keep]], h20[j[keep]]).view(np.uint8)].reshape(-1, 8).sum(axis=1)
    summary["random_pairs_within_T"] = float((rand <= T).mean())
    print(f"random pairs of different patients within {T}: {100 * (rand <= T).mean():.3f}%")

    # 3. ISIC 2019 vs ISIC 2020: same image name, or hash within T.
    d, k = nearest(h19, h20)
    ext["nearest_2020"] = ours.image_name.values[k]
    ext["nearest_2020_where"] = ours["where"].values[k]
    ext["distance"] = d
    ext["same_name"] = ext.image_name.isin(ours.image_name)
    hist = np.bincount(d, minlength=33)[:33]
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

    # 4. Inside ISIC 2020: near-identical images on different sides of a split.
    d20, k20 = nearest(h20, h20, same_set=True)
    pairs = ours.assign(other=ours.image_name.values[k20], other_where=ours["where"].values[k20],
                        other_patient=ours.patient_id.values[k20], distance=d20)
    cross = pairs[(pairs.distance <= T) & (pairs["where"] != pairs.other_where)]
    cross = cross[cross.image_name < cross.other]           # each pair once
    cross[["image_name", "where", "other", "other_where", "distance", "target"]].to_csv(
        splits_dir / "isic2020_near_duplicates.csv", index=False)
    summary.update({"isic2020_cross_split_pairs": int(len(cross)),
                    "isic2020_cross_split_pairs_touching_test": int(((cross["where"] == "test") |
                                                                     (cross.other_where == "test")).sum())})

    if out_figure:
        _figure(dup.sort_values("distance", ascending=False), ours, out_figure, T)
    (splits_dir / "isic2019_check.json").write_text(json.dumps(summary, indent=2))
    return summary


def _figure(dup, ours, path, T, n=8):
    """The removed pairs closest to the threshold, side by side, to check by eye
    that they really are the same photo."""
    import matplotlib.pyplot as plt
    rows = dup.head(n)
    if rows.empty:
        return
    where = dict(zip(ours.image_name, ours.path))
    fig, ax = plt.subplots(2, len(rows), figsize=(2 * len(rows), 4.4))
    ax = np.atleast_2d(ax).reshape(2, -1)
    for c, (_, r) in enumerate(rows.iterrows()):
        ax[0, c].imshow(Image.open(r.path)); ax[0, c].set_title(f"2019 {r.image_name[5:]}", fontsize=7)
        ax[1, c].imshow(Image.open(where[r.nearest_2020])); ax[1, c].set_title(f"2020 d={r.distance}", fontsize=7)
    for a in ax.flat:
        a.axis("off")
    fig.suptitle(f"Removed ISIC 2019 images closest to the threshold (T = {T} bits)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pairs_figure(splits_dir, root2020, root2019, path, per_distance=6):
    """For the report: example pairs at each hash distance, ISIC 2019 vs its nearest
    ISIC 2020 image (rows 0, 2, 4, 6 bits), plus near-identical ISIC 2020 pairs that
    cross a fold or the test set (last row, 0 bits). Shows by eye from which distance
    on the pairs are different lesions. pHash sets exactly half of its 64 bits, so
    distances are always even."""
    import matplotlib.pyplot as plt
    splits_dir, root2020, root2019 = Path(splits_dir), Path(root2020), Path(root2019)
    dup = pd.read_csv(splits_dir / "isic2019_duplicates.csv")
    near = pd.read_csv(splits_dir / "isic2020_near_duplicates.csv")
    rows = [(f"2019 vs 2020, d={d}", dup[dup.distance == d].head(per_distance),
             lambda r: (root2019 / "train" / f"{r.image_name}.jpg", root2020 / "train" / f"{r.nearest_2020}.jpg"))
            for d in (0, 2, 4, 6)]
    rows.append(("2020 vs 2020 across splits, d=0", near[near.distance == 0].head(per_distance),
                 lambda r: (root2020 / "train" / f"{r.image_name}.jpg", root2020 / "train" / f"{r.other}.jpg")))
    fig, ax = plt.subplots(len(rows), 2 * per_distance, figsize=(2 * per_distance * 1.1, len(rows) * 1.35))
    for i, (title, df, paths) in enumerate(rows):
        ax[i, 0].set_title(title, fontsize=8, loc="left")
        for c, (_, r) in enumerate(df.iterrows()):
            a, b = paths(r)
            ax[i, 2 * c].imshow(Image.open(a)); ax[i, 2 * c + 1].imshow(Image.open(b))
    for a in ax.flat:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def load_extra(splits_dir, root2019):
    """The ISIC 2019 training images that passed the duplicate check."""
    kept_csv = Path(splits_dir) / "isic2019_train.csv"
    if not kept_csv.exists():
        raise FileNotFoundError("splits/isic2019_train.csv is missing: run notebooks/"
                                "G10_isic2019_check.ipynb first (duplicate check)")
    kept = pd.read_csv(kept_csv)
    kept["path"] = kept.image_name.map(lambda n: str(Path(root2019) / "train" / f"{n}.jpg"))
    return kept[["image_name", "target", "path"]]
