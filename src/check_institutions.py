"""Which institution each ISIC 2020 image comes from, and how many of our images
and melanomas each institution has in the dev and test splits.

Why (18 Oct 2026): PanDerm (Yan et al. 2025) was pretrained on the in-house
Australian cohorts "MYM" (Mind Your Moles) and "HOP" (high-risk sequential
dermoscopy). ISIC 2020's University of Queensland images come from the same
kind of studies (Rotemberg et al. 2021, Sci Data 8:34). If the Queensland share
is small, PanDerm can be evaluated on non-Queensland images only.

Data: the ISIC Archive API, collection 70 = "Challenge 2020: Training"
(33,126 images). Each image record has an `attribution` field.
  https://api.isic-archive.com/api/docs/v2

Usage (Colab, from the repo root):  python src/check_institutions.py
Writes splits/isic2020_attribution.csv (image_name, patient_id, attribution)
and prints the counts. Needs internet; takes a few minutes (~330 requests).
"""
from pathlib import Path
import time

import pandas as pd
import requests

URL = "https://api.isic-archive.com/api/v2/images/search/"
root = Path(__file__).resolve().parents[1]
out = root / "splits" / "isic2020_attribution.csv"

if out.exists():
    attr = pd.read_csv(out)
    print(f"Read {len(attr)} rows from {out.name}")
else:
    rows, url = [], f"{URL}?collections=70&limit=100"
    while url:
        for attempt in range(5):
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"retry {attempt + 1}: {e}")
                time.sleep(5)
        else:
            raise RuntimeError("ISIC API not reachable")
        page = r.json()
        for im in page["results"]:
            rows.append({
                "image_name": im["isic_id"],
                "patient_id": im.get("metadata", {}).get("clinical", {}).get("patient_id"),
                "attribution": im.get("attribution"),
            })
        url = page.get("next")
        if len(rows) % 5000 < 100:
            print(f"{len(rows)} / {page.get('count')}")
    attr = pd.DataFrame(rows)
    attr.to_csv(out, index=False)
    print(f"Wrote {len(attr)} rows to {out.name}")

dev = pd.read_csv(root / "splits" / "dev_split.csv")
test = pd.read_csv(root / "splits" / "test_split.csv")
dev["split"], test["split"] = "dev", "test"
df = pd.concat([dev, test])[["image_name", "target", "split", "fold"]]
df = df.merge(attr[["image_name", "attribution"]], on="image_name", how="left")
print(f"\nImages without attribution: {df.attribution.isna().sum()}")

t = (df.groupby(["attribution", "split"])
       .agg(images=("target", "size"), melanomas=("target", "sum"))
       .unstack("split", fill_value=0))
t.columns = [f"{s}_{c}" for c, s in t.columns]
for s in ["dev", "test"]:
    t[f"{s}_img_%"] = (100 * t[f"{s}_images"] / t[f"{s}_images"].sum()).round(1)
    t[f"{s}_mel_%"] = (100 * t[f"{s}_melanomas"] / t[f"{s}_melanomas"].sum()).round(1)
cols = ["dev_images", "dev_img_%", "dev_melanomas", "dev_mel_%",
        "test_images", "test_img_%", "test_melanomas", "test_mel_%"]
pd.set_option("display.width", 200)
print("\nPer institution:\n", t[cols].sort_values("dev_images", ascending=False))

q = df.attribution.fillna("").str.contains("Queensland", case=False)
print("\nQueensland melanomas per held-out fold:",
      df[q & (df.split == "dev")].groupby("fold").target.sum().astype(int).to_dict())
