"""Finds the institution of each ISIC 2020 image, and counts images and melanomas per institution.

PanDerm (Yan et al. 2025) was pretrained on Australian cohorts. The Queensland images in
ISIC 2020 (Rotemberg et al. 2021, Sci Data 8:34) may come from the same studies.

Data: ISIC Archive API, collection 70: https://api.isic-archive.com/api/docs/v2
Saves splits/isic2020_attribution.csv.
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
    # the api gives 100 images at a time, so we go page by page
    rows, url = [], f"{URL}?collections=70&limit=100"
    while url:
        # the api sometimes fails, so we try 5 times
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                break
            except requests.RequestException as error:
                print(f"retry {attempt + 1}: {error}")
                time.sleep(5)
        else:
            raise RuntimeError("ISIC API not reachable")
        page = response.json()
        for image in page["results"]:
            rows.append({
                "image_name": image["isic_id"],
                "patient_id": image.get("metadata", {}).get("clinical", {}).get("patient_id"),
                "attribution": image.get("attribution"),
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

# images and melanomas per hospital, in dev and test
table = (
    df.groupby(["attribution", "split"])
    .agg(images=("target", "size"), melanomas=("target", "sum"))
    .unstack("split", fill_value=0)
)
table.columns = [f"{split_name}_{column}" for column, split_name in table.columns]

for split_name in ["dev", "test"]:
    table[f"{split_name}_img_%"] = (
        100 * table[f"{split_name}_images"] / table[f"{split_name}_images"].sum()
    ).round(1)
    table[f"{split_name}_mel_%"] = (
        100 * table[f"{split_name}_melanomas"] / table[f"{split_name}_melanomas"].sum()
    ).round(1)

cols = [
    "dev_images", "dev_img_%", "dev_melanomas", "dev_mel_%",
    "test_images", "test_img_%", "test_melanomas", "test_mel_%",
]
pd.set_option("display.width", 200)
print("\nPer institution:\n", table[cols].sort_values("dev_images", ascending=False))

queensland = df.attribution.fillna("").str.contains("Queensland", case=False)
print(
    "\nQueensland melanomas per held-out fold:",
    df[queensland & (df.split == "dev")].groupby("fold").target.sum().astype(int).to_dict(),
)
