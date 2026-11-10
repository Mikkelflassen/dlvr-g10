"""Step `panderm_ft_isic2019_meta`: age, sex and body site next to the image.

Adapted from Ha et al. (2020, MIT): `get_meta_data` in dataset.py and `Effnet_Melanoma` in models.py.
  https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
  https://arxiv.org/abs/2010.05351
We left out `n_images` and `image_size`, because they show which dataset an image comes from.
For the same reason, ISIC 2019's torso sites are all called "torso".
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# The ISIC 2019 body site mapping was written with AI (Claude).
SITES = ["head/neck", "upper extremity", "lower extremity", "torso", "palms/soles", "oral/genital"]
SITE_MAP = {"anterior torso": "torso", "posterior torso": "torso", "lateral torso": "torso"}
META_COLS = (["meta_sex", "meta_age"] + [f"meta_site_{i}" for i in range(len(SITES))]
             + ["meta_site_missing"])


def add_meta(df, name=""):
    """Adds the metadata columns to a table."""
    out = df.copy()
    out["meta_sex"] = out["sex"].map({"male": 1.0, "female": 0.0}).fillna(-1.0)
    out["meta_age"] = (out["age_approx"] / 90).fillna(0.0)
    site = out["anatom_site_general_challenge"].replace(SITE_MAP)
    unknown = site.notna() & ~site.isin(SITES)
    if unknown.any():
        print(f"{name}: {int(unknown.sum())} images with an unknown site counted as missing:",
              site[unknown].value_counts().to_dict())
        site = site.where(~unknown)
    for i, site_name in enumerate(SITES):           # one column per body site
        out[f"meta_site_{i}"] = (site == site_name).astype(float)
    out["meta_site_missing"] = site.isna().astype(float)
    out[META_COLS] = out[META_COLS].astype(np.float32)
    return out


# the isic 2019 patient info is in its own csv
def meta_2019(root2019):
    """Sex, age and body site of the ISIC 2019 images."""
    df = pd.read_csv(f"{root2019}/train.csv")
    return df[["image_name", "sex", "age_approx", "anatom_site_general_challenge"]]


class WithMeta(nn.Module):
    """Image features + metadata give one output. From Ha et al.'s Effnet_Melanoma, without their five dropout heads."""

    def __init__(self, backbone, n_features, n_meta=len(META_COLS), dims=(512, 128)):
        super().__init__()
        self.backbone = backbone
        self.meta = nn.Sequential(
            nn.Linear(n_meta, dims[0]), nn.BatchNorm1d(dims[0]), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(dims[0], dims[1]), nn.BatchNorm1d(dims[1]), nn.SiLU())
        self.fc = nn.Linear(n_features + dims[1], 1)

    @property
    def blocks(self):          # for panderm.layer_decay_groups
        return self.backbone.blocks

    def forward(self, image, meta):
        features = torch.cat([self.backbone(image), self.meta(meta)], dim=1)
        return self.fc(features)
