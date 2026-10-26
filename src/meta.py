"""M12 step `panderm_ft_isic2019_meta`: patient metadata (age, sex, body site) next to the image.

Adapted from the ISIC 2020 winning solution (Ha et al., 2020, MIT licence):
  - features: `get_meta_data` in dataset.py (sex male=1 / female=0 / missing=-1,
    age / 90 with missing = 0, body site one-hot including a "missing" column)
  - network: `Effnet_Melanoma` in models.py: the metadata go through a small
    network (512 -> 128, batch norm, Swish, dropout 0.3) and are concatenated with
    the image features before the final linear layer.
  https://github.com/haqishen/SIIM-ISIC-Melanoma-Classification-1st-Place-Solution
  https://arxiv.org/abs/2010.05351

Deliberately NOT used from Ha et al.: images per patient (`n_images`) and file size
(`image_size`). Both describe how and where an image was taken, not the lesion,
and ISIC 2019 has no patient ids, so `n_images` would also tell the network which
dataset an image comes from (ISIC 2019 has ~18% melanomas, ISIC 2020 ~1.8%).

Body sites: ISIC 2020 uses six general sites; ISIC 2019 splits the torso into
anterior / posterior / lateral, which are mapped to "torso" so that a site never
identifies the dataset. Any other unknown site counts as missing (printed).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SITES = ["head/neck", "upper extremity", "lower extremity", "torso", "palms/soles", "oral/genital"]
SITE_MAP = {"anterior torso": "torso", "posterior torso": "torso", "lateral torso": "torso"}
META_COLS = (["meta_sex", "meta_age"] + [f"meta_site_{i}" for i in range(len(SITES))]
             + ["meta_site_missing"])


def add_meta(df, name=""):
    """A copy of df with the META_COLS columns (float32), from sex, age_approx and
    anatom_site_general_challenge."""
    out = df.copy()
    out["meta_sex"] = out["sex"].map({"male": 1.0, "female": 0.0}).fillna(-1.0)
    out["meta_age"] = (out["age_approx"] / 90).fillna(0.0)
    site = out["anatom_site_general_challenge"].replace(SITE_MAP)
    unknown = site.notna() & ~site.isin(SITES)
    if unknown.any():
        print(f"{name}: {int(unknown.sum())} images with an unknown site counted as missing:",
              site[unknown].value_counts().to_dict())
        site = site.where(~unknown)
    for i, s in enumerate(SITES):
        out[f"meta_site_{i}"] = (site == s).astype(float)
    out["meta_site_missing"] = site.isna().astype(float)
    out[META_COLS] = out[META_COLS].astype(np.float32)
    return out


def meta_2019(root2019):
    """sex, age_approx and body site of the ISIC 2019 images, from the dataset's train.csv."""
    df = pd.read_csv(f"{root2019}/train.csv")
    return df[["image_name", "sex", "age_approx", "anatom_site_general_challenge"]]


class WithMeta(nn.Module):
    """Image features + metadata -> one logit (Ha et al.'s Effnet_Melanoma, without
    their five averaged dropout heads). `backbone` must return a feature vector."""

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

    def forward(self, x, m):
        return self.fc(torch.cat([self.backbone(x), self.meta(m)], dim=1))
