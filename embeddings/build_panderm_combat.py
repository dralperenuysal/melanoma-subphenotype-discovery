"""Batch-effect harmonization of PanDerm CLS embeddings (melanoma-only, raw and
cropped cohorts), analogous to `build_melanoma_combat.py` (DINOv2 run H) but for
PanDerm (runs K/L). `attribution` (source institution) is the batch variable.

No biological covariate is protected (same caveat as the DINOv2 ComBat run:
diagnosis_3 is exactly what we're trying to discover, so ComBat may remove real
signal that correlates with batch).

Batches with <30 images are grouped into "Other".

Run with `.venv_cluster/bin/python` (has neuroCombat installed).
Usage: python build_panderm_combat.py {raw,cropped}
"""
import os
import sys

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

VARIANT = sys.argv[1]
assert VARIANT in ("raw", "cropped"), "usage: python build_panderm_combat.py {raw,cropped}"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS_DIR = os.path.join(ROOT, "embeddings", "panderm", VARIANT, "cls")
METADATA_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")

OUT_NPY = os.path.join(ROOT, "embeddings", f"panderm_{VARIANT}_cls_combat.npy")
OUT_IDS_CSV = os.path.join(ROOT, "embeddings", f"panderm_{VARIANT}_cls_combat_isic_ids.csv")

MIN_BATCH_SIZE = 30

if __name__ == "__main__":
    isic_ids = sorted(f[:-4] for f in os.listdir(CLS_DIR) if f.endswith(".npy"))
    X = np.stack([np.load(os.path.join(CLS_DIR, f"{i}.npy")) for i in isic_ids]).astype(np.float64)

    meta = pd.read_csv(METADATA_CSV, low_memory=False).set_index("isic_id")
    attribution = meta.loc[isic_ids, "attribution"].fillna("Anonymous")

    counts = attribution.value_counts()
    small_batches = counts[counts < MIN_BATCH_SIZE].index
    batch = attribution.where(~attribution.isin(small_batches), "Other")
    print(f"[{VARIANT}] Batch sizes after grouping (<{MIN_BATCH_SIZE} -> 'Other'):")
    print(batch.value_counts())

    data = X.T  # neuroCombat wants features x samples
    covars = pd.DataFrame({"batch": batch.to_numpy()})
    result = neuroCombat(dat=data, covars=covars, batch_col="batch")
    harmonized = result["data"].T.astype(np.float32)

    assert harmonized.shape == X.shape
    np.save(OUT_NPY, harmonized)
    pd.DataFrame({"isic_id": isic_ids}).to_csv(OUT_IDS_CSV, index=False)
    print(f"[{VARIANT}] Saved ComBat-harmonized PanDerm embeddings to {OUT_NPY} ({harmonized.shape})")
