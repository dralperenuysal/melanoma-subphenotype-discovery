"""Batch-effect harmonization of melanoma-only cropped patch-meanpool embeddings,
analogous to ComBat batch correction in RNAseq, using `attribution` (source
institution) as the batch variable.

No biological covariate is protected in the ComBat model (diagnosis_3 is exactly
what we're trying to discover, not something to preserve) — so ComBat may also
remove real biological signal that happens to correlate with batch. Same caveat
as simple batch-mean-centering.

Batches with <30 images are grouped into a single "Other" batch (ComBat needs
enough samples per batch to estimate variance reliably).

Run with `.venv_cluster/bin/python` (has neuroCombat installed).
"""
import os

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLED_NPY = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_cropped.npy")
POOLED_IDS_CSV = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_cropped_isic_ids.csv")
METADATA_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")

OUT_NPY = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_cropped_combat.npy")
OUT_IDS_CSV = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_cropped_combat_isic_ids.csv")

MIN_BATCH_SIZE = 30

if __name__ == "__main__":
    isic_ids = pd.read_csv(POOLED_IDS_CSV)["isic_id"].tolist()
    X = np.load(POOLED_NPY).astype(np.float64)  # neuroCombat wants float64
    assert X.shape[0] == len(isic_ids)

    meta = pd.read_csv(METADATA_CSV, low_memory=False).set_index("isic_id")
    attribution = meta.loc[isic_ids, "attribution"].fillna("Anonymous")

    counts = attribution.value_counts()
    small_batches = counts[counts < MIN_BATCH_SIZE].index
    batch = attribution.where(~attribution.isin(small_batches), "Other")
    print(f"Batch sizes after grouping (<{MIN_BATCH_SIZE} -> 'Other'):")
    print(batch.value_counts())

    # neuroCombat expects features x samples
    data = X.T
    covars = pd.DataFrame({"batch": batch.to_numpy()})
    result = neuroCombat(dat=data, covars=covars, batch_col="batch")
    harmonized = result["data"].T.astype(np.float32)

    assert harmonized.shape == X.shape
    np.save(OUT_NPY, harmonized)
    pd.DataFrame({"isic_id": isic_ids}).to_csv(OUT_IDS_CSV, index=False)
    print(f"Saved ComBat-harmonized embeddings to {OUT_NPY} ({harmonized.shape})")
