"""Build mean-pooled patch-token embeddings for the melanoma-only subset.

Alternative to the CLS token: averages each image's (1369, 768) patch grid
(37x37 DINOv2 patch14 tokens, flattened) over the spatial dimension to get one
768-dim vector per image. Global-average-pooling is the standard cheap
alternative to CLS pooling — full (37,37,768) per-image tensors are out of
scope for UMAP on ~6,714 samples (1M+ dims/sample).

Run with `.venv_cluster/bin/python` (default env segfaults on `import umap`
elsewhere in this project; h5py/numpy/pandas alone don't need it, but kept
consistent).
"""
import os

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_PATH = os.path.join(ROOT, "embeddings", "dinov2_patch", "patch_embeddings.h5")
METADATA_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")
OUT_NPY = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool.npy")
OUT_IDS_CSV = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_isic_ids.csv")

MELANOMA_DIAGNOSES = {"Melanoma, NOS", "Melanoma in situ", "Melanoma Invasive"}

if __name__ == "__main__":
    df = pd.read_csv(METADATA_CSV, low_memory=False)
    df = df[df["diagnosis_3"].isin(MELANOMA_DIAGNOSES)].reset_index(drop=True)
    isic_ids = df["isic_id"].tolist()
    print(f"Melanoma-only subset: {len(isic_ids)} images")

    with h5py.File(H5_PATH, "r") as f:
        missing = [iid for iid in isic_ids if iid not in f]
        assert not missing, f"{len(missing)} melanoma isic_ids missing from patch_embeddings.h5: {missing[:5]}"
        pooled = np.stack([f[iid][:].astype(np.float32).mean(axis=0) for iid in isic_ids])

    print(f"Pooled shape: {pooled.shape}")
    np.save(OUT_NPY, pooled)
    pd.DataFrame({"isic_id": isic_ids}).to_csv(OUT_IDS_CSV, index=False)
    print(f"Saved {OUT_NPY} and {OUT_IDS_CSV}")

    assert pooled.shape == (len(isic_ids), 768)
    assert not np.isnan(pooled).any()
    print("Sanity checks passed.")
