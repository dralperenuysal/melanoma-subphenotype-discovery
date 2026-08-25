"""Same as build_melanoma_patch_meanpool.py, but from the FOV-cropped
re-extraction (confound remediation). Run with .venv_cluster/bin/python.
"""
import os

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H5_PATH = os.path.join(ROOT, "embeddings", "dinov2_patch_cropped_melanoma_only",
                        "patch_embeddings_cropped_melanoma_only.h5")
POOLED_IDS_CSV_ORIG = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_isic_ids.csv")
OUT_NPY = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_cropped.npy")
OUT_IDS_CSV = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_cropped_isic_ids.csv")

if __name__ == "__main__":
    # Same isic_id order as the original (uncropped) melanoma-only run, for direct comparability.
    isic_ids = pd.read_csv(POOLED_IDS_CSV_ORIG)["isic_id"].tolist()
    print(f"Melanoma-only subset: {len(isic_ids)} images")

    with h5py.File(H5_PATH, "r") as f:
        missing = [iid for iid in isic_ids if iid not in f]
        assert not missing, f"{len(missing)} isic_ids missing from cropped patch h5: {missing[:5]}"
        pooled = np.stack([f[iid][:].astype(np.float32).mean(axis=0) for iid in isic_ids])

    print(f"Pooled shape: {pooled.shape}")
    np.save(OUT_NPY, pooled)
    pd.DataFrame({"isic_id": isic_ids}).to_csv(OUT_IDS_CSV, index=False)
    print(f"Saved {OUT_NPY} and {OUT_IDS_CSV}")

    assert pooled.shape == (len(isic_ids), 768)
    assert not np.isnan(pooled).any()
    print("Sanity checks passed.")
