"""Figure 20: before/after sanity-check for the FOV-crop confound fix
(data/preprocessing/crop_lesion_fov.py). Same palette as the other figures/*.py.
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, SURFACE = "#0b0b0b", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK})

ORIG_DIR = "data/processed/images_518"
CROP_DIR = "data/processed/images_518_cropped_melanoma_only"
LABELS_CSV = "clustering/melanoma_only_patchpool/final_cluster_labels.csv"
OUT_PNG = "figures/20_fov_crop_before_after.png"
SEED = 42

if __name__ == "__main__":
    labels = pd.read_csv(LABELS_CSV)
    rng = np.random.default_rng(SEED)
    samples = []
    for c in sorted(labels["cluster_label"].unique()):
        ids = labels[labels["cluster_label"] == c]["isic_id"].tolist()
        n = min(3, len(ids))
        picks = rng.choice(ids, size=n, replace=False)
        samples += [(c, iid) for iid in picks]

    fig, axes = plt.subplots(2, len(samples), figsize=(2.2 * len(samples), 4.6))
    for col, (c, iid) in enumerate(samples):
        orig = cv2.cvtColor(cv2.imread(os.path.join(ORIG_DIR, f"{iid}.jpg")), cv2.COLOR_BGR2RGB)
        crop = cv2.cvtColor(cv2.imread(os.path.join(CROP_DIR, f"{iid}.jpg")), cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(orig)
        axes[0, col].set_title(f"cluster {c}\n{iid}", fontsize=8)
        axes[1, col].imshow(crop)
        for row in (0, 1):
            axes[row, col].axis("off")
    axes[0, 0].set_ylabel("original", fontsize=9)
    axes[1, 0].set_ylabel("cropped", fontsize=9)
    fig.suptitle("FOV-crop confound fix: before (top) / after (bottom)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")

    assert os.path.exists(OUT_PNG)
    print("sanity check passed")
