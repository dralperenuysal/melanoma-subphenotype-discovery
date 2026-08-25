"""Shades-of-Gray color constancy, p=6, exactly as specified.
Applied to the remediation-B excluded cohort's images (raw and cropped variants).
"""
import os
import sys
import numpy as np
import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIRS = {
    "raw": os.path.join(ROOT, "data", "processed", "images_518"),
    "cropped": os.path.join(ROOT, "data", "processed", "images_518_cropped_melanoma_only"),
}
DST_DIRS = {
    "raw": os.path.join(ROOT, "data", "processed", "images_518_socorrected_raw_excluded"),
    "cropped": os.path.join(ROOT, "data", "processed", "images_518_socorrected_cropped_excluded"),
}


def shades_of_gray(img, p=6):
    """img: HxWx3 float [0,1] -> color-constant image (protocol SS4, verbatim)."""
    img = img.astype(np.float64) + 1e-8
    illum = np.power(np.power(img, p).mean(axis=(0, 1)), 1.0 / p)
    illum = illum / np.sqrt((illum ** 2).sum())
    out = img / (illum * np.sqrt(3))
    return np.clip(out, 0, 1)


def run(variant):
    ids = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}_excluded.csv"))["image_id"].tolist()
    src, dst = SRC_DIRS[variant], DST_DIRS[variant]
    os.makedirs(dst, exist_ok=True)
    n_ok = n_missing = 0
    for iid in ids:
        p_src = os.path.join(src, f"{iid}.jpg")
        if not os.path.exists(p_src):
            n_missing += 1
            continue
        img = cv2.imread(p_src).astype(np.float32) / 255.0  # BGR, [0,1]
        corrected = shades_of_gray(img, p=6)
        cv2.imwrite(os.path.join(dst, f"{iid}.jpg"), (corrected * 255.0).clip(0, 255).astype(np.uint8))
        n_ok += 1
    print(f"{variant}: {n_ok} corrected, {n_missing} missing -> {dst}")


if __name__ == "__main__":
    for v in ["raw", "cropped"]:
        run(v)
