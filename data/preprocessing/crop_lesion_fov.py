"""Crop out the circular dermoscope FOV vignette / ruler marks / black corners
from melanoma-only images, to remove an acquisition-style confound found in
patch-meanpool clustering: small satellite
clusters turned out to be ~100% single-source by `attribution`, tracking a
distinct dermoscope acquisition style (vignette shape, ruler, ink) rather than
lesion biology.

Method (pragmatic heuristic, not true lesion segmentation):
1. Sample the 4 corners to estimate background color (white-bg or black-bg
   contact-dermoscope styles both occur in this dataset).
2. Threshold pixels far from that background color -> foreground mask
   (the circular FOV interior, ruler ink, lesion, skin).
3. Take the largest external contour's bounding box; if too small/too large
   (no clear circular border, e.g. black-corner contact style already
   near-square) fall back to a fixed center region.
4. Crop the largest square inscribed in that bounding circle (side =
   diameter / sqrt(2)), so vignette corners/ruler-near-border pixels are cut.
5. Resize back to 518x518 (=37*14, DINOv2 patch14 grid requirement).

Limitation: this is a heuristic, not segmentation --
ruler marks or ink close to the lesion center can survive, and the fallback
center-crop for ambiguous cases doesn't remove any border artifact at all.

Run with the default/system python3 (needs opencv-python + pandas/numpy only,
no torch/umap import -> no segfault risk).
"""
import os

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")
IN_DIR = os.path.join(ROOT, "data", "processed", "images_518")
OUT_DIR = os.path.join(ROOT, "data", "processed", "images_518_cropped_melanoma_only")

MELANOMA_DIAGNOSES = {"Melanoma, NOS", "Melanoma in situ", "Melanoma Invasive"}
OUT_SIZE = 518
AREA_FRAC_MIN, AREA_FRAC_MAX = 0.15, 0.97
BG_DIFF_THRESH = 20


def crop_fov(img):
    """Returns (cropped_bgr_uint8, used_fallback: bool)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    corners = [gray[0:15, 0:15], gray[0:15, -15:], gray[-15:, 0:15], gray[-15:, -15:]]
    corner_val = float(np.median([p.mean() for p in corners]))

    mask = (np.abs(gray.astype(np.float32) - corner_val) > BG_DIFF_THRESH).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fallback = True
    cx, cy, diam = w / 2, h / 2, min(w, h)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area_frac = cv2.contourArea(c) / (w * h)
        if AREA_FRAC_MIN <= area_frac <= AREA_FRAC_MAX:
            x, y, cw, ch = cv2.boundingRect(c)
            cx, cy, diam = x + cw / 2, y + ch / 2, max(cw, ch)
            fallback = False

    side = min(diam / np.sqrt(2), w, h)
    x0 = int(max(0, min(w - side, cx - side / 2)))
    y0 = int(max(0, min(h - side, cy - side / 2)))
    crop = img[y0:y0 + int(side), x0:x0 + int(side)]
    crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    return crop, fallback


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(METADATA_CSV, low_memory=False)
    df = df[df["diagnosis_3"].isin(MELANOMA_DIAGNOSES)].reset_index(drop=True)
    isic_ids = df["isic_id"].tolist()
    print(f"Melanoma-only subset: {len(isic_ids)} images")

    n_fallback, n_failed = 0, 0
    for i, isic_id in enumerate(isic_ids):
        in_path = os.path.join(IN_DIR, f"{isic_id}.jpg")
        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            n_failed += 1
            continue
        crop, fallback = crop_fov(img)
        n_fallback += fallback
        cv2.imwrite(os.path.join(OUT_DIR, f"{isic_id}.jpg"), crop)
        if i % 1000 == 0:
            print(f"{i}/{len(isic_ids)}", flush=True)

    print(f"done: {len(isic_ids)} total, {n_failed} unreadable, "
          f"{n_fallback} used fallback center-crop ({n_fallback/len(isic_ids):.1%})")

    assert n_failed == 0
    out_files = os.listdir(OUT_DIR)
    assert len(out_files) == len(isic_ids), f"{len(out_files)} written vs {len(isic_ids)} expected"
    sample = cv2.imread(os.path.join(OUT_DIR, out_files[0]))
    assert sample.shape == (OUT_SIZE, OUT_SIZE, 3)
    print("sanity checks passed")
