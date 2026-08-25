"""Segment the lesion out of the already FOV-cropped melanoma-only images and
mask out everything else, to further reduce the acquisition-style confound
found in patch-meanpool clustering:
cropping alone (`crop_lesion_fov.py`) reduced but did not eliminate a
source-institution confound. Masking out all non-lesion pixels should remove
acquisition-style signal (skin texture, remaining vignette/ruler fragments)
much more thoroughly than a square crop.

Method (classical/pragmatic, no deep-learning segmentation model -- consistent
with this project's existing FOV-crop heuristic):
1. Estimate normal-skin color from the image's outer ring (10% margin) in LAB
   space (mean + std per channel) -- the ring is assumed to be mostly skin,
   since the FOV crop already removed the vignette border.
2. Classify each pixel as "lesion" if its LAB distance from the ring's mean
   color exceeds a threshold (in ring std units) -- pigmented lesions are
   reliably darker/more chromatic than surrounding skin in dermoscopy images.
3. Keep only the largest connected component that overlaps the image center
   (the lesion should dominate the central region after cropping), morphological
   close+open to clean up, fill holes.
4. If the resulting mask covers too small or too large a fraction of the image
   (no clear lesion boundary), fall back to keeping the image unmasked, and log
   the fallback rate -- do not crash.
5. Replace non-lesion pixels with plain mid-gray (127,127,127): simplest,
   content-free fill that shouldn't read as "another texture" to DINOv2 (unlike
   e.g. mean skin color, which is itself a texture-adjacent signal).

Limitation: this is a color-threshold heuristic, not
trained segmentation -- it can fail on low-contrast lesions or non-skin-colored
artifacts remaining after cropping (kept in "fallback" cases, unmasked).

Run with the default/system python3 (opencv + numpy/pandas only).
"""
import os

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")
IN_DIR = os.path.join(ROOT, "data", "processed", "images_518_cropped_melanoma_only")
OUT_DIR = os.path.join(ROOT, "data", "processed", "images_518_segmented_melanoma_only")

MELANOMA_DIAGNOSES = {"Melanoma, NOS", "Melanoma in situ", "Melanoma Invasive"}
RING_FRAC = 0.10
DIST_THRESH_STD = 2.2
AREA_FRAC_MIN, AREA_FRAC_MAX = 0.03, 0.90
GRAY_FILL = 127


def segment_lesion(img):
    """Returns (masked_bgr_uint8, used_fallback: bool)."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    m = max(2, int(RING_FRAC * min(h, w)))
    ring_mask = np.ones((h, w), dtype=bool)
    ring_mask[m:h - m, m:w - m] = False
    ring_pixels = lab[ring_mask]
    ring_mean = ring_pixels.mean(axis=0)
    ring_std = ring_pixels.std(axis=0) + 1e-6

    dist = np.linalg.norm((lab - ring_mean) / ring_std, axis=2)
    fg = (dist > DIST_THRESH_STD).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = None
    if contours:
        cx0, cy0 = w / 2, h / 2
        central = [c for c in contours if cv2.pointPolygonTest(c, (cx0, cy0), False) >= 0]
        candidates = central if central else contours
        c = max(candidates, key=cv2.contourArea)
        area_frac = cv2.contourArea(c) / (w * h)
        if AREA_FRAC_MIN <= area_frac <= AREA_FRAC_MAX:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)

    if mask is None:
        return img.copy(), True

    out = np.full_like(img, GRAY_FILL)
    out[mask > 0] = img[mask > 0]
    return out, False


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
        out, fallback = segment_lesion(img)
        n_fallback += fallback
        cv2.imwrite(os.path.join(OUT_DIR, f"{isic_id}.jpg"), out)
        if i % 1000 == 0:
            print(f"{i}/{len(isic_ids)}", flush=True)

    print(f"done: {len(isic_ids)} total, {n_failed} unreadable, "
          f"{n_fallback} used fallback (no mask applied) ({n_fallback/len(isic_ids):.1%})")

    assert n_failed == 0
    out_files = os.listdir(OUT_DIR)
    assert len(out_files) == len(isic_ids), f"{len(out_files)} written vs {len(isic_ids)} expected"
