"""Filter biopsy-confirmed ISIC images by resolution and visual artifact load.

Inclusion criteria: low-resolution, heavily artifacted (hair, bubbles,
ruler marks) images are excluded via pre-filtering.

Resolution: shorter side must be >= MIN_SIDE (DINOv2 patch-grid input floor).
Artifact: DullRazor-style blackhat morphological hair detection — a dark,
line-like structure (hair, ruler edge) blackhat-highlights against skin on
the grayscale channel. Fraction of pixels above an intensity threshold after
blackhat is used as a hair/clutter density proxy; images above ARTIFACT_FRAC
are dropped. Thresholds (BLACKHAT_THRESH=10, ARTIFACT_FRAC=0.08) were picked
by eyeballing a handful of heavily-haired ISIC images vs clean ones — not a
tuned classifier, ponytail: revisit with a labeled artifact set if dermatologist
review later flags too many hairy images slipping through.
"""
import os
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_CSV = os.path.join(ROOT, "data", "processed", "biopsy_confirmed_metadata.csv")
IMAGE_DIR = os.path.join(ROOT, "data", "raw", "isic_dermoscopic_targeted")
OUT_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")

MIN_SIDE = 224
BLACKHAT_THRESH = 10
ARTIFACT_FRAC = 0.08


def has_heavy_artifact(path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if img is None:
        return True  # unreadable file -> exclude
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    frac = np.mean(blackhat > BLACKHAT_THRESH)
    return frac > ARTIFACT_FRAC


def run():
    df = pd.read_csv(IN_CSV, low_memory=False)
    total = len(df)

    short_side = df[["pixels_x", "pixels_y"]].min(axis=1)
    res_ok = short_side >= MIN_SIDE
    excluded_resolution = total - res_ok.sum()
    after_res = df[res_ok]

    artifact_flags = after_res["isic_id"].apply(
        lambda isic_id: has_heavy_artifact(os.path.join(IMAGE_DIR, f"{isic_id}.jpg"))
    )
    final = after_res[~artifact_flags]
    excluded_artifact = len(after_res) - len(final)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    final.to_csv(OUT_CSV, index=False)

    print(f"input rows:                 {total}")
    print(f"excluded (resolution < {MIN_SIDE}px): {excluded_resolution}")
    print(f"excluded (artifact heuristic):  {excluded_artifact}")
    print(f"final rows:                 {len(final)}")
    print(f"written to: {OUT_CSV}")
    return final


if __name__ == "__main__":
    result = run()
    assert len(result) > 0
    assert set(result["isic_id"]).issubset(set(pd.read_csv(IN_CSV)["isic_id"]))
