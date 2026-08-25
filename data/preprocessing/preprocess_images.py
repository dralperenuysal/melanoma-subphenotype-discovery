"""Resize quality-filtered ISIC images to DINOv2 input resolution (Step 1 preprocessing).

Preprocessing: fixed resolution matching DINOv2 input; color normalization;
artifact masking is optional.

Design choices (ponytail-lazy):
- Resize only, save as uint8 RGB jpg. ImageNet mean/std normalization is a
  model-input-time transform, not a stored-file step — storing normalized
  float32 arrays for 28,705 images would be ~46GB vs a few hundred MB of jpgs.
  IMAGENET_MEAN/STD below are the constants the embedding-extraction step reuses.
- Hair/artifact removal (DullRazor) skipped: it is optional, and
  heavily-artifacted images were already dropped by filter_quality.py.
- 518x518 = 37*14 patches, standard DINOv2 vitb14/vitl14 eval resolution.
"""
import os
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")
IMAGE_DIR = os.path.join(ROOT, "data", "raw", "isic_dermoscopic_targeted")
OUT_DIR = os.path.join(ROOT, "data", "processed", "images_518")

TARGET_SIZE = 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize(img_uint8_rgb):
    """ImageNet mean/std normalization, for the embedding-extraction dataloader to reuse."""
    x = img_uint8_rgb.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def process_one(isic_id):
    src = os.path.join(IMAGE_DIR, f"{isic_id}.jpg")
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return False
    resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    dst = os.path.join(OUT_DIR, f"{isic_id}.jpg")
    return bool(cv2.imwrite(dst, resized))


def run():
    df = pd.read_csv(IN_CSV, low_memory=False)
    total = len(df)
    os.makedirs(OUT_DIR, exist_ok=True)

    ok, failed = 0, []
    for isic_id in df["isic_id"]:
        if process_one(isic_id):
            ok += 1
        else:
            failed.append(isic_id)

    print(f"input rows:      {total}")
    print(f"processed ok:    {ok}")
    print(f"failed/skipped:  {len(failed)}")
    if failed:
        print(f"first few failed: {failed[:5]}")
    return ok, failed


if __name__ == "__main__":
    ok, failed = run()
    assert ok > 0
    out_files = os.listdir(OUT_DIR)
    assert len(out_files) == ok
    sample = cv2.imread(os.path.join(OUT_DIR, out_files[0]))
    assert sample.shape == (TARGET_SIZE, TARGET_SIZE, 3)
