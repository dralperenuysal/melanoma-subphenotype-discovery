"""Extract frozen DINOv2 CLS + patch-token embeddings for all quality-filtered images.

DINOv2 (frozen, no fine-tuning), both CLS token (global) and
patch-token grid (local) are kept.

Model: dinov2_vitb14 (ViT-B/14), not ViT-L/14 -- this runs on a single rented
vast.ai RTX 4090; ViT-B keeps the full 28,705-image run inside a practical
time/cost budget on one GPU, which is the deciding compute constraint for
the model-size choice.

Storage: CLS embeddings (768-dim, tiny) are one .npy per image. Patch grids
(37*37=1369 patches x 768-dim per image) would be ~115GB at float32 for 28,705
images, so they are (a) stored as float16 (halves size) and (b) written into a
single sharded HDF5 file rather than per-image .npy, keeping total patch storage
in the tens-of-GB range instead of hundreds.

No randomness in this step (pure inference, shuffle=False) -- reproducibility
policy is trivially satisfied.
"""
import os
import sys
import time

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")
IMAGE_DIR = os.path.join(ROOT, "data", "processed", "images_518")
CLS_DIR = os.path.join(ROOT, "embeddings", "dinov2_cls")
PATCH_DIR = os.path.join(ROOT, "embeddings", "dinov2_patch")
PATCH_H5 = os.path.join(PATCH_DIR, "patch_embeddings.h5")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BATCH_SIZE = 32
NUM_WORKERS = 8


class ImageDataset(Dataset):
    def __init__(self, isic_ids):
        self.isic_ids = isic_ids

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        img = cv2.imread(os.path.join(IMAGE_DIR, f"{isic_id}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            return isic_id, torch.zeros(3, 518, 518), False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return isic_id, tensor, True


def run():
    df = pd.read_csv(IN_CSV, low_memory=False)
    isic_ids = df["isic_id"].tolist()
    total = len(isic_ids)
    print(f"total images: {total}")

    os.makedirs(CLS_DIR, exist_ok=True)
    os.makedirs(PATCH_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)

    loader = DataLoader(
        ImageDataset(isic_ids), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    succeeded, failed = 0, []
    t0 = time.time()
    with h5py.File(PATCH_H5, "w") as h5, torch.inference_mode():
        for batch_idx, (ids, imgs, ok) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            out = model.forward_features(imgs)
            cls = out["x_norm_clstoken"].cpu().numpy().astype(np.float32)
            patches = out["x_norm_patchtokens"].cpu().numpy().astype(np.float16)

            for i, isic_id in enumerate(ids):
                if not ok[i]:
                    failed.append((isic_id, "unreadable image"))
                    continue
                np.save(os.path.join(CLS_DIR, f"{isic_id}.npy"), cls[i])
                h5.create_dataset(isic_id, data=patches[i])
                succeeded += 1

            if batch_idx % 20 == 0:
                elapsed = time.time() - t0
                done = batch_idx * BATCH_SIZE
                print(f"batch {batch_idx}: {done}/{total} images, {elapsed:.0f}s elapsed", flush=True)

    elapsed = time.time() - t0
    print(f"\ntotal images:  {total}")
    print(f"succeeded:     {succeeded}")
    print(f"failed:        {len(failed)}")
    for isic_id, reason in failed[:20]:
        print(f"  {isic_id}: {reason}")
    print(f"elapsed:       {elapsed:.0f}s")
    print(f"CLS dir:       {CLS_DIR}")
    print(f"patch h5:      {PATCH_H5}")
    return succeeded, failed


if __name__ == "__main__":
    succeeded, failed = run()
    assert succeeded > 0
    assert os.path.exists(PATCH_H5)
    sample_cls = np.load(os.path.join(CLS_DIR, os.listdir(CLS_DIR)[0]))
    assert sample_cls.shape == (768,), f"unexpected CLS shape {sample_cls.shape}"
    with h5py.File(PATCH_H5, "r") as h5:
        sample_key = next(iter(h5.keys()))
        assert h5[sample_key].shape == (1369, 768), f"unexpected patch shape {h5[sample_key].shape}"
    print("sanity checks passed")
