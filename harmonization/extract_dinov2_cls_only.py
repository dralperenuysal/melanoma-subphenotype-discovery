"""CLS-only DINOv2 ViT-B/14 extraction for the Shades-of-Gray-corrected excluded cohort.
Run on the GPU VM. Usage: python extract_dinov2_cls_only.py <image_dir> <out_dir>
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BATCH_SIZE = 32
NUM_WORKERS = 8


class ImageDataset(Dataset):
    def __init__(self, image_dir, isic_ids):
        self.image_dir = image_dir
        self.isic_ids = isic_ids

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        img = cv2.imread(os.path.join(self.image_dir, f"{isic_id}.jpg"), cv2.IMREAD_COLOR)
        if img is None:
            return isic_id, torch.zeros(3, 518, 518), False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return isic_id, tensor, True


def run(image_dir, out_dir):
    isic_ids = [f[:-4] for f in os.listdir(image_dir) if f.endswith(".jpg")]
    print(f"total images: {len(isic_ids)}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)

    loader = DataLoader(ImageDataset(image_dir, isic_ids), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    succeeded, failed = 0, 0
    t0 = time.time()
    with torch.inference_mode():
        for ids, imgs, ok in loader:
            imgs = imgs.to(device, non_blocking=True)
            out = model.forward_features(imgs)
            cls = out["x_norm_clstoken"].cpu().numpy().astype(np.float32)
            for i, isic_id in enumerate(ids):
                if not ok[i]:
                    failed += 1
                    continue
                np.save(os.path.join(out_dir, f"{isic_id}.npy"), cls[i])
                succeeded += 1
    print(f"succeeded={succeeded} failed={failed} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
