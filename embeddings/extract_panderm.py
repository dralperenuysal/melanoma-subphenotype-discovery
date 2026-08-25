"""Extract frozen PanDerm_Base (ViT-B/16) CLS + patch-token embeddings.

Same role as embeddings/extract_dinov2.py but for the PanDerm dermatology
foundation model. Uses PanDerm's own
official preprocessing (resize 256 + center-crop 224, ImageNet norm) -- NOT
DINOv2's 518x518 pipeline -- per the experiment spec's explicit instruction not
to copy DINOv2 preprocessing.

Storage convention matches extract_dinov2.py: CLS as one .npy per image,
patch tokens (196 patches x 768-dim per image, float16) in one HDF5 file.

No randomness (pure inference) -- reproducibility policy trivially satisfied.
"""
import os
import sys
import time

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/workspace/panderm/PanDerm/classification")
from models import get_encoder  # noqa: E402

IMAGE_DIR = sys.argv[1]
CLS_DIR = sys.argv[2]
PATCH_H5 = sys.argv[3]
CHECKPOINT = "/workspace/panderm/panderm_base.pth"
BATCH_SIZE = 64
NUM_WORKERS = 8


class Args:
    pretrained_checkpoint = CHECKPOINT


class ImageDataset(Dataset):
    def __init__(self, isic_ids, transform):
        self.isic_ids = isic_ids
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        path = os.path.join(IMAGE_DIR, f"{isic_id}.jpg")
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            ok = True
        except Exception:
            tensor = torch.zeros(3, 224, 224)
            ok = False
        return isic_id, tensor, ok


def get_all_tokens(model, x):
    """Replicates VisionTransformer.forward_features but returns the full
    (cls + patch) token sequence instead of just the CLS token -- the
    official forward_features only returns x[:, 0], patch tokens are not
    exposed by the official API so this reimplements the same forward pass.
    """
    x = model.patch_embed(x)
    b, seq_len, _ = x.size()
    cls_tokens = model.cls_token.expand(b, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    if model.pos_embed is not None:
        x = x + model.pos_embed.expand(b, -1, -1).type_as(x).to(x.device)
    x = model.pos_drop(x)
    rel_pos_bias = model.rel_pos_bias() if model.rel_pos_bias is not None else None
    for blk in model.blocks:
        x = blk(x, rel_pos_bias=rel_pos_bias)
    x = model.norm(x)
    return x  # (B, 1+num_patches, embed_dim)


def run():
    isic_ids = sorted(f[:-4] for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg"))
    total = len(isic_ids)
    print(f"total images: {total}")

    os.makedirs(CLS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PATCH_H5), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, eval_transform = get_encoder(Args(), model_name="PanDerm_Base_LP")
    model.eval().to(device)

    loader = DataLoader(
        ImageDataset(isic_ids, eval_transform), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    succeeded, failed = 0, []
    t0 = time.time()
    with h5py.File(PATCH_H5, "w") as h5, torch.inference_mode():
        for batch_idx, (ids, imgs, ok) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            all_tok = get_all_tokens(model, imgs)
            cls = all_tok[:, 0, :].cpu().numpy().astype(np.float32)
            patches = all_tok[:, 1:, :].cpu().numpy().astype(np.float16)

            for i, isic_id in enumerate(ids):
                if not ok[i]:
                    failed.append((isic_id, "unreadable image"))
                    continue
                np.save(os.path.join(CLS_DIR, f"{isic_id}.npy"), cls[i])
                h5.create_dataset(isic_id, data=patches[i])
                succeeded += 1

            if batch_idx % 10 == 0:
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
    return succeeded, failed


if __name__ == "__main__":
    succeeded, failed = run()
    assert succeeded > 0
    assert os.path.exists(PATCH_H5)
    sample_cls = np.load(os.path.join(CLS_DIR, os.listdir(CLS_DIR)[0]))
    assert sample_cls.shape == (768,), f"unexpected CLS shape {sample_cls.shape}"
    with h5py.File(PATCH_H5, "r") as h5:
        sample_key = next(iter(h5.keys()))
        assert h5[sample_key].shape == (196, 768), f"unexpected patch shape {h5[sample_key].shape}"
    print("sanity checks passed")
