"""Interpretability figures: patch-level attention/contribution
maps + melanoma-only UMAP/representative-image panel, for the approved primary
melanoma-only analysis (patch-token mean-pool clustering, 4 clusters).

Attention-map method (caveat: not a rigorous attribution method): DINOv2 is
frozen and unsupervised here, so there is no gradient to attribute with. Instead,
per patch we compute cosine similarity between that patch's 768-dim embedding and
the mean-pooled embedding of its image's assigned cluster (the cluster centroid in
`embeddings/melanoma_patch_meanpool.npy` space) -- i.e. "how typical is this patch
of the cluster's identity". This is a similarity-based proxy for what's driving the
image toward its cluster, not a causal/gradient attribution like Grad-CAM.

Run with `.venv_cluster/bin/python` (h5py/numpy/pandas/matplotlib/PIL; default env
segfaults on unrelated imports elsewhere in this project).
"""
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
NOISE_COLOR = "#c9c7bf"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

FIG_DIR = "figures"
IMG_DIR = "data/processed/images_518"
H5_PATH = "embeddings/dinov2_patch/patch_embeddings.h5"
LABELS_CSV = "clustering/melanoma_only_patchpool/final_cluster_labels.csv"
UMAP_CSV = "clustering/melanoma_only_patchpool/final_umap_2d.csv"
POOLED_NPY = "embeddings/melanoma_patch_meanpool.npy"
POOLED_IDS_CSV = "embeddings/melanoma_patch_meanpool_isic_ids.csv"
SEED = 42  # reproducibility policy: fixed random_state
GRID_SIDE = 37  # DINOv2 patch14 grid for 518x518 input


def load_labels():
    return pd.read_csv(LABELS_CSV)


def cluster_centroids():
    pooled = np.load(POOLED_NPY)
    ids = pd.read_csv(POOLED_IDS_CSV)["isic_id"].tolist()
    labels = load_labels().set_index("isic_id").loc[ids, "cluster_label"].values
    centroids = {}
    for c in np.unique(labels):
        if c == -1:
            continue
        centroids[c] = pooled[labels == c].mean(axis=0)
    return centroids


def patch_attention_map(isic_id, centroid, h5file):
    patches = h5file[isic_id][:].astype(np.float32)  # (1369, 768)
    patches = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
    c = centroid / (np.linalg.norm(centroid) + 1e-8)
    sim = patches @ c  # (1369,) cosine similarity per patch
    return sim.reshape(GRID_SIDE, GRID_SIDE)


def fig17_patch_attention_maps(labels_df, n_per_cluster=4):
    rng = np.random.RandomState(SEED)
    centroids = cluster_centroids()
    # focus on the two clinically interesting satellite clusters (0=in-situ-skewed,
    # 2=invasive-skewed, see Step 5) plus one row from the dominant cluster 1 for contrast
    rows = [0, 2, 1]
    rows = [c for c in rows if c in centroids]

    fig, axes = plt.subplots(len(rows), n_per_cluster, figsize=(3 * n_per_cluster, 3.2 * len(rows)))
    with h5py.File(H5_PATH, "r") as h5f:
        for i, c in enumerate(rows):
            members = labels_df.loc[labels_df["cluster_label"] == c, "isic_id"].values
            picks = rng.choice(members, size=min(n_per_cluster, len(members)), replace=False)
            for j in range(n_per_cluster):
                ax = axes[i][j]
                ax.axis("off")
                if j >= len(picks):
                    continue
                isic_id = picks[j]
                img_path = os.path.join(IMG_DIR, f"{isic_id}.jpg")
                if not os.path.exists(img_path):
                    continue
                img = np.array(Image.open(img_path))
                sim_map = patch_attention_map(isic_id, centroids[c], h5f)
                heat = np.array(
                    Image.fromarray(sim_map).resize((img.shape[1], img.shape[0]), Image.BILINEAR)
                )
                ax.imshow(img)
                ax.imshow(heat, cmap="magma", alpha=0.45)
                if j == 0:
                    ax.text(-0.15, 0.5, f"Cluster {c}", transform=ax.transAxes, fontsize=11,
                            color=INK, rotation=90, va="center", ha="center")
    fig.suptitle(
        "Patch-level contribution maps (cosine similarity to assigned cluster centroid)\n"
        "similarity-based proxy, not gradient attribution -- DINOv2 is frozen/unsupervised here",
        fontsize=11, color=INK,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "17_patch_attention_maps.png"), dpi=150)
    plt.close(fig)


def fig18_umap_by_cluster():
    umap_df = pd.read_csv(UMAP_CSV).merge(load_labels(), on="isic_id")
    fig, ax = plt.subplots(figsize=(9, 7))
    clusters = sorted(c for c in umap_df["cluster_label"].unique() if c != -1)
    for i, c in enumerate(clusters):
        sel = umap_df["cluster_label"] == c
        ax.scatter(umap_df.loc[sel, "umap_x"], umap_df.loc[sel, "umap_y"],
                   s=6, alpha=0.6, color=CAT[i % len(CAT)], label=f"Cluster {c} (n={sel.sum()})")
    noise = umap_df["cluster_label"] == -1
    if noise.any():
        ax.scatter(umap_df.loc[noise, "umap_x"], umap_df.loc[noise, "umap_y"],
                   s=6, alpha=0.6, color=NOISE_COLOR, marker="x", label=f"Noise (n={noise.sum()})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Melanoma-only UMAP projection, colored by patch-meanpool cluster")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "18_melanoma_only_umap_by_cluster.png"), dpi=150)
    plt.close(fig)


def fig19_representative_images(labels_df, n_samples=8):
    rng = np.random.RandomState(SEED)
    clusters = sorted(c for c in labels_df["cluster_label"].unique() if c != -1)

    fig, axes = plt.subplots(len(clusters), n_samples, figsize=(2 * n_samples, 2 * len(clusters)))
    missing = 0
    for i, c in enumerate(clusters):
        members = labels_df.loc[labels_df["cluster_label"] == c, "isic_id"].values
        picks = rng.choice(members, size=min(n_samples, len(members)), replace=False)
        for j in range(n_samples):
            ax = axes[i][j]
            ax.axis("off")
            if j >= len(picks):
                continue
            path = os.path.join(IMG_DIR, f"{picks[j]}.jpg")
            if os.path.exists(path):
                ax.imshow(plt.imread(path))
            else:
                missing += 1
            if j == 0:
                ax.text(-0.15, 0.5, f"Cluster {c}", transform=ax.transAxes, fontsize=11,
                        color=INK, rotation=90, va="center", ha="center")
    if missing:
        print(f"WARNING: {missing} representative image files not found")
    fig.suptitle("Melanoma-only: representative dermoscopy images sampled per cluster",
                 fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "19_melanoma_only_cluster_representative_images.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    labels_df = load_labels()
    fig17_patch_attention_maps(labels_df)
    print("Saved 17_patch_attention_maps.png")
    fig18_umap_by_cluster()
    print("Saved 18_melanoma_only_umap_by_cluster.png")
    fig19_representative_images(labels_df)
    print("Saved 19_melanoma_only_cluster_representative_images.png")
