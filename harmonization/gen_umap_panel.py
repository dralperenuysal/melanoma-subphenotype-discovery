"""2x3 UMAP panel (raw|harmony|combat columns, dataset|diagnosis rows)."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from umap.umap_ import UMAP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gen(variant):
    out_dir = os.path.join(ROOT, "results", "harmonization_balanced", variant)
    meta = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}.csv"))
    ids = set(json.load(open(os.path.join(ROOT, "harmonization", "remediation_a_result.json")))["image_ids"])
    meta = meta[meta["image_id"].isin(ids)].reset_index(drop=True)

    mats = {
        "raw": None,  # filled from Z_raw recompute below for consistency isn't saved; use PCA-free? just reuse combat/harmony npy + recompute raw PCA
        "harmony": np.load(os.path.join(out_dir, "02_harmony.npy")),
        "combat": np.load(os.path.join(out_dir, "03_combat.npy")),
    }
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = np.load(os.path.join(ROOT, "data", "embeddings", f"dinov2_embeddings_{variant}.npy"))
    full_meta = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}.csv"))
    mask = full_meta["image_id"].isin(ids).values
    X = X[mask].astype(np.float32)
    mats["raw"] = PCA(n_components=50, random_state=0).fit_transform(StandardScaler().fit_transform(X))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    cols = ["raw", "harmony", "combat"]
    for j, name in enumerate(cols):
        emb = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(mats[name])
        for i, color_col in enumerate(["dataset", "diagnosis"]):
            ax = axes[i, j]
            cats = pd.Categorical(meta[color_col])
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=cats.codes, cmap="tab20", s=6, alpha=0.7)
            ax.set_title(f"{name} / {color_col}")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Remediation A (balanced subsample) - {variant}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_umap_panel.png"), dpi=150)
    plt.close(fig)
    print(f"{variant}: saved panel")


if __name__ == "__main__":
    for v in ["raw", "cropped"]:
        gen(v)
