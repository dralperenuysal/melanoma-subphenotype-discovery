"""Generate 2 exploratory CLS-embedding figures for the melanoma subphenotype
discovery project. Uses `embeddings/dinov2_cls/*.npy` (768-dim DINOv2 ViT-B/14
CLS vectors) joined against `data/processed/quality_filtered_metadata.csv`.

Palette: same reference instance as `generate_metadata_figures.py` — reused
verbatim for visual consistency.

Figure 8 is a PRELIMINARY 2D projection (PCA, default params) for a quick look
at global structure — it is NOT the systematic UMAP+HDBSCAN sweep used
in the main clustering analysis; that sweep is a separate, later piece of work. umap-learn is
not installed in this environment, and installing it just for a one-off preview
plot would be scope creep — PCA is already a project dependency (scikit-learn)
and is sufficient to sanity-check the embeddings before the real Step 3 sweep.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# --- palette (dataviz skill reference instance, light mode) ---
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

FIG_DIR = "figures"
CLS_DIR = "embeddings/dinov2_cls"
META_CSV = "data/processed/quality_filtered_metadata.csv"
SEED = 42  # reproducibility policy: fixed random_state


def load_cls_embeddings(meta):
    vecs = np.empty((len(meta), 768), dtype=np.float32)
    missing = []
    for i, isic_id in enumerate(meta["isic_id"]):
        path = os.path.join(CLS_DIR, f"{isic_id}.npy")
        if os.path.exists(path):
            vecs[i] = np.load(path)
        else:
            vecs[i] = np.nan
            missing.append(isic_id)
    if missing:
        print(f"WARNING: {len(missing)} isic_ids have no CLS embedding file")
    return vecs


def fig8_projection(meta, vecs):
    nan_mask = np.isnan(vecs).any(axis=1)
    valid = ~nan_mask
    top_diag = meta["diagnosis_1"].value_counts().head(len(CAT) - 1).index
    diag = meta["diagnosis_1"].where(meta["diagnosis_1"].isin(top_diag), "Other").fillna("Unknown")

    proj = PCA(n_components=2, random_state=SEED).fit_transform(vecs[valid])

    fig, ax = plt.subplots(figsize=(9, 7))
    categories = list(top_diag) + ["Other"]
    for i, cat in enumerate(categories):
        sel = (diag[valid].values == cat)
        ax.scatter(proj[sel, 0], proj[sel, 1], s=4, alpha=0.5, color=CAT[i % len(CAT)], label=cat)

    ax.set_title(
        "CLS embedding — preliminary 2D projection (PCA, default params)\n"
        "Not the systematic UMAP+HDBSCAN sweep — quick sanity check only",
        fontsize=11, color=INK,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    leg = ax.legend(markerscale=4, fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_cls_embedding_projection.png"), dpi=150)
    plt.close(fig)


def fig9_norms(vecs):
    norms = np.linalg.norm(vecs, axis=1)
    nan_count = np.isnan(norms).sum()
    zero_count = int((norms == 0).sum())
    print(f"CLS embedding norm check: {nan_count} NaN, {zero_count} exactly-zero (of {len(norms)})")

    finite = norms[np.isfinite(norms)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finite, bins=60, color=CAT[0], edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlabel("L2 norm of CLS embedding")
    ax.set_ylabel("Count")
    title = "CLS embedding L2-norm distribution (sanity check for degenerate vectors)"
    if nan_count or zero_count:
        title += f"\n{nan_count} NaN, {zero_count} zero-norm flagged"
    ax.set_title(title, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "09_cls_embedding_norms.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    meta = pd.read_csv(META_CSV, low_memory=False)
    vecs = load_cls_embeddings(meta)
    fig8_projection(meta, vecs)
    fig9_norms(vecs)

    for name in ["08_cls_embedding_projection.png", "09_cls_embedding_norms.png"]:
        p = os.path.join(FIG_DIR, name)
        assert os.path.getsize(p) > 10_000, f"{p} looks too small/corrupt"
    print("figures 8-9 written and sanity-checked")
