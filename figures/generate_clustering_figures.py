"""Generate 5 clustering-result figures (10-14) for the melanoma subphenotype
discovery project, from the Step 3/4 UMAP+HDBSCAN pipeline outputs in
`clustering/`. Palette: same reference instance as the other figures/*.py
scripts — reused verbatim.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
IMG_DIR = "data/processed/images_518"
META_CSV = "data/processed/quality_filtered_metadata.csv"
UMAP_CSV = "clustering/umap_projections/final_umap_2d.csv"
LABELS_CSV = "clustering/cluster_assignments/final_cluster_labels.csv"
SWEEP_CSV = "clustering/cluster_assignments/sweep_results.csv"
VALIDATION_JSON = "clustering/stability_analysis/step4_validation_report.json"
SEED = 42  # reproducibility policy: fixed random_state

WINNING_CONFIG = dict(n_neighbors=50, min_dist=0.25, min_cluster_size=15, min_samples=15)
NOISE_COLOR = "#c9c7bf"


def load_umap_and_labels():
    umap_df = pd.read_csv(UMAP_CSV)
    labels_df = pd.read_csv(LABELS_CSV)
    return umap_df.merge(labels_df, on="isic_id")


def fig10_umap_by_cluster(df):
    fig, ax = plt.subplots(figsize=(9, 7))
    clusters = sorted(c for c in df["cluster_label"].unique() if c != -1)
    for i, c in enumerate(clusters):
        sel = df["cluster_label"] == c
        ax.scatter(df.loc[sel, "umap_x"], df.loc[sel, "umap_y"], s=4, alpha=0.5,
                   color=CAT[i % len(CAT)], label=f"Cluster {c} (n={sel.sum()})")
    noise = df["cluster_label"] == -1
    if noise.any():
        ax.scatter(df.loc[noise, "umap_x"], df.loc[noise, "umap_y"], s=20, alpha=0.9,
                   color=NOISE_COLOR, marker="x", label=f"Noise (n={noise.sum()})")
    ax.set_title("UMAP projection colored by HDBSCAN cluster", fontsize=11, color=INK)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "10_umap_by_cluster.png"), dpi=150)
    plt.close(fig)


def fig11_umap_by_diagnosis(df, meta):
    merged = df.merge(meta[["isic_id", "diagnosis_1"]], on="isic_id", how="left")
    # group rare diagnosis categories into "Other" so the legend stays readable
    top_diag = merged["diagnosis_1"].value_counts().head(len(CAT) - 1).index
    diag = merged["diagnosis_1"].where(merged["diagnosis_1"].isin(top_diag), "Other").fillna("Unknown")

    fig, ax = plt.subplots(figsize=(9, 7))
    categories = list(top_diag) + ["Other"]
    for i, cat in enumerate(categories):
        sel = diag.values == cat
        if sel.sum() == 0:
            continue
        ax.scatter(merged.loc[sel, "umap_x"], merged.loc[sel, "umap_y"], s=4, alpha=0.5,
                   color=CAT[i % len(CAT)], label=cat)
    ax.set_title("Same UMAP projection colored by clinical diagnosis", fontsize=11, color=INK)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "11_umap_by_diagnosis.png"), dpi=150)
    plt.close(fig)


def fig12_sweep_heatmap(sweep):
    mcs_vals = sorted(sweep["min_cluster_size"].unique())
    ms_vals = sorted(sweep["min_samples"].unique())
    n_neighbors_vals = sorted(sweep["n_neighbors"].unique())
    min_dist_vals = sorted(sweep["min_dist"].unique())

    fig, axes = plt.subplots(len(mcs_vals), len(ms_vals), figsize=(4 * len(ms_vals), 4 * len(mcs_vals)),
                              squeeze=False)
    vmin, vmax = sweep["silhouette"].min(), sweep["silhouette"].max()
    im = None
    for i, mcs in enumerate(mcs_vals):
        for j, ms in enumerate(ms_vals):
            ax = axes[i][j]
            sub = sweep[(sweep["min_cluster_size"] == mcs) & (sweep["min_samples"] == ms)]
            grid = sub.pivot(index="n_neighbors", columns="min_dist", values="silhouette") \
                      .reindex(index=n_neighbors_vals, columns=min_dist_vals)
            im = ax.imshow(grid.values, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(min_dist_vals)))
            ax.set_xticklabels(min_dist_vals, fontsize=8)
            ax.set_yticks(range(len(n_neighbors_vals)))
            ax.set_yticklabels(n_neighbors_vals, fontsize=8)
            ax.set_title(f"min_cluster_size={mcs}, min_samples={ms}", fontsize=9, color=INK)
            if i == len(mcs_vals) - 1:
                ax.set_xlabel("min_dist")
            if j == 0:
                ax.set_ylabel("n_neighbors")
            # highlight the winning cell
            if mcs == WINNING_CONFIG["min_cluster_size"] and ms == WINNING_CONFIG["min_samples"]:
                yi = n_neighbors_vals.index(WINNING_CONFIG["n_neighbors"])
                xi = min_dist_vals.index(WINNING_CONFIG["min_dist"])
                ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False,
                                            edgecolor=CAT[1], linewidth=3))

    fig.colorbar(im, ax=axes, label="silhouette score", shrink=0.6)
    fig.suptitle("Silhouette score across the UMAP+HDBSCAN hyperparameter sweep\n"
                 "(winning cell outlined)", fontsize=12, color=INK)
    fig.savefig(os.path.join(FIG_DIR, "12_sweep_heatmap.png"), dpi=150)
    plt.close(fig)


def fig13_bootstrap_ari_histogram(report):
    ari = np.array(report["bootstrap"]["ari_values"])
    mean_ari = report["bootstrap"]["mean_ari"]
    threshold = report["bootstrap"]["stability_threshold"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ari, bins=30, color=CAT[0], edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(threshold, color=CAT[6], linestyle="--", linewidth=2, label=f"stability threshold ({threshold})")
    ax.axvline(mean_ari, color=CAT[1], linestyle="-", linewidth=2, label=f"mean ARI ({mean_ari:.3f})")
    ax.set_xlabel("Adjusted Rand Index (bootstrap iteration vs full-data labels)")
    ax.set_ylabel("Count")
    ax.set_title("Lesion-level bootstrap stability — 100 iterations\n"
                 "(bimodal: distinct low- and high-agreement subsample regimes)", fontsize=11, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "13_bootstrap_ari_histogram.png"), dpi=150)
    plt.close(fig)


def fig14_cluster_representative_images(df, n_samples=8):
    rng = np.random.RandomState(SEED)  # reproducibility policy: fixed random_state
    clusters = sorted(c for c in df["cluster_label"].unique() if c != -1)  # skip noise (only 2 points)

    fig, axes = plt.subplots(len(clusters), n_samples, figsize=(2 * n_samples, 2 * len(clusters)))
    missing = 0
    for i, c in enumerate(clusters):
        members = df.loc[df["cluster_label"] == c, "isic_id"].values
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
    fig.suptitle("Representative dermoscopy images sampled per cluster", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "14_cluster_representative_images.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    meta = pd.read_csv(META_CSV, low_memory=False)
    df = load_umap_and_labels()
    sweep = pd.read_csv(SWEEP_CSV)
    with open(VALIDATION_JSON) as f:
        report = json.load(f)

    fig10_umap_by_cluster(df)
    fig11_umap_by_diagnosis(df, meta)
    fig12_sweep_heatmap(sweep)
    fig13_bootstrap_ari_histogram(report)
    fig14_cluster_representative_images(df)

    names = ["10_umap_by_cluster.png", "11_umap_by_diagnosis.png", "12_sweep_heatmap.png",
             "13_bootstrap_ari_histogram.png", "14_cluster_representative_images.png"]
    for name in names:
        p = os.path.join(FIG_DIR, name)
        assert os.path.getsize(p) > 10_000, f"{p} looks too small/corrupt"
    print("figures 10-14 written and sanity-checked")
