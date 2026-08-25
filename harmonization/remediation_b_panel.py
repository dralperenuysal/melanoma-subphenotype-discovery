"""2x3 UMAP panel (rows=dataset/diagnosis, cols=raw/harmony/combat)
for the remediation-B (excluded-source) cohort.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                      "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# colorblind-safe-ish categorical set, fixed order, capped -- folds overflow into "Other"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def cap_categories(series, max_n=10):
    top = series.value_counts().index[:max_n]
    return series.where(series.isin(top), "Other")


def run(variant):
    out_dir = os.path.join(ROOT, "results", "harmonization_excluded", variant)
    meta = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}_excluded.csv"))
    mats = {
        "raw": np.load(os.path.join(out_dir, "Z_raw_socorrected.npy")),
        "harmony": np.load(os.path.join(out_dir, "02_harmony.npy")),
        "combat": np.load(os.path.join(out_dir, "03_combat.npy")),
    }
    ds_cap = cap_categories(meta["dataset"], max_n=10)
    dx = meta["diagnosis"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, (name, Z) in enumerate(mats.items()):
        emb = UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Z)
        for row, (label_series, title) in enumerate([(ds_cap, "dataset"), (dx, "diagnosis")]):
            ax = axes[row, col]
            cats = sorted(label_series.unique())
            for i, c in enumerate(cats):
                m = (label_series == c).values
                ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6,
                           color=PALETTE[i % len(PALETTE)], label=str(c))
            ax.set_title(f"{name} | colored by {title}")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 2:
                ax.legend(fontsize=6, markerscale=2, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(f"Remediation B (excluded source) -- {variant} cohort")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_umap_panel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{variant}: saved {out_dir}/05_umap_panel.png")


if __name__ == "__main__":
    for v in ["raw", "cropped"]:
        run(v)
