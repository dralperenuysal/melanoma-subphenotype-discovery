"""Step 5 figure: diagnosis_1 proportion per cluster (stacked bar).
Style/palette reused from figures/generate_clustering_figures.py.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

RESULTS_JSON = "validation/clinical_correlation_results.json"
OUT_PNG = "figures/15_cluster_clinical_correlation.png"

if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    row_pct = pd.DataFrame(results["diagnosis_crosstab"]["row_pct"]).T
    row_pct.index = [f"Cluster {c}" for c in row_pct.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = pd.Series(0.0, index=row_pct.index)
    for i, diag in enumerate(row_pct.columns):
        ax.bar(row_pct.index, row_pct[diag], bottom=bottom, color=CAT[i % len(CAT)], label=diag, width=0.45)
        bottom += row_pct[diag]
    ax.set_ylabel("% of cluster")
    ax.set_title("diagnosis_1 composition per cluster", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), ncol=len(row_pct.columns))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    assert os.path.getsize(OUT_PNG) > 5_000
    print(f"wrote {OUT_PNG}")
