"""Figure 16: Step 5 clinical correlation for the melanoma-only patch-meanpool
clustering (4 clusters). Stacked bar of diagnosis_3 (melanoma subtype)
composition per cluster — the most informative Step 5 result for this subset
(Breslow thickness / ulceration were underpowered, see
validation/clinical_correlation_melanoma_only_results.json).
Palette: same reference instance as figures/generate_clustering_figures.py.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

RESULTS_JSON = "validation/clinical_correlation_melanoma_only_results.json"
OUT_PNG = "figures/16_melanoma_only_clinical_correlation.png"

if __name__ == "__main__":
    with open(RESULTS_JSON) as f:
        results = json.load(f)

    row_pct = results["diagnosis3_crosstab"]["row_pct"]
    counts = results["diagnosis3_crosstab"]["counts"]
    clusters = sorted(row_pct.keys(), key=int)
    subtypes = ["Melanoma in situ", "Melanoma, NOS", "Melanoma Invasive"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(len(clusters))
    for i, subtype in enumerate(subtypes):
        vals = np.array([row_pct[c].get(subtype, 0.0) for c in clusters])
        ax.bar(range(len(clusters)), vals, bottom=bottom, label=subtype, color=CAT[i], width=0.6)
        bottom += vals

    n_per_cluster = [sum(counts[c].values()) for c in clusters]
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([f"Cluster {c}\n(n={n_per_cluster[i]})" for i, c in enumerate(clusters)])
    ax.set_ylabel("% of cluster (diagnosis_3)")
    ax.set_title("Melanoma subtype composition per cluster (patch-meanpool, melanoma-only)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")
