"""Summary figure for the manuscript: bootstrap ARI across every clustering/confound-control
attempt in this project, colored by stability verdict. Static print figure (no interactivity
needed for a LaTeX manuscript figure)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (label, mean_ari, std_ari, status) — status in {"stable", "unstable", "artifact"}
# Order: mixed cohort first, then melanoma-only runs A–N in run order.
runs = [
    ("A: Mixed cohort (CLS)", 0.780, 0.204, "stable"),
    ("B: Mel-only CLS (grid)", 0.488, 0.314, "unstable"),
    ("C: Mel-only CLS (wide sweep)", 0.618, 0.228, "confounded"),
    ("D: Mel-only patch (pre-crop)", 0.865, 0.181, "confounded"),
    ("E: Mel-only patch (cropped)", 0.586, 0.278, "unstable"),
    ("F: Barcelona-only", 0.363, 0.195, "unstable"),
    ("G: Anonymous-only", 0.211, 0.317, "unstable"),
    ("H: DINOv2 + ComBat", 0.377, 0.126, "unstable"),
    ("I: Segmented (artifact)", 0.999, 0.002, "artifact"),
    ("J: Segmented, filtered", 0.347, 0.140, "unstable"),
    ("K: PanDerm (pre-crop)", 0.730, 0.178, "confounded"),
    ("L: PanDerm (cropped)", 0.251, 0.281, "unstable"),
    ("M: PanDerm + ComBat (raw)", 0.331, 0.199, "unstable"),
    ("N: PanDerm + ComBat (cropped)", 0.504, 0.174, "unstable"),
]

COLORS = {"stable": "#1B7A3D", "confounded": "#1B7A3D", "unstable": "#8A94A3", "artifact": "#C77D1F"}
HATCH = {"stable": None, "confounded": "//", "unstable": None, "artifact": None}
labels = [r[0] for r in runs][::-1]
means = [r[1] for r in runs][::-1]
stds = [r[2] for r in runs][::-1]
statuses = [r[3] for r in runs][::-1]

fig, ax = plt.subplots(figsize=(7.2, 7.0), dpi=200)
y = range(len(runs))
bar_colors = [COLORS[st] for st in statuses]
bars = ax.barh(y, means, xerr=stds, color=bar_colors, height=0.42, edgecolor="none",
               error_kw=dict(ecolor="#4a4a4a", elinewidth=1, capsize=2))
for b, st in zip(bars, statuses):
    if HATCH[st]:
        b.set_hatch(HATCH[st])
# Annotate the one bar whose apparent stability is an artifact (see paper text).
for yi, st, m in zip(y, statuses, means):
    if st == "artifact":
        ax.text(m + 0.055, yi, "artifact, not biology", va="center", ha="left",
                fontsize=7, style="italic", color="#4a4a4a")
ax.axvline(0.6, color="#1a1a1a", linestyle="--", linewidth=1.1, zorder=0)
# Placed in axes-fraction y (via get_xaxis_transform) so it sits just above the
# plot frame, clear of every bar regardless of how long the top bar is, and
# well below the title (which is pushed up further via `pad=` on set_title below).
ax.text(0.6, 1.02, "stability threshold (0.6)", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=7.5, color="#1a1a1a", clip_on=False)

ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel("Bootstrap ARI (mean ± s.d., 100 lesion-level 80% subsamples)", fontsize=9)
ax.set_xlim(-0.15, 1.45)
ax.set_ylim(-1.6, len(runs) - 0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="y", length=0)

ax.set_title("Bootstrap stability across all clustering variants attempted", fontsize=10.5, pad=46)

ax.legend(handles=[
    Patch(fc=COLORS["stable"], label="Stable (ARI \u2265 0.6)"),
    Patch(fc=COLORS["confounded"], hatch="//", label="Stable but confounded"),
    Patch(fc=COLORS["unstable"], label="Unstable"),
    Patch(fc=COLORS["artifact"], label="Segmentation artifact"),
], loc="lower right", fontsize=7, frameon=False, labelspacing=0.8)

fig.tight_layout()
fig.savefig("/home/alperen/data/PycharmProjects/melanoma-subphenotype-discovery/figures/22_ari_summary_all_runs.png",
            dpi=200, bbox_inches="tight")
print("saved figures/22_ari_summary_all_runs.png")
