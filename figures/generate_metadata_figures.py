"""Generate the 7 exploratory metadata/filtering figures for the melanoma
subphenotype discovery project. Uses metadata CSVs only (no embeddings).

Palette: validated default — fixed
categorical hue order, one-hue sequential ramp for magnitude, status colors
reserved for pass/fail flags. Static PNGs for a manuscript, light mode only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib_venn import venn2

# --- palette (dataviz skill reference instance, light mode) ---
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

FIG_DIR = "figures"
DATA = "data"


def fig1_filtering_funnel():
    stages = [
        ("Total ISIC\ndermoscopic", 124811),
        ("Histopathology\nconfirmed", 42068),
        ("Image file\navailable", 37952),
        ("Resolution\n≥224px", 37948),
        ("Passed artifact\nheuristic (final)", 28705),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    y = range(len(stages))
    ax.barh(y, counts, color=SEQ_BLUE[3], height=0.6)
    for i, (label, count) in enumerate(stages):
        ax.text(count + 1500, i, f"{count:,}", va="center", ha="left", color=INK, fontsize=10)
        if i > 0:
            excluded = stages[i - 1][1] - count
            ax.text(count / 2, i, f"−{excluded:,}", va="center", ha="center",
                     color="white", fontsize=9, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of images")
    ax.set_title("Inclusion/exclusion filtering funnel")
    ax.set_xlim(0, 140000)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_filtering_funnel.png", dpi=150)
    plt.close(fig)


def fig2_metadata_fillrate():
    fields = [
        ("Breslow thickness\n(mel_thick_mm)", 0.6, False),
        ("Ulceration\n(mel_ulcer)", 0.2, False),
        ("Anatomic location\n(anatom_site_1)", 75.7, True),
        ("Histopath. diagnosis\n(diagnosis_1)", 98.4, True),
        ("Diagnosis confirm type\n(diagnosis_confirm_type)", 86.0, True),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f[0] for f in fields]
    values = [f[1] for f in fields]
    colors = [GOOD if f[2] else CRITICAL for f in fields]
    y = range(len(fields))
    ax.barh(y, values, color=colors, height=0.55)
    ax.axvline(50, color=MUTED, linestyle="--", linewidth=1)
    ax.text(50, len(fields) - 0.4, "50% inclusion\nthreshold", color=MUTED, fontsize=8,
            ha="center", va="bottom")
    for i, v in enumerate(values):
        ax.text(v + 1.5, i, f"{v:.1f}%", va="center", color=INK, fontsize=10)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Fill rate (% of 124,811 rows)")
    ax.set_title("Metadata field completeness (full ISIC dermoscopic set)")
    handles = [mpatches.Patch(color=GOOD, label="Retained (≥50%)"),
               mpatches.Patch(color=CRITICAL, label="Dropped (<50%)")]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_metadata_fillrate.png", dpi=150)
    plt.close(fig)


def fig3_diagnosis_distribution(df):
    counts = df["diagnosis_1"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(counts.index, counts.values, color=CAT[:len(counts)], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}", ha="center", color=INK, fontsize=10)
    ax.set_ylabel("Number of images")
    ax.set_title("Diagnosis distribution (final cohort, n=28,705)")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_diagnosis_distribution.png", dpi=150)
    plt.close(fig)


def fig4_anatomic_site_distribution(df):
    counts = df["anatom_site_1"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(counts.index, counts.values, color=CAT[0], width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 150, f"{v:,}", ha="center", color=INK, fontsize=9)
    ax.set_ylabel("Number of images")
    n_missing = df["anatom_site_1"].isna().sum()
    ax.set_title(f"Anatomic site distribution (final cohort, {n_missing:,} missing)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_anatomic_site_distribution.png", dpi=150)
    plt.close(fig)


def fig5_age_sex_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for sex, color, label in [("male", CAT[0], "Male"), ("female", CAT[1], "Female")]:
        ages = df.loc[df["sex"] == sex, "age_approx"].dropna()
        ax.hist(ages, bins=18, range=(0, 90), alpha=0.6, color=color, label=label)
    ax.set_xlabel("Approximate age")
    ax.set_ylabel("Number of images")
    n_missing = df["sex"].isna().sum() + df["age_approx"].isna().sum()
    ax.set_title(f"Age distribution by sex (final cohort, {n_missing:,} missing values excluded)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_age_sex_distribution.png", dpi=150)
    plt.close(fig)


def fig6_images_per_lesion(df):
    per_lesion = df["lesion_id"].dropna().value_counts()
    n_missing = df["lesion_id"].isna().sum()
    fig, ax = plt.subplots(figsize=(7, 5))
    max_n = per_lesion.max()
    bins = range(1, max_n + 2)
    ax.hist(per_lesion.values, bins=bins, color=CAT[2], align="left")
    ax.set_xlabel("Images per lesion")
    ax.set_ylabel("Number of lesions")
    n_multi = (per_lesion > 1).sum()
    ax.set_title(
        f"Images per lesion (final cohort)\n{per_lesion.shape[0]:,} lesions with a known "
        f"lesion_id, {n_multi:,} have >1 image\n{n_missing:,} rows have no lesion_id",
        fontsize=11,
    )
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_images_per_lesion.png", dpi=150)
    plt.close(fig)


def fig7_ham10000_isic_overlap(df, ham_ids, isic_all_ids):
    fig, ax = plt.subplots(figsize=(8, 7.5))
    v = venn2(
        [isic_all_ids, ham_ids],
        set_labels=("ISIC archive\n(124,811)", "HAM10000\n(11,720)"),
        set_colors=(CAT[0], CAT[1]),
        alpha=0.6,
        ax=ax,
    )
    for text in v.set_labels:
        if text:
            text.set_color(INK)
    for text in v.subset_labels:
        if text:
            text.set_color("white")
            text.set_fontweight("bold")
    final_ham_overlap = df["isic_id"].isin(ham_ids).sum()
    fig.suptitle("HAM10000 is (almost) fully contained in the ISIC archive", fontsize=13, y=0.98)
    ax.set_title(
        f"Final cohort (n=28,705) includes {final_ham_overlap:,} HAM10000-labeled images\n"
        "— not double-counted, single row per image",
        fontsize=10, color=INK2,
    )
    # venn2 locks equal aspect to the data range, so any attempt to widen xlim/the box
    # gets silently reverted and the tiny HAM10000-only sliver's count label (which sits
    # right at the small circle's edge) clips off the canvas. Just nudge that one text
    # object left instead of fighting the aspect lock.
    ham_only_label = v.subset_labels[1]  # subset order is (A-only, B-only, A∩B)
    if ham_only_label:
        x, y = ham_only_label.get_position()
        ham_only_label.set_position((x - 0.08, y))
        ham_only_label.set_ha("right")
    fig.subplots_adjust(top=0.82)
    fig.savefig(f"{FIG_DIR}/07_ham10000_isic_overlap.png", dpi=150)
    plt.close(fig)


def run():
    df = pd.read_csv(f"{DATA}/processed/quality_filtered_metadata.csv", low_memory=False)
    ham = pd.read_csv(f"{DATA}/metadata/ham10000_metadata.csv", low_memory=False)
    isic_all = pd.read_csv(f"{DATA}/metadata/isic_all_dermoscopic.csv", low_memory=False)

    fig1_filtering_funnel()
    fig2_metadata_fillrate()
    fig3_diagnosis_distribution(df)
    fig4_anatomic_site_distribution(df)
    fig5_age_sex_distribution(df)
    fig6_images_per_lesion(df)
    fig7_ham10000_isic_overlap(df, set(ham["isic_id"]), set(isic_all["isic_id"]))

    print("Generated 7 figures in figures/")


if __name__ == "__main__":
    run()
    import os
    expected = [f"{FIG_DIR}/{n:02d}_{s}.png" for n, s in enumerate(
        ["filtering_funnel", "metadata_fillrate", "diagnosis_distribution",
         "anatomic_site_distribution", "age_sex_distribution",
         "images_per_lesion", "ham10000_isic_overlap"], start=1)]
    for path in expected:
        assert os.path.exists(path), f"missing {path}"
        assert os.path.getsize(path) > 10_000, f"suspiciously small {path}"
    print("sanity checks passed: all 7 figures exist and are non-trivial size")
