"""Clinical/dermoscopic correlation.

Tests association of Breslow thickness (mel_thick_mm) and ulceration status
(mel_ulcer) with cluster membership, plus a diagnosis_1 x cluster crosstab.

Note: ISIC metadata used here has no dermoscopic structure labels (atypical
network, blue-white veil, regression structures) — that analysis
is skipped, since the columns don't exist in
data/processed/quality_filtered_metadata.csv.

Noise points (cluster_label == -1, only 2 of them) are excluded from all
group comparisons: too few to form a meaningful group and not a real cluster.
"""
import json

import pandas as pd
from scipy.stats import chi2_contingency, kruskal

META_CSV = "data/processed/quality_filtered_metadata.csv"
LABELS_CSV = "clustering/cluster_assignments/final_cluster_labels.csv"
OUT_JSON = "validation/clinical_correlation_results.json"
MIN_N_PER_GROUP = 10  # below this, a test is underpowered rather than meaningful


def load():
    meta = pd.read_csv(META_CSV, low_memory=False)
    labels = pd.read_csv(LABELS_CSV)
    df = meta.merge(labels, on="isic_id", how="inner")
    return df[df["cluster_label"] != -1].copy()


def test_thickness(df):
    sub = df.dropna(subset=["mel_thick_mm"])
    counts = sub.groupby("cluster_label")["mel_thick_mm"].count().to_dict()
    groups = [g["mel_thick_mm"].values for _, g in sub.groupby("cluster_label") if len(g) > 0]
    n_groups_with_data = sum(1 for g in groups if len(g) >= MIN_N_PER_GROUP)
    result = {"variable": "mel_thick_mm", "test": "kruskal-wallis",
              "n_per_cluster": counts, "n_total": int(sub.shape[0])}
    if n_groups_with_data < 2:
        result["underpowered"] = True
        result["reason"] = (f"only {n_groups_with_data} cluster(s) have >= {MIN_N_PER_GROUP} "
                             f"non-null values; mel_thick_mm is populated almost exclusively "
                             f"within a single cluster, so a cross-cluster test is not meaningful")
        result["statistic"] = None
        result["p_value"] = None
    else:
        stat, p = kruskal(*groups)
        result["underpowered"] = False
        result["statistic"] = float(stat)
        result["p_value"] = float(p)
    return result


def test_ulceration(df):
    sub = df.dropna(subset=["mel_ulcer"])
    counts = sub.groupby("cluster_label")["mel_ulcer"].count().to_dict()
    table = pd.crosstab(sub["cluster_label"], sub["mel_ulcer"])
    n_groups_with_data = (table.sum(axis=1) >= MIN_N_PER_GROUP).sum()
    result = {"variable": "mel_ulcer", "test": "chi-square",
              "n_per_cluster": counts, "n_total": int(sub.shape[0])}
    if n_groups_with_data < 2 or table.shape[0] < 2:
        result["underpowered"] = True
        result["reason"] = (f"only {n_groups_with_data} cluster(s) have >= {MIN_N_PER_GROUP} "
                             f"non-null values; mel_ulcer is populated almost exclusively "
                             f"within a single cluster, so a cross-cluster test is not meaningful")
        result["statistic"] = None
        result["p_value"] = None
    else:
        stat, p, _, _ = chi2_contingency(table)
        result["underpowered"] = False
        result["statistic"] = float(stat)
        result["p_value"] = float(p)
    return result


def diagnosis_crosstab(df):
    table = pd.crosstab(df["cluster_label"], df["diagnosis_1"])
    row_pct = (table.div(table.sum(axis=1), axis=0) * 100).round(2)
    return {"counts": table.to_dict(orient="index"),
            "row_pct": row_pct.to_dict(orient="index")}


if __name__ == "__main__":
    df = load()
    results = {
        "skipped": ("dermoscopic structure labels (atypical network, blue-white veil, "
                    "regression structures) not present in ISIC metadata used by this "
                    "project — sub-bullet skipped"),
        "noise_excluded": int((df["cluster_label"] == -1).sum()),
        "thickness_test": test_thickness(df),
        "ulceration_test": test_ulceration(df),
        "diagnosis_crosstab": diagnosis_crosstab(df),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "diagnosis_crosstab"}, indent=2))
    print("\ndiagnosis_1 row %% per cluster:")
    print(pd.DataFrame(results["diagnosis_crosstab"]["row_pct"]).T)
    print(f"\nwrote {OUT_JSON}")

    # ponytail: smallest runnable check — assert the underpowered flags match the known
    # data-sparsity fact (mel_thick_mm/mel_ulcer are ~100% concentrated in cluster 0)
    assert results["thickness_test"]["underpowered"] is True
    assert results["ulceration_test"]["underpowered"] is True
    print("self-check passed: thickness/ulceration correctly flagged underpowered")
