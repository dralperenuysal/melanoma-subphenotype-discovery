"""Clinical/dermoscopic correlation for the melanoma-only
patch-meanpool clustering (4 clusters: sizes 6,355/246/54/50 + 9 noise).

Same discipline as validation/clinical_correlation.py (mixed-cohort Step 5):
a test on a cluster with < MIN_N_PER_GROUP non-null values is flagged
underpowered rather than reported with a false-confidence p-value.

Note: ISIC metadata used here has no dermoscopic structure labels (atypical
network, blue-white veil, regression structures) — that analysis
is skipped, same as the mixed-cohort analysis.

Noise points (cluster_label == -1, 9 of them) are excluded from all group
comparisons: too few to form a meaningful group and not a real cluster.
"""
import json

import pandas as pd
from scipy.stats import chi2_contingency, kruskal

META_CSV = "data/processed/quality_filtered_metadata.csv"
LABELS_CSV = "clustering/melanoma_only_patchpool/final_cluster_labels.csv"
OUT_JSON = "validation/clinical_correlation_melanoma_only_results.json"
MIN_N_PER_GROUP = 10  # below this, a test is underpowered rather than meaningful


def load():
    meta = pd.read_csv(META_CSV, low_memory=False)
    labels = pd.read_csv(LABELS_CSV)
    df = meta.merge(labels, on="isic_id", how="inner")
    return df[df["cluster_label"] != -1].copy()


def kruskal_test(df, col):
    sub = df.dropna(subset=[col])
    counts = sub.groupby("cluster_label")[col].count().to_dict()
    groups = [g[col].values for _, g in sub.groupby("cluster_label") if len(g) > 0]
    n_groups_with_data = sum(1 for g in groups if len(g) >= MIN_N_PER_GROUP)
    result = {"variable": col, "test": "kruskal-wallis",
              "n_per_cluster": counts, "n_total": int(sub.shape[0])}
    if n_groups_with_data < 2:
        result["underpowered"] = True
        result["reason"] = (f"only {n_groups_with_data} cluster(s) have >= {MIN_N_PER_GROUP} "
                             f"non-null values for {col}, so a cross-cluster test is not meaningful")
        result["statistic"] = None
        result["p_value"] = None
    else:
        stat, p = kruskal(*groups)
        result["underpowered"] = False
        result["statistic"] = float(stat)
        result["p_value"] = float(p)
    return result


def chi2_test(df, col):
    sub = df.dropna(subset=[col])
    counts = sub.groupby("cluster_label")[col].count().to_dict()
    table = pd.crosstab(sub["cluster_label"], sub[col])
    n_groups_with_data = (table.sum(axis=1) >= MIN_N_PER_GROUP).sum()
    result = {"variable": col, "test": "chi-square",
              "n_per_cluster": counts, "n_total": int(sub.shape[0])}
    if n_groups_with_data < 2 or table.shape[0] < 2 or table.shape[1] < 2:
        result["underpowered"] = True
        result["reason"] = (f"only {n_groups_with_data} cluster(s) have >= {MIN_N_PER_GROUP} "
                             f"non-null values for {col}, so a cross-cluster test is not meaningful")
        result["statistic"] = None
        result["p_value"] = None
    else:
        stat, p, _, _ = chi2_contingency(table)
        result["underpowered"] = False
        result["statistic"] = float(stat)
        result["p_value"] = float(p)
    return result


def crosstab(df, col):
    table = pd.crosstab(df["cluster_label"], df[col])
    row_pct = (table.div(table.sum(axis=1), axis=0) * 100).round(2)
    return {"counts": table.to_dict(orient="index"),
            "row_pct": row_pct.to_dict(orient="index")}


if __name__ == "__main__":
    df = load()
    results = {
        "n_images": int(df.shape[0]),
        "noise_excluded": 9,
        "skipped": ("dermoscopic structure labels (atypical network, blue-white veil, "
                    "regression structures) not present in ISIC metadata used by this "
                    "project — sub-bullet skipped, same as mixed-cohort Step 5"),
        "thickness_test": kruskal_test(df, "mel_thick_mm"),
        "ulceration_test": chi2_test(df, "mel_ulcer"),
        "diagnosis3_crosstab": crosstab(df, "diagnosis_3"),
        # exploratory, not mandated by the analysis plan but well-filled in this subset
        "anatom_site_test": chi2_test(df, "anatom_site_1"),
        "sex_test": chi2_test(df, "sex"),
        "age_test": kruskal_test(df, "age_approx"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items()
                       if k not in ("diagnosis3_crosstab",)}, indent=2))
    print("\ndiagnosis_3 row %% per cluster:")
    print(pd.DataFrame(results["diagnosis3_crosstab"]["row_pct"]).T)
    print(f"\nwrote {OUT_JSON}")

    # ponytail: smallest runnable check — thickness/ulceration stay underpowered
    # (known data sparsity: mel_thick_mm/mel_ulcer concentrated in cluster 1),
    # while the well-filled exploratory variables are NOT underpowered.
    assert results["thickness_test"]["underpowered"] is True
    assert results["ulceration_test"]["underpowered"] is True
    assert results["anatom_site_test"]["underpowered"] is False
    assert results["sex_test"]["underpowered"] is False
    assert results["age_test"]["underpowered"] is False
    print("self-check passed: underpowered flags match known data sparsity")
