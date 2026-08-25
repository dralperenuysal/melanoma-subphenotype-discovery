"""Remediation A: run full protocol (GATE0->GATE5) on the
balanced-subsampled melanoma-only cohort, for raw and cropped DINOv2 CLS variants.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score
import harmonypy
from neuroHarmonize import harmonizationLearn
import hdbscan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT = json.load(open(os.path.join(ROOT, "harmonization", "remediation_a_result.json")))
KEEP_IDS = set(RESULT["image_ids"])


def cramers_v(meta):
    tab = pd.crosstab(meta["dataset"], meta["diagnosis"])
    chi2 = chi2_contingency(tab)[0]
    n = tab.values.sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def batch_auc(Z, labels):
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, Z, labels, cv=cv, scoring="accuracy").mean())


def mixing_score(Z, labels, k=30):
    labels = pd.factorize(labels)[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    same = (labels[idx] == labels[:, None]).mean()
    expected = (pd.Series(labels).value_counts(normalize=True) ** 2).sum()
    return float((1 - same) / (1 - expected))


def bio_conservation(Z, dx):
    m = pd.notna(dx)
    return float(silhouette_score(Z[m.values], dx[m]))


def run_variant(variant):
    out_dir = os.path.join(ROOT, "results", "harmonization_balanced", variant)
    os.makedirs(out_dir, exist_ok=True)

    meta_full = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}.csv"))
    X_full = np.load(os.path.join(ROOT, "data", "embeddings", f"dinov2_embeddings_{variant}.npy"))
    mask = meta_full["image_id"].isin(KEEP_IDS).values
    meta = meta_full[mask].reset_index(drop=True)
    X_raw = X_full[mask].astype(np.float32)
    assert len(meta) == len(X_raw) == len(KEEP_IDS)

    # GATE 0 (recheck on subsample)
    ct = pd.crosstab(meta["dataset"], meta["diagnosis"], normalize="index")
    ct.to_csv(os.path.join(out_dir, "00_confounding.csv"))
    v = cramers_v(meta)
    gate0 = "STOP" if v >= 0.35 else ("PROCEED_WITH_CAVEAT" if v >= 0.20 else "PROCEED")
    print(f"[{variant}] GATE0 cramers_v={v:.3f} -> {gate0}")
    if gate0 == "STOP":
        json.dump({"gate0": gate0, "cramers_v": v}, open(os.path.join(out_dir, "gate0_result.json"), "w"), indent=2)
        return {"variant": variant, "stopped_at": "GATE0", "cramers_v": v}

    # GATE 1
    Z_raw = PCA(n_components=50, random_state=0).fit_transform(StandardScaler().fit_transform(X_raw))
    acc_raw = batch_auc(Z_raw, meta["dataset"])
    chance = 1.0 / meta["dataset"].nunique()
    json.dump({"batch_acc_raw": acc_raw, "chance": chance}, open(os.path.join(out_dir, "01_batch_signal.json"), "w"), indent=2)
    print(f"[{variant}] GATE1 batch_acc={acc_raw:.3f} chance={chance:.3f}")
    if acc_raw < chance + 0.10:
        np.save(os.path.join(out_dir, "02_harmony.npy"), Z_raw)
        result = {"variant": variant, "stopped_at": "GATE1_negligible_batch_effect",
                  "cramers_v": v, "batch_acc_raw": acc_raw, "chance": chance}
        json.dump(result, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
        return result

    # Adim 3: Harmony (theta=1.0 fixed per doc; doc's own GATE5 fallback allows
    # retrying theta=0.5 if criterion 3 fails, done below after first GATE5 pass)
    def run_harmony_theta(theta):
        ho = harmonypy.run_harmony(Z_raw, meta, vars_use=["dataset"], theta=theta, max_iter_harmony=20, random_state=0)
        Z = np.asarray(ho.Z_corr)
        if Z.shape[0] != len(meta):
            Z = Z.T
        return np.ascontiguousarray(Z)

    Z_h = run_harmony_theta(1.0)
    np.save(os.path.join(out_dir, "02_harmony.npy"), Z_h)

    # Adim 4: ComBat on RAW D-dim (never diagnosis as covariate)
    covars = meta[["dataset"]].copy().rename(columns={"dataset": "SITE"})
    _, X_cb = harmonizationLearn(X_raw.astype(np.float64), covars)
    Z_cb = PCA(n_components=50, random_state=0).fit_transform(StandardScaler().fit_transform(X_cb))
    np.save(os.path.join(out_dir, "03_combat.npy"), Z_cb)

    # GATE 5
    metrics = {}
    for name, Z in [("raw", Z_raw), ("harmony", Z_h), ("combat", Z_cb)]:
        metrics[name] = {
            "batch_acc": batch_auc(Z, meta["dataset"]),
            "mixing": mixing_score(Z, meta["dataset"]),
            "bio_silhouette": bio_conservation(Z, meta["diagnosis"]),
        }
    json.dump(metrics, open(os.path.join(out_dir, "04_metrics.json"), "w"), indent=2)
    print(f"[{variant}] GATE5 metrics: {json.dumps(metrics, indent=2)}")

    def gate5_check(m):
        return (m["harmony"]["batch_acc"] < chance + 0.10 and
                m["harmony"]["mixing"] > m["raw"]["mixing"] and
                m["harmony"]["bio_silhouette"] >= 0.85 * m["raw"]["bio_silhouette"])

    gate5_pass = gate5_check(metrics)
    theta_used = 1.0
    if not gate5_pass:
        # doc's explicit fallback: retry at theta=0.5 before giving up
        Z_h2 = run_harmony_theta(0.5)
        metrics2 = dict(metrics)
        metrics2["harmony"] = {
            "batch_acc": batch_auc(Z_h2, meta["dataset"]),
            "mixing": mixing_score(Z_h2, meta["dataset"]),
            "bio_silhouette": bio_conservation(Z_h2, meta["diagnosis"]),
        }
        if gate5_check(metrics2):
            Z_h, metrics, gate5_pass, theta_used = Z_h2, metrics2, True, 0.5
            np.save(os.path.join(out_dir, "02_harmony.npy"), Z_h)
        else:
            metrics["harmony_theta_0.5_retry"] = metrics2["harmony"]
        json.dump(metrics, open(os.path.join(out_dir, "04_metrics.json"), "w"), indent=2)

    # Method-independence: HDBSCAN on Z_h vs Z_cb, ARI
    def cluster(Z):
        return hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(Z)
    labels_h = cluster(Z_h)
    labels_cb = cluster(Z_cb)
    ari_hc = float(adjusted_rand_score(labels_h, labels_cb))

    result = {
        "variant": variant, "cramers_v": v, "gate0": gate0,
        "batch_acc_raw": acc_raw, "chance": chance,
        "gate5_metrics": metrics, "gate5_pass": bool(gate5_pass), "theta_used": theta_used,
        "harmony_vs_combat_ari": ari_hc,
        "n_cohort": len(meta),
    }
    json.dump(result, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    return result


if __name__ == "__main__":
    results = {}
    for variant in ["raw", "cropped"]:
        results[variant] = run_variant(variant)
    json.dump(results, open(os.path.join(ROOT, "harmonization", "remediation_a_final_results.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))
