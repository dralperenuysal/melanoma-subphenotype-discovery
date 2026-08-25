"""Harmony primary, ComBat sensitivity, GATE 5 metrics
applied to the remediation-B (excluded-source) cohort. `diagnosis` is never a
covariate for either method (SS9 Yasaklar).
"""
import os
import json
import numpy as np
import pandas as pd
import harmonypy
from neuroHarmonize import harmonizationLearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def run(variant, theta=1.0):
    out_dir = os.path.join(ROOT, "results", "harmonization_excluded", variant)
    meta = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}_excluded.csv"))
    # use the Shades-of-Gray color-corrected embeddings (SS4) as the current-best input --
    # color constancy barely moved GATE 1 (0.86->0.86 raw, 0.79->0.78 cropped), confirming
    # the confound is structural (vignette/ruler/framing), not illumination -- but it's still
    # the more-corrected input per protocol order, so it's what Harmony/ComBat run on.
    X_raw = np.load(os.path.join(ROOT, "data", "embeddings", f"dinov2_embeddings_{variant}_excluded_socorrected.npy")).astype(np.float32)
    Z_raw = np.load(os.path.join(out_dir, "Z_raw_socorrected.npy"))
    assert len(X_raw) == len(meta) == len(Z_raw)

    # SS5 Harmony (primary)
    ho = harmonypy.run_harmony(Z_raw, meta, vars_use=["dataset"], theta=theta,
                                max_iter_harmony=20, random_state=0)
    # harmonypy 2.0.0's Z_corr is already (N, D); the protocol was written against an
    # older version where it was (D, N) needing .T -- transpose only if shapes require it.
    Z_h = np.asarray(ho.Z_corr)
    if Z_h.shape[0] != Z_raw.shape[0]:
        Z_h = Z_h.T
    Z_h = np.ascontiguousarray(Z_h)
    np.save(os.path.join(out_dir, "02_harmony.npy"), Z_h)

    # SS6 ComBat (sensitivity only) -- raw D-dim first, PCA after, diagnosis NEVER a covariate
    covar_cols = ["dataset"] + [c for c in ["age", "sex", "anatom_site"] if c in meta.columns]
    covars = meta[covar_cols].copy().rename(columns={"dataset": "SITE"})
    dummy_cols = [c for c in ["sex", "anatom_site"] if c in covars.columns]
    covars = pd.get_dummies(covars, columns=dummy_cols, drop_first=True)
    covars = covars.fillna(covars.median(numeric_only=True))
    _, X_cb = harmonizationLearn(X_raw.astype(np.float64), covars)
    Z_cb = PCA(n_components=50, random_state=0).fit_transform(StandardScaler().fit_transform(X_cb))
    np.save(os.path.join(out_dir, "03_combat.npy"), Z_cb)

    # SS7 GATE 5 metrics
    metrics = {}
    for name, Z in [("raw", Z_raw), ("harmony", Z_h), ("combat", Z_cb)]:
        metrics[name] = {
            "batch_acc": batch_auc(Z, meta["dataset"]),
            "mixing": mixing_score(Z, meta["dataset"]),
            "bio_silhouette": bio_conservation(Z, meta["diagnosis"]),
        }
    json.dump(metrics, open(os.path.join(out_dir, "04_metrics.json"), "w"), indent=2)

    chance = 1.0 / meta["dataset"].nunique()
    g1 = metrics["harmony"]["batch_acc"] < chance + 0.10
    g2 = metrics["harmony"]["mixing"] > metrics["raw"]["mixing"]
    g3 = metrics["harmony"]["bio_silhouette"] >= 0.85 * metrics["raw"]["bio_silhouette"]
    gate5_pass = bool(g1 and g2 and g3)

    print(variant, "theta=", theta, json.dumps(metrics, indent=2),
          "GATE5:", {"batch_dropped": g1, "mixing_improved": g2, "bio_preserved": g3, "pass": gate5_pass})
    return metrics, gate5_pass


if __name__ == "__main__":
    for v in ["raw", "cropped"]:
        run(v)
