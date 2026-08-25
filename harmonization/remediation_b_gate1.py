"""GATE 1: batch-signal check on the excluded (remediation B) cohort."""
import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def batch_auc(Z, labels):
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(clf, Z, labels, cv=cv, scoring="accuracy").mean())


def run(variant):
    out_dir = os.path.join(ROOT, "results", "harmonization_excluded", variant)
    meta = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}_excluded.csv"))
    X = np.load(os.path.join(ROOT, "data", "embeddings", f"dinov2_embeddings_{variant}_excluded.npy")).astype(np.float32)
    assert len(X) == len(meta)

    Z_raw = PCA(n_components=50, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    np.save(os.path.join(out_dir, "Z_raw.npy"), Z_raw)

    acc_raw = batch_auc(Z_raw, meta["dataset"])
    chance = 1.0 / meta["dataset"].nunique()
    gate1 = "PROCEED" if acc_raw >= chance + 0.10 else "SKIP_HARMONIZATION"

    result = {"variant": variant, "batch_acc_raw": acc_raw, "chance": chance, "gate1": gate1}
    json.dump(result, open(os.path.join(out_dir, "01_batch_signal.json"), "w"), indent=2)
    print(variant, result)
    return result


if __name__ == "__main__":
    for v in ["raw", "cropped"]:
        run(v)
