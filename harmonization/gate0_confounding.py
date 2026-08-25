"""GATE 0: dataset x diagnosis confounding check (Cramer's V)."""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(variant):
    out_dir = os.path.join(ROOT, "results", "harmonization", variant)
    os.makedirs(out_dir, exist_ok=True)
    meta = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}.csv"))

    ct = pd.crosstab(meta["dataset"], meta["diagnosis"], normalize="index")
    ct.to_csv(os.path.join(out_dir, "00_confounding.csv"))

    tab = pd.crosstab(meta["dataset"], meta["diagnosis"])
    chi2 = chi2_contingency(tab)[0]
    n = tab.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))

    small_batches = meta["dataset"].value_counts()
    small_batches = small_batches[small_batches < 50].to_dict()

    if cramers_v >= 0.35:
        gate = "STOP"
    elif cramers_v >= 0.20:
        gate = "PROCEED_WITH_CAVEAT"
    else:
        gate = "PROCEED"

    result = {"variant": variant, "cramers_v": cramers_v, "gate0": gate,
              "n_datasets": int(meta["dataset"].nunique()),
              "small_batches_n_lt_50": small_batches}
    json.dump(result, open(os.path.join(out_dir, "gate0_result.json"), "w"), indent=2)
    print(variant, result)
    return result


if __name__ == "__main__":
    for v in ["raw", "cropped"]:
        run(v)
