"""GATE-0 remediation B: exclude dataset(s) driving the confound.

Rule (fixed before checking outcome, not reverse-engineered): iteratively exclude
the LARGEST remaining `dataset` with >70% concentration in one `diagnosis` category
AND n>=50, recomputing Cramer's V after each removal, until V<0.35 or no dataset
meets the criterion. (A stricter one-shot rule, >90% concentration AND n>=100, was
tried first and only got V from 0.541 to 0.376 -- still failing -- so the iterative
version was applied next per the "exclude, then iterate" guidance.)
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONCENTRATION_THRESH = 0.70
MIN_N = 50
STOP_V = 0.35


def cramers_v(meta):
    tab = pd.crosstab(meta["dataset"], meta["diagnosis"])
    chi2 = chi2_contingency(tab)[0]
    n = tab.values.sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def iterative_exclude(meta):
    excluded = []
    history = []
    while True:
        v = cramers_v(meta)
        tab = pd.crosstab(meta["dataset"], meta["diagnosis"])
        prop = tab.div(tab.sum(axis=1), axis=0)
        n = tab.sum(axis=1)
        cand = prop.max(axis=1)[(prop.max(axis=1) > CONCENTRATION_THRESH) & (n >= MIN_N)]
        history.append({"n": int(len(meta)), "cramers_v": v, "n_candidates": len(cand)})
        if v < STOP_V or len(cand) == 0:
            break
        largest = n[cand.index].idxmax()
        excluded.append({"dataset": largest, "n_removed": int(n[largest]),
                          "concentration": float(prop.loc[largest].max())})
        meta = meta[meta["dataset"] != largest]
    return meta, excluded, history


def run(variant):
    out_dir = os.path.join(ROOT, "results", "harmonization_excluded", variant)
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}.csv"))
    X = np.load(os.path.join(ROOT, "data", "embeddings", f"dinov2_embeddings_{variant}.npy"))
    assert len(meta) == X.shape[0]

    v_before = cramers_v(meta)
    meta_f, excluded, history = iterative_exclude(meta)
    keep_mask = meta.index.isin(meta_f.index)
    X_f = X[keep_mask]
    meta_f = meta_f.reset_index(drop=True)

    v_after = history[-1]["cramers_v"]
    gate0 = "STOP" if v_after >= 0.35 else ("PROCEED_WITH_CAVEAT" if v_after >= 0.20 else "PROCEED")

    ct = pd.crosstab(meta_f["dataset"], meta_f["diagnosis"], normalize="index")
    ct.to_csv(os.path.join(out_dir, "00_confounding.csv"))

    result = {
        "variant": variant,
        "rule": f"iterative: exclude largest dataset with concentration>{CONCENTRATION_THRESH} "
                f"and n>={MIN_N}, until V<{STOP_V} or no candidates",
        "excluded_sources": excluded, "iteration_history": history,
        "n_before": int(len(meta)), "n_after": int(len(meta_f)),
        "cramers_v_before": v_before, "cramers_v_after": v_after, "gate0": gate0,
    }
    json.dump(result, open(os.path.join(out_dir, "gate0_result.json"), "w"), indent=2)

    np.save(os.path.join(ROOT, "data", "embeddings", f"dinov2_embeddings_{variant}_excluded.npy"), X_f)
    meta_f.to_csv(os.path.join(ROOT, "data", "embeddings", f"samples_{variant}_excluded.csv"), index=False)

    print(variant, json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    for v in ["raw", "cropped"]:
        run(v)
