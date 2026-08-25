"""GATE 0 remediation A: balanced/stratified subsampling.

Caps each dataset's sample count so no single (very large, single-diagnosis) source
dominates the cohort and thereby drives dataset<->diagnosis association (Cramer's V).
Iterates over decreasing per-dataset caps until Cramer's V < 0.35 (stop threshold),
preferring the largest cap (= largest resulting cohort) that passes.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_STATE = 42
CAPS_TO_TRY = [3000, 2000, 1500, 1000, 750, 500, 350, 250, 150, 100, 75, 50]
MIN_USABLE_N = 1000


def cramers_v(meta):
    tab = pd.crosstab(meta["dataset"], meta["diagnosis"])
    chi2 = chi2_contingency(tab)[0]
    n = tab.values.sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


def dataset_purity(meta):
    """Fraction of each dataset's samples in its single most common diagnosis --
    uniform downsampling doesn't change Cramer's V (scale-invariant), so the
    actual lever is capping high-purity (near-single-diagnosis) datasets hard
    while leaving already-mixed datasets alone."""
    ct = pd.crosstab(meta["dataset"], meta["diagnosis"])
    return (ct.max(axis=1) / ct.sum(axis=1)).to_dict()


def subsample_purity_targeted(meta, pure_cap, mid_cap, rng):
    purity = dataset_purity(meta)
    keep_idx = []
    for ds, grp in meta.groupby("dataset"):
        p = purity[ds]
        if p >= 0.95:
            cap = pure_cap
        elif p >= 0.70:
            cap = mid_cap
        else:
            cap = len(grp)  # already mixed, don't touch
        n_take = min(len(grp), cap)
        keep_idx.extend(rng.choice(grp.index.values, size=n_take, replace=False))
    return meta.loc[sorted(keep_idx)]


def find_best_cap(meta):
    rng = np.random.RandomState(RANDOM_STATE)
    # (pure_cap, mid_cap) pairs, most permissive first
    schemes = [(200, 400), (100, 300), (50, 200), (30, 150), (20, 100), (10, 75), (5, 50)]
    for pure_cap, mid_cap in schemes:
        sub = subsample_purity_targeted(meta, pure_cap, mid_cap, np.random.RandomState(RANDOM_STATE))
        if len(sub) < MIN_USABLE_N:
            continue
        v = cramers_v(sub)
        print(f"  pure_cap={pure_cap}, mid_cap={mid_cap}: n={len(sub)}, cramers_v={v:.3f}")
        if v < 0.35:
            return (pure_cap, mid_cap), sub, v
    return None, None, None


if __name__ == "__main__":
    meta = pd.read_csv(os.path.join(ROOT, "data", "embeddings", "samples_raw.csv"))
    print(f"Full cohort: n={len(meta)}, cramers_v={cramers_v(meta):.3f}")
    cap, sub, v = find_best_cap(meta)
    if sub is None:
        print("FAILED: no purity-targeted scheme achieved V<0.35 at n>=", MIN_USABLE_N)
        json.dump({"status": "failed"}, open(os.path.join(ROOT, "harmonization", "remediation_a_result.json"), "w"), indent=2)
    else:
        print(f"Selected scheme (pure_cap,mid_cap)={cap}: n={len(sub)} (from {len(meta)}), cramers_v={v:.3f}")
        keep_ids = sub["image_id"].tolist()
        json.dump({"status": "ok", "cap": cap, "n": len(sub), "n_full": len(meta), "cramers_v": v,
                   "image_ids": keep_ids},
                  open(os.path.join(ROOT, "harmonization", "remediation_a_result.json"), "w"), indent=2)
