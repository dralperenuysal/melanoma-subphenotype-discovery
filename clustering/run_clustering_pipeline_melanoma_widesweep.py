"""Wider/finer hyperparameter sweep for the melanoma-only subset.
The original 54-combo grid (n_neighbors in {15,30,50},
min_dist in {0.0,0.1,0.25}, min_cluster_size in {15,25,50}, min_samples in {5,15})
was tuned for the 28,705-image mixed cohort and gave an UNSTABLE result on the
~6,714-image melanoma-only subset (bootstrap mean ARI=0.488, seed-sensitivity
ARI~0.03-0.11). This variant searches a wider/finer grid to check whether
instability is a hyperparameter artifact or persists regardless of config.

Grid: 7 n_neighbors x 5 min_dist x 7 min_cluster_size x 4 min_samples = 980 combos.
35 UMAP fits (n_neighbors x min_dist pairs) took ~52s for 9 fits in the original
54-combo run on this same N=6,714 subset with 20 local cores, so 35 fits is not a
scaling concern (a few batches of the 20-core pool). Only the WINNING config (by
the existing silhouette-based selection rule) gets the expensive bootstrap (100
lesion-level iterations) + 5-seed sensitivity validation, same as the other
variants -- running that on all 980 combos would be wasteful and unnecessary.

Run with `.venv_cluster/bin/python` (default env segfaults on `import umap`).
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from umap.umap_ import UMAP
import hdbscan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS_DIR = os.path.join(ROOT, "embeddings", "dinov2_cls")
METADATA_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")

OUT_DIR = os.path.join(ROOT, "clustering", "melanoma_only_widesweep")
SWEEP_CSV = os.path.join(OUT_DIR, "sweep_results.csv")
FINAL_LABELS_CSV = os.path.join(OUT_DIR, "final_cluster_labels.csv")
FINAL_UMAP_CSV = os.path.join(OUT_DIR, "final_umap_2d.csv")
VALIDATION_JSON = os.path.join(OUT_DIR, "step4_validation_report.json")

MELANOMA_DIAGNOSES = {"Melanoma, NOS", "Melanoma in situ", "Melanoma Invasive"}

# Wider/finer than the original grid: add small n_neighbors (small-N regime),
# finer min_dist coverage, and min_cluster_size up to ~1.5% of N=6,714.
UMAP_GRID = [(n, d) for n in (5, 10, 15, 20, 30, 50, 75) for d in (0.0, 0.05, 0.1, 0.25, 0.5)]
HDBSCAN_GRID = [(mcs, ms) for mcs in (10, 15, 20, 30, 50, 75, 100) for ms in (5, 10, 15, 25)]

MAIN_SEED = 42
BOOTSTRAP_N_ITER = 100
BOOTSTRAP_SUBSAMPLE_FRAC = 0.8
BOOTSTRAP_SEED_BASE = 1000
SEED_SENSITIVITY_SEEDS = [1, 2, 3, 4, 5]
ARI_STABILITY_THRESHOLD = 0.6

N_WORKERS_SWEEP = min(len(UMAP_GRID), max(1, mp.cpu_count() - 2))
N_WORKERS_BOOTSTRAP = min(16, max(1, mp.cpu_count() - 2))


def fit_umap(X, n_neighbors, min_dist, seed):
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=seed)
    return reducer.fit_transform(X)


def fit_hdbscan(embedding2d, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    return clusterer.fit_predict(embedding2d)


def cluster_metrics(embedding2d, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_frac = float(np.mean(labels == -1))
    mask = labels != -1
    if n_clusters < 2 or mask.sum() < 2:
        return n_clusters, noise_frac, float("nan"), float("nan")
    sil = silhouette_score(embedding2d[mask], labels[mask])
    db = davies_bouldin_score(embedding2d[mask], labels[mask])
    return n_clusters, noise_frac, sil, db


def _sweep_worker(args):
    n_neighbors, min_dist = args
    emb = fit_umap(X, n_neighbors, min_dist, seed=MAIN_SEED)
    rows = []
    for min_cluster_size, min_samples in HDBSCAN_GRID:
        labels = fit_hdbscan(emb, min_cluster_size, min_samples)
        n_clusters, noise_frac, sil, db = cluster_metrics(emb, labels)
        rows.append(dict(
            n_neighbors=n_neighbors, min_dist=min_dist,
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            n_clusters=n_clusters, noise_frac=noise_frac,
            silhouette=sil, davies_bouldin=db,
        ))
    # Don't keep all 35 (6714, 2) embeddings in the returned payload at once if
    # avoidable -- they're small (6714*2*8 bytes ~ 107KB each * 35 ~ 3.7MB), fine.
    return (n_neighbors, min_dist), emb, rows


def choose_best_config(sweep_df):
    candidates = sweep_df[
        (sweep_df["n_clusters"] >= 2) & (sweep_df["n_clusters"] <= 15)
        & (sweep_df["noise_frac"] < 0.3) & sweep_df["silhouette"].notna()
    ]
    fallback_used = False
    if candidates.empty:
        fallback_used = True
        candidates = sweep_df[(sweep_df["n_clusters"] >= 2) & sweep_df["silhouette"].notna()]
    best = candidates.loc[candidates["silhouette"].idxmax()]
    return best, fallback_used


def _bootstrap_worker(i):
    rng = np.random.default_rng(BOOTSTRAP_SEED_BASE + i)
    n_sample = int(round(BOOTSTRAP_SUBSAMPLE_FRAC * len(unique_lesions)))
    subsample_lesions = rng.choice(unique_lesions, size=n_sample, replace=False)
    mask = np.isin(lesion_ids, subsample_lesions)
    emb_sub = fit_umap(X[mask], best_n_neighbors, best_min_dist, seed=BOOTSTRAP_SEED_BASE + i)
    labels_sub = fit_hdbscan(emb_sub, best_min_cluster_size, best_min_samples)
    ari = adjusted_rand_score(full_labels[mask], labels_sub)
    return ari


def _seed_worker(seed):
    emb = fit_umap(X, best_n_neighbors, best_min_dist, seed=seed)
    labels = fit_hdbscan(emb, best_min_cluster_size, best_min_samples)
    ari = adjusted_rand_score(full_labels, labels)
    return seed, ari


if __name__ == "__main__":
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading metadata + CLS embeddings (melanoma-only subset)...")
    df = pd.read_csv(METADATA_CSV, low_memory=False)
    df = df[df["diagnosis_3"].isin(MELANOMA_DIAGNOSES)].reset_index(drop=True)
    df["lesion_id"] = df["lesion_id"].fillna(df["isic_id"])
    isic_ids = df["isic_id"].tolist()
    lesion_ids = df["lesion_id"].to_numpy()
    unique_lesions = np.unique(lesion_ids)

    missing = [iid for iid in isic_ids if not os.path.exists(os.path.join(CLS_DIR, f"{iid}.npy"))]
    assert not missing, f"{len(missing)} melanoma isic_ids have no CLS embedding file: {missing[:5]}"

    X = np.stack([np.load(os.path.join(CLS_DIR, f"{iid}.npy")) for iid in isic_ids]).astype(np.float32)
    print(f"X shape: {X.shape}, unique lesions: {len(unique_lesions)}")

    print(f"Wide sweep: {len(UMAP_GRID)} UMAP fits x {len(HDBSCAN_GRID)} HDBSCAN configs "
          f"= {len(UMAP_GRID) * len(HDBSCAN_GRID)} combos, {N_WORKERS_SWEEP} worker processes...")
    t0 = time.time()
    with mp.Pool(N_WORKERS_SWEEP) as pool:
        sweep_results = pool.map(_sweep_worker, UMAP_GRID)
    print(f"Sweep done in {time.time()-t0:.1f}s")

    embeddings_by_config = {key: emb for key, emb, _ in sweep_results}
    sweep_rows = [row for _, _, rows in sweep_results for row in rows]
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(SWEEP_CSV, index=False)
    print(f"Saved sweep results ({len(sweep_df)} rows) to {SWEEP_CSV}")

    best, fallback_used = choose_best_config(sweep_df)
    best_n_neighbors = int(best["n_neighbors"])
    best_min_dist = float(best["min_dist"])
    best_min_cluster_size = int(best["min_cluster_size"])
    best_min_samples = int(best["min_samples"])
    print(f"Winning config (fallback_used={fallback_used}): "
          f"n_neighbors={best_n_neighbors}, min_dist={best_min_dist}, "
          f"min_cluster_size={best_min_cluster_size}, min_samples={best_min_samples}, "
          f"silhouette={best['silhouette']:.4f}, davies_bouldin={best['davies_bouldin']:.4f}, "
          f"n_clusters={int(best['n_clusters'])}, noise_frac={best['noise_frac']:.3f}")

    final_emb = embeddings_by_config[(best_n_neighbors, best_min_dist)]
    full_labels = fit_hdbscan(final_emb, best_min_cluster_size, best_min_samples)

    pd.DataFrame({"isic_id": isic_ids, "cluster_label": full_labels}).to_csv(FINAL_LABELS_CSV, index=False)
    pd.DataFrame({"isic_id": isic_ids, "umap_x": final_emb[:, 0], "umap_y": final_emb[:, 1]}).to_csv(
        FINAL_UMAP_CSV, index=False)
    print(f"Saved final cluster labels to {FINAL_LABELS_CSV}, UMAP coords to {FINAL_UMAP_CSV}")

    print(f"Bootstrap stability: {BOOTSTRAP_N_ITER} iterations, lesion-level "
          f"{BOOTSTRAP_SUBSAMPLE_FRAC:.0%} subsampling, {N_WORKERS_BOOTSTRAP} workers...")
    t0 = time.time()
    with mp.Pool(N_WORKERS_BOOTSTRAP) as pool:
        bootstrap_aris = pool.map(_bootstrap_worker, range(BOOTSTRAP_N_ITER))
    bootstrap_aris = np.array(bootstrap_aris)
    print(f"Bootstrap done in {time.time()-t0:.1f}s | "
          f"mean ARI={bootstrap_aris.mean():.4f}, std={bootstrap_aris.std():.4f}")

    print(f"Seed sensitivity: {len(SEED_SENSITIVITY_SEEDS)} refits on full data...")
    t0 = time.time()
    with mp.Pool(min(len(SEED_SENSITIVITY_SEEDS), N_WORKERS_SWEEP)) as pool:
        seed_results = pool.map(_seed_worker, SEED_SENSITIVITY_SEEDS)
    print(f"Seed sensitivity done in {time.time()-t0:.1f}s")

    is_stable = bool(bootstrap_aris.mean() >= ARI_STABILITY_THRESHOLD)
    report = dict(
        subset="melanoma_only_widesweep", n_images=len(isic_ids),
        grid_size=len(UMAP_GRID) * len(HDBSCAN_GRID),
        winning_config=dict(
            n_neighbors=best_n_neighbors, min_dist=best_min_dist,
            min_cluster_size=best_min_cluster_size, min_samples=best_min_samples,
            fallback_used=fallback_used,
        ),
        main_seed=MAIN_SEED,
        n_clusters=int(best["n_clusters"]), noise_frac=float(best["noise_frac"]),
        silhouette=float(best["silhouette"]), davies_bouldin=float(best["davies_bouldin"]),
        bootstrap=dict(
            n_iter=BOOTSTRAP_N_ITER, subsample_frac=BOOTSTRAP_SUBSAMPLE_FRAC,
            level="lesion", seed_base=BOOTSTRAP_SEED_BASE,
            mean_ari=float(bootstrap_aris.mean()), std_ari=float(bootstrap_aris.std()),
            min_ari=float(bootstrap_aris.min()), max_ari=float(bootstrap_aris.max()),
            ari_values=bootstrap_aris.tolist(),
            stability_threshold=ARI_STABILITY_THRESHOLD, is_stable=is_stable,
        ),
        seed_sensitivity=dict(
            reference_seed=MAIN_SEED,
            aris={str(seed): ari for seed, ari in seed_results},
        ),
        wall_clock_seconds=time.time() - t_start,
    )
    with open(VALIDATION_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved Step 4 validation report to {VALIDATION_JSON}")
    print(f"STABLE (mean ARI >= {ARI_STABILITY_THRESHOLD})" if is_stable else
          f"UNSTABLE (mean ARI < {ARI_STABILITY_THRESHOLD}) — flag per pre-registered acceptance threshold")
    print(f"Total wall clock: {time.time()-t_start:.1f}s")

    assert len(sweep_df) == len(UMAP_GRID) * len(HDBSCAN_GRID)
    assert len(full_labels) == len(isic_ids) == X.shape[0]
    assert len(bootstrap_aris) == BOOTSTRAP_N_ITER
    assert all(a <= 1.0 for a in bootstrap_aris)
    print("Sanity checks passed.")
