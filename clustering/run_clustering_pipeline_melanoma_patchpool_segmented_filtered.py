"""Melanoma-only UMAP+HDBSCAN clustering/validation, patch-meanpool variant.
Identical pipeline to run_clustering_pipeline_melanoma_only.py
but clusters on mean-pooled DINOv2 patch-token embeddings
(embeddings/melanoma_patch_meanpool_segmented_filtered.npy, built by
embeddings/build_melanoma_patch_meanpool.py) instead of the CLS token, to test
whether local/patch-level signal reveals stabler within-melanoma structure
than the CLS-based run did (that run came back UNSTABLE: mean ARI=0.488).

Outputs go to clustering/melanoma_only_patchpool/ — the CLS-based
clustering/melanoma_only/ outputs are untouched.

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
POOLED_NPY = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_segmented_filtered.npy")
POOLED_IDS_CSV = os.path.join(ROOT, "embeddings", "melanoma_patch_meanpool_segmented_filtered_isic_ids.csv")
METADATA_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")

OUT_DIR = os.path.join(ROOT, "clustering", "melanoma_only_patchpool_segmented_filtered")
SWEEP_CSV = os.path.join(OUT_DIR, "sweep_results.csv")
FINAL_LABELS_CSV = os.path.join(OUT_DIR, "final_cluster_labels.csv")
FINAL_UMAP_CSV = os.path.join(OUT_DIR, "final_umap_2d.csv")
VALIDATION_JSON = os.path.join(OUT_DIR, "step4_validation_report.json")

# Same grid as the other clustering runs — kept identical for comparability.
UMAP_GRID = [(n, d) for n in (15, 30, 50) for d in (0.0, 0.1, 0.25)]
HDBSCAN_GRID = [(mcs, ms) for mcs in (15, 25, 50) for ms in (5, 15)]

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

    print("Loading melanoma-only patch-meanpool embeddings + lesion metadata...")
    isic_ids = pd.read_csv(POOLED_IDS_CSV)["isic_id"].tolist()
    X = np.load(POOLED_NPY).astype(np.float32)
    assert X.shape[0] == len(isic_ids)

    df = pd.read_csv(METADATA_CSV, low_memory=False).set_index("isic_id")
    lesion_ids = df.loc[isic_ids, "lesion_id"].fillna(pd.Series(isic_ids, index=isic_ids)).to_numpy()
    unique_lesions = np.unique(lesion_ids)
    print(f"X shape: {X.shape}, unique lesions: {len(unique_lesions)}, "
          f"lesions with >1 image: {int((pd.Series(lesion_ids).value_counts() > 1).sum())}")

    print(f"Sweep: {len(UMAP_GRID)} UMAP fits x {len(HDBSCAN_GRID)} HDBSCAN configs, "
          f"{N_WORKERS_SWEEP} worker processes...")
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
        subset="melanoma_only_patchpool_segmented_filtered", n_images=len(isic_ids),
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
