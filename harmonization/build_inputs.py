"""Build the harmonization-protocol input contract for the melanoma-only cohort, both variants
(raw, cropped): data/embeddings/dinov2_embeddings_{variant}.npy + samples_{variant}.csv.

image_id <- isic_id, dataset <- attribution, diagnosis <- diagnosis_3.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT_CSV = os.path.join(ROOT, "clustering", "melanoma_only_patchpool_cropped", "final_cluster_labels.csv")
META_CSV = os.path.join(ROOT, "data", "processed", "quality_filtered_metadata.csv")
EMB_DIR = os.path.join(ROOT, "data", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

VARIANTS = {
    "raw": os.path.join(ROOT, "embeddings", "dinov2_cls"),
    "cropped": os.path.join(ROOT, "embeddings", "dinov2_cls_cropped_melanoma_only"),
}


def build():
    cohort_ids = pd.read_csv(COHORT_CSV)["isic_id"].tolist()
    meta = pd.read_csv(META_CSV, low_memory=False).set_index("isic_id")

    for variant, cls_dir in VARIANTS.items():
        rows = []
        vecs = []
        for iid in cohort_ids:
            p = os.path.join(cls_dir, f"{iid}.npy")
            if not os.path.exists(p):
                continue
            vecs.append(np.load(p))
            m = meta.loc[iid]
            rows.append({
                "image_id": iid,
                "dataset": m["attribution"],
                "diagnosis": m["diagnosis_3"],
                "age": m.get("age_approx"),
                "sex": m.get("sex"),
                "anatom_site": m.get("anatom_site_1"),
            })
        X = np.stack(vecs).astype(np.float32)
        df = pd.DataFrame(rows)
        np.save(os.path.join(EMB_DIR, f"dinov2_embeddings_{variant}.npy"), X)
        df.to_csv(os.path.join(EMB_DIR, f"samples_{variant}.csv"), index=False)
        print(f"{variant}: X={X.shape}, samples={len(df)}, "
              f"age_fill={df['age'].notna().mean():.2f}, sex_fill={df['sex'].notna().mean():.2f}, "
              f"site_fill={df['anatom_site'].notna().mean():.2f}")
        assert len(df) == X.shape[0]


if __name__ == "__main__":
    build()
