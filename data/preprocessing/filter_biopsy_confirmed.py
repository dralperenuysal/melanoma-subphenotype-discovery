"""Filter ISIC metadata to biopsy-confirmed cases with an available image file.

Inclusion criterion: sadece histopatolojik olarak doğrulanmış
(diagnosis_confirm_type == "histopathology") vakalar dahil edilir.
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_CSV = os.path.join(ROOT, "data", "metadata", "isic_all_dermoscopic.csv")
IMAGE_DIR = os.path.join(ROOT, "data", "raw", "isic_dermoscopic_targeted")
OUT_CSV = os.path.join(ROOT, "data", "processed", "biopsy_confirmed_metadata.csv")


def run():
    df = pd.read_csv(METADATA_CSV, low_memory=False)
    total = len(df)

    confirmed = df[df["diagnosis_confirm_type"] == "histopathology"]
    excluded_not_histopath = total - len(confirmed)

    available_ids = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR)}
    has_image = confirmed["isic_id"].isin(available_ids)
    final = confirmed[has_image]
    excluded_no_image = len(confirmed) - len(final)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    final.to_csv(OUT_CSV, index=False)

    print(f"total rows:                     {total}")
    print(f"histopathology-confirmed:       {len(confirmed)} (excluded {excluded_not_histopath})")
    print(f"with available image file:      {len(final)} (excluded {excluded_no_image})")
    print(f"written to: {OUT_CSV}")
    return final


if __name__ == "__main__":
    result = run()
    assert len(result) > 0
    available_ids = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR)}
    assert result["isic_id"].isin(available_ids).all()