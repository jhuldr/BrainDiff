"""Build the unified stage-1 CSV: bounding-box grounding + pathology classification.

Row-concatenates the two stage sources into the single dataframe the unified
stage-1 trainer reads:

  step1_intake_norm.csv          aref/gcap/caref, boxes normalized to 0..100
  single_timepoint_final.csv     pcls, one pathology label per study

Previously this lived as loose cells in trainer/med_gemma.ipynb, so the file that
training actually consumes could not be regenerated without hand-running a
notebook. Behaviour is preserved deliberately, including that `pcls` rows carry
NO prompt -- `prompt` is not in the column intersection, so those rows come out
NaN and the dataloader substitutes PATHOLOGY_PROMPT
(dataloaders/MultiModal/multi_unified_dataloader.py).

  python -m dataset3D.build_unified_dataframe
"""
import argparse

import numpy as np
import pandas as pd

S1_CSV = "/home/data/BRAIN_DIFF_S1/step1_intake_norm.csv"
S2_CSV = "/home/data/BRAIN_DIFF_S2/single_timepoint_final.csv"
OUT_CSV = "/home/data/BRAIN_DIFF_S1/unified_dataframe_norm.csv"


def build(s1_csv: str = S1_CSV, s2_csv: str = S2_CSV, out_csv: str = OUT_CSV) -> pd.DataFrame:
    s1 = pd.read_csv(s1_csv)
    s2 = pd.read_csv(s2_csv)

    if "pathologies" not in s2.columns:
        raise RuntimeError(
            f"{s2_csv} has no `pathologies` column. Run\n"
            "  python -m process_mrrate.build_multilabel_pathology --patch-s2\n"
            "first -- otherwise pcls rows silently fall back to one label per study."
        )

    s2["task"] = "pcls"
    # The generation target is now every intracranial pathology present,
    # alphabetically ordered and comma-separated (`pathologies`), NOT the single
    # priority-order winner (`pathology`, which stays in the S2 frame for S2's
    # counterfactual grouping). Pathology mirrors caption as before.
    s2["caption"] = s2["pathologies"]
    s2["Pathology"] = s2["pathologies"]

    # Stratification/sampler key is the RAREST label present, not the caption:
    # the comma-joined string has 1,835 distinct values, which would shatter the
    # sampler's buckets and the stratified split. Bounding rows bucket on
    # (with_lesion, task) and carry NaN here.
    s1["pathology_bucket"] = np.nan

    # Keep only columns both sides share. pandas realigns by name on concat, so
    # the alphabetical reorder np.intersect1d introduces is harmless. NOTE: this
    # intersect is what NaNs `prompt` on pcls rows -- and it will silently drop
    # ANY new column missing from one side, so assert what must survive.
    keep = np.intersect1d(s1.columns, s2.columns)
    required = {"caption", "Pathology", "pathology_bucket", "task"}
    missing = required - set(keep)
    if missing:
        raise RuntimeError(f"column intersection dropped required columns: {sorted(missing)}")
    s2 = s2[keep]

    out = pd.concat([s1, s2], axis=0)
    out.to_csv(out_csv, index=False)

    print(f"{s1_csv}: {len(s1):,} rows")
    print(f"{s2_csv}: {len(s2):,} rows")
    print(f"-> {out_csv}: {len(out):,} rows x {len(out.columns)} cols")
    print(f"   columns: {list(out.columns)}")
    print(f"   task distribution:\n{out['task'].value_counts().to_string()}")
    print(f"   rows with no prompt (expected == pcls count): {out['prompt'].isna().sum():,}")

    pcls = out[out["task"] == "pcls"]
    n_lab = pcls["caption"].str.count(", ") + 1
    print(f"   pcls labels/study: mean {n_lab.mean():.2f}  max {n_lab.max()}  "
          f"multi-label {(n_lab > 1).mean():.1%}")
    print(f"   pcls distinct captions: {pcls['caption'].nunique():,}  "
          f"buckets: {pcls['pathology_bucket'].nunique()}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--s1_csv", default=S1_CSV)
    p.add_argument("--s2_csv", default=S2_CSV)
    p.add_argument("--out_csv", default=OUT_CSV)
    a = p.parse_args()
    build(a.s1_csv, a.s2_csv, a.out_csv)
