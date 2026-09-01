"""Build the supervised S3 pair table: S3's own pairs + S4-train pairs, with labels.

Two label sets:

  has_dense   both timepoints have an aligned BraTS mask, so a per-token tumour change
              target exists. ~1,633 of S3's 33,393 pairs. Objective and spatially
              localised, but glioma/metastasis only.
  has_global  the pair carries S4's 7-way `classification`. Whole-volume and report-derived
              (~0.80 AUC text-only ceiling, so noisy), but covers the atrophy,
              ventriculomegaly and small-vessel change BraTS does not.

S3 and S4 corpora share no study UIDs, so S4 pairs are appended rather than joined. Writes
main_supervised.csv and leaves main.csv alone so the original S3 stage stays reproducible.

The S4 side is a flag, not a constant, so the extended corpus builds a second table rather
than overwriting the one nv_stage3_deltasup.pt was trained on:

    python -m process_mrrate.build_s3sup_pairs \
        --s4-splits /home/data/BRAIN_DIFF_S4/splits_extended \
        --s4-image  /home/data/BRAIN_DIFF_S4/image_extended.csv \
        --out       /home/data/BRAIN_DIFF_S3/main_supervised_extended.csv \
        --image-out /home/data/BRAIN_DIFF_S3/image_supervised_extended.csv

The leakage check follows --s4-splits, so val/test are read from the same split directory
the train pairs came from; checking against the original holdout would pass while leaking.
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class _Notify:
    """Slack notifications are not part of the release; calls become no-ops."""
    @staticmethod
    def send(*_a, **_k):
        return None


notify = _Notify()

S3_ROOT = "/home/data/BRAIN_DIFF_S3"
S4_ROOT = "/home/data/BRAIN_DIFF_S4"
TOKEN_DIR = os.path.join(S3_ROOT, "LESION_TOKENS")
OUT = os.path.join(S3_ROOT, "main_supervised.csv")
IMAGE_OUT = os.path.join(S3_ROOT, "image_supervised.csv")
MODALITIES = ("T1w", "T1ce", "T2w", "FLAIR")

CHANGE_CLASSES = ("Stable", "New lesion", "Indeterminate", "Progressed",
                  "Improved", "Mixed interval change", "Resolved")
CHANGE_CLASS_TO_IDX = {c: i for i, c in enumerate(CHANGE_CLASSES)}


def lesion_prefix(path):
    """ALIGNED volume path -> the {prefix} that LESION_TOKENS keys are built from."""
    if not isinstance(path, str):
        return None
    base = os.path.basename(path)
    m = re.match(r"^(.+)_(?:T1w|T1ce|T2w|FLAIR)\.nii\.gz$", base)
    return m.group(1) if m else None


def has_tokens(prefix):
    """True if at least one modality has a cached per-token occupancy map."""
    if prefix is None:
        return False
    return any(os.path.exists(os.path.join(TOKEN_DIR, f"{prefix}_{m}.npy"))
               for m in MODALITIES)


def build_s3_rows():
    pairs = pd.read_csv(os.path.join(S3_ROOT, "main.csv"))
    image = pd.read_csv(os.path.join(S3_ROOT, "image.csv"))
    # Resolve the prefix from an ALIGNED modality path, NOT from image.csv's Lesion
    # column: that column stores the pre-alignment source path, which is in original
    # BraTS space and has no cached tokens.
    pref = {}
    for _, r in image.iterrows():
        for m in MODALITIES:
            p = lesion_prefix(r.get(m))
            if p:
                pref[r["UID"]] = p
                break

    rows = []
    for _, r in pairs.iterrows():
        p1, p2 = pref.get(r["UID_1"]), pref.get(r["UID_2"])
        dense = has_tokens(p1) and has_tokens(p2)
        rows.append({"UID_1": r["UID_1"], "UID_2": r["UID_2"], "source": "s3",
                     "prefix_1": p1, "prefix_2": p2,
                     "has_dense": int(dense), "has_global": 0, "change_label": -1,
                     "patient_uid": ""})
    return pd.DataFrame(rows)


def build_s4_rows(s4_splits):
    """S4 TRAIN pairs only. See assert_no_leakage -- this is the load-bearing part."""
    train = pd.read_csv(glob.glob(os.path.join(s4_splits, "*train.csv"))[0])
    rows = []
    for _, r in train.iterrows():
        cls = str(r["classification"])
        if cls not in CHANGE_CLASS_TO_IDX:
            continue
        rows.append({"UID_1": r["study_uid1"], "UID_2": r["study_uid2"], "source": "s4",
                     "prefix_1": None, "prefix_2": None,
                     "has_dense": 0, "has_global": 1,
                     "change_label": CHANGE_CLASS_TO_IDX[cls],
                     "patient_uid": r["patient_uid"]})
    return pd.DataFrame(rows)


def assert_no_leakage(df, s4_splits):
    """Abort if any S4 val/test patient reached the S3 training table.

S3-sup sees S4 pairs with their change labels, so an S4 val patient among them invalidates
every downstream S4 number. The splits are patient-grouped with zero overlap by
construction, so this should never fire.
    """
    holdout = set()
    for split in ("val", "test"):
        for path in glob.glob(os.path.join(s4_splits, f"*{split}.csv")):
            d = pd.read_csv(path)
            if len(d):
                holdout |= set(d["patient_uid"].astype(str))
    present = set(df.loc[df["source"] == "s4", "patient_uid"].astype(str))
    bad = present & holdout
    if bad:
        msg = (f"LEAKAGE: {len(bad)} S4 held-out patients in the S3-sup table, "
               f"e.g. {sorted(bad)[:5]}")
        notify.send(f"*build_s3sup_pairs* ABORTED — {msg}")
        raise SystemExit(msg)
    return len(holdout)


def build_image_table(s4_image):
    """One UID -> modality-path table covering both corpora.

DiffPairDataset takes a single image_csv keyed by `UID`, but S3 and S4 keep separate ones
(S4's key is `study_uid`). Emitting the union keeps the dataset unchanged.
    """
    s3 = pd.read_csv(os.path.join(S3_ROOT, "image.csv"))
    s4 = pd.read_csv(s4_image).rename(columns={"study_uid": "UID"})
    cols = ["UID"] + list(MODALITIES)
    out = pd.concat([s3[cols], s4[cols]], ignore_index=True)
    out = out.drop_duplicates(subset="UID", keep="first")
    return out


def main(a):
    s3 = build_s3_rows()
    s4 = build_s4_rows(a.s4_splits)
    df = pd.concat([s3, s4], ignore_index=True)
    n_holdout = assert_no_leakage(df, a.s4_splits)

    # Drop pairs whose volumes are missing BEFORE writing, so DiffPairDataset never
    # drops rows itself. Its constructor filters silently and keeps no index, so any
    # row it drops would shift every label in the table against the sample it
    # belongs to -- the same off-by-N alignment trap that has bitten this project
    # before, and here it would silently mislabel the supervision.
    img = build_image_table(a.s4_image).set_index("UID")
    keep = []
    for _, r in df.iterrows():
        ok = True
        for uid in (r["UID_1"], r["UID_2"]):
            if uid not in img.index:
                ok = False
                break
            row = img.loc[uid]
            if not any(isinstance(row.get(m), str) and os.path.exists(row.get(m))
                       for m in MODALITIES):
                ok = False
                break
        keep.append(ok)
    n_before = len(df)
    df = df[pd.Series(keep, index=df.index)].reset_index(drop=True)
    print(f"pre-filtered {n_before - len(df)} pairs with missing volumes "
          f"({n_before} -> {len(df)})", flush=True)

    summary = (f"S3-sup pairs [{os.path.basename(a.out)}]: {len(df)} total "
               f"({len(s3)} s3 + {len(s4)} s4-train)\n"
               f"  has_dense  {int(df['has_dense'].sum())}\n"
               f"  has_global {int(df['has_global'].sum())}\n"
               f"  unlabelled {int(((df['has_dense'] == 0) & (df['has_global'] == 0)).sum())}\n"
               f"  leakage check passed against {n_holdout} held-out S4 patients")
    print(summary, flush=True)
    if int(df["has_dense"].sum()) == 0:
        raise SystemExit("no pairs have dense labels -- run dataset3D/build_lesion_tokens.py first")

    if not a.dry_run:
        df.to_csv(a.out, index=False)
        build_image_table(a.s4_image).to_csv(a.image_out, index=False)
        print(f"Wrote {a.out} and {a.image_out}", flush=True)
        notify.send(f"*build_s3sup_pairs* done\n```\n{summary}\n```")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--s4-splits", default=os.path.join(S4_ROOT, "splits"),
                   help="directory holding *train.csv / *val.csv / *test.csv")
    p.add_argument("--s4-image", default=os.path.join(S4_ROOT, "image.csv"))
    p.add_argument("--out", default=OUT)
    p.add_argument("--image-out", default=IMAGE_OUT)
    main(p.parse_args())
