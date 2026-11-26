#!/usr/bin/env python3
"""
Registration QC of every curriculum session, and the BIND patient split. CPU only; no image is viewed.

    python bind_qc_split.py qc --workers 64     (env bind_proc)  -> $BRAINDIFF_BUILD/qc/session_qc.csv
    python bind_qc_split.py split                                -> $BRAINDIFF_BUILD/splits/bind_patient_split.csv

qc: per modality, the brain (voxels > 0) of the template-space volume against the MNI reference brain
(reference > its 5th intensity percentile, same 193x229x193 grid):
    coverage   = |brain ∩ ref| / |ref|     placement = |brain ∩ ref| / |brain|
    broken     coverage < 0.10 or placement < 0.80   -> modality dropped downstream (build_stage2.py)
    partial    0.10 <= coverage < 0.50                -> kept, flagged (limited-coverage slab)
    ok         otherwise;  missing file -> "missing"
Sessions: both studies of every BIND report pair (bind_longitudinal_meta_paired.csv) + MR-RATE (S4 image_extended.csv).

split: I0004 -> external_test (whole site; never in training); I0001 -> train / val / test 80/10/10 by
md5(site:patient_uid); other sites -> train. Hashing by patient keeps late-registered pairs in an existing split.
Patients = report pairs + image-only pairs.

Env: BIND_TABLES, BIND_OUT, MNI_REFERENCE (see bind_preprocess.py); BRAINDIFF_DATA (holds BRAIN_DIFF_S4);
BRAINDIFF_BUILD (output root).
"""
import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from bind_preprocess import MNI_REFERENCE, MODALITIES, OUTPUT_ROOT, PAIRED_CSV, TABLES, session_dir

BUILD = Path(os.environ.get("BRAINDIFF_BUILD", "build"))
S4_IMAGES = Path(os.environ.get("BRAINDIFF_DATA", "data")) / "BRAIN_DIFF_S4" / "image_extended.csv"
S4_COLS = {"t1w": "T1w", "t1ce": "T1ce", "t2w": "T2w", "flair": "FLAIR"}
EXTERNAL_SITE = "I0004"


@lru_cache(maxsize=1)
def ref_brain():
    import nibabel as nib
    ref = nib.load(MNI_REFERENCE).get_fdata()
    return ref > np.percentile(ref[ref > 0], 5)


# ------------------------------------------------------------------ qc
def measure(item):
    import nibabel as nib
    source, site, session, modality, path = item
    row = {"source": source, "site": site, "session": session, "modality": modality, "path": path}
    if not isinstance(path, str) or not Path(path).is_file():
        return row | {"coverage": np.nan, "placement": np.nan, "status": "missing"}
    try:
        b = np.asarray(nib.load(path).dataobj) > 0
    except Exception as error:
        return row | {"coverage": np.nan, "placement": np.nan, "status": f"unreadable {type(error).__name__}"}
    rb = ref_brain()
    inter = (b & rb).sum()
    cov, pla = inter / rb.sum(), inter / max(1, b.sum())
    status = "broken" if (cov < 0.10 or pla < 0.80) else "partial" if cov < 0.50 else "ok"
    return row | {"coverage": round(float(cov), 4), "placement": round(float(pla), 4), "status": status}


def items():
    p = pd.read_csv(PAIRED_CSV, usecols=["site", "study_uid1", "study_uid2"], dtype=str)
    sess = sorted(set(zip(p.site, p.study_uid1)) | set(zip(p.site, p.study_uid2)))
    out = []
    for site, ses in sess:
        d = session_dir(OUTPUT_ROOT, site, ses)
        present = [m for m in MODALITIES if (d / f"{m}.nii.gz").is_file()]
        out += [("BIND", site, ses, m, str(d / f"{m}.nii.gz")) for m in present] or [("BIND", site, ses, "-", None)]
    s4 = pd.read_csv(S4_IMAGES, usecols=["study_uid"] + list(S4_COLS.values()), dtype=str)
    for r in s4.itertuples():
        out += [("MR-RATE", "MR-RATE", r.study_uid, m, getattr(r, c)) for m, c in S4_COLS.items()
                if isinstance(getattr(r, c), str)]
    return out


def qc(a):
    ref_brain()
    todo = items()
    print(f"{len(todo):,} modality volumes", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = pd.DataFrame(list(ex.map(measure, todo, chunksize=64)))
    (BUILD / "qc").mkdir(parents=True, exist_ok=True)
    res.to_csv(BUILD / "qc" / "session_qc.csv", index=False)           # csv: bind_proc has no parquet engine
    print(res.groupby(["source", "status"]).size().to_string())
    s = res[res.status != "missing"].groupby(["source", "site", "session"]).status.agg(
        lambda x: "all broken" if (x == "broken").all() else "has broken" if (x == "broken").any()
        else "has partial" if (x == "partial").any() else "ok")
    print("sessions:\n" + s.groupby(level=[0, 1]).value_counts().to_string())


# ------------------------------------------------------------------ split
def unit(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 16 ** 8


def split(a):
    rp = pd.read_csv(PAIRED_CSV, usecols=["patient_uid", "site", "duration"], dtype=str)
    io = pd.read_csv(TABLES / "bind_imageonly_pairs.csv", usecols=["patient_uid", "site"], dtype=str)
    pts = pd.concat([rp, io])[["site", "patient_uid"]].drop_duplicates()
    pts["patient"] = pts.site + ":" + pts.patient_uid

    def one(r):
        if r.site == EXTERNAL_SITE:
            return "external_test"
        if r.site != "I0001":
            return "train"
        u = unit(r.patient)
        return "train" if u < 0.8 else "val" if u < 0.9 else "test"

    pts["split"] = [one(r) for r in pts.itertuples()]
    out = BUILD / "splits" / "bind_patient_split.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pts[["patient", "site", "split"]].sort_values("patient").to_csv(out, index=False)
    rp["patient"] = rp.site + ":" + rp.patient_uid
    rp = rp.merge(pts[["patient", "split"]], on="patient")
    print(f"{len(pts):,} patients -> {out}")
    print(rp.groupby("split").agg(pairs=("patient", "size"), patients=("patient", "nunique")).to_string())
    print(pd.crosstab(pts.site, pts.split).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("qc").add_argument("--workers", type=int, default=64)
    sub.add_parser("split")
    a = ap.parse_args()
    {"qc": qc, "split": split}[a.cmd](a)


if __name__ == "__main__":
    main()
