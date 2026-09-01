"""Index the aligned lesion masks -> lesion_index.csv, for the gate diagnostic.

Standalone by design: it walks `<DATASET>_LESION/` and pairs each mask with its
aligned volume by filename. It does NOT read image.csv/main.csv, so it is safe to
run while those are being rebuilt, survives them being rebuilt, and picks up any
new dataset automatically.

    volume_path,mask_path,dataset,subject,timepoint,modality,
    n_lesion_voxels,n_brain_voxels,lesion_frac,labels,ok

`volume_path,mask_path` are the two columns
`trainer/testers/diag_gate_localization.py --mask-csv` requires; the rest are QC.

Rows with an EMPTY mask are kept and flagged, not dropped -- a timepoint with no
lesion is a legitimate half of a "resolved" pair, and the diagnostic needs it to
compute the CHANGE (xor) region.

QC note: do NOT compare aligned lesion volume against the raw seg's. Affine
normalisation to the template rescales the whole head -- measured 2.49x on
BraTS-MET-00559, i.e. 1.36x linear -- so an unscaled comparison flags every case.
`lesion_frac` (lesion voxels / brain voxels) is the scale-invariant quantity;
measured raw 0.177% vs aligned 0.168% for that case.

    python -m dataset3D.build_lesion_index
    python -m dataset3D.build_lesion_index --datasets BraTS-GLI --out /tmp/x.csv
"""
import argparse
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

S3 = Path("/home/data/BRAIN_DIFF_S3")
CANON_SHAPE = (193, 229, 193)
MODALITIES = ("T1w", "T1ce", "T2w", "FLAIR")
# Subject id as it appears at the head of an aligned filename.
SUBJECT_RE = re.compile(r"^(BraTS-\w+-\d+|sub-[\w\-]+?)_")


def parse_name(mask_path: Path):
    """`<subject>_<timepointdir>_<MOD>_lesion.nii.gz` -> (subject, timepoint, modality)."""
    stem = mask_path.name[: -len("_lesion.nii.gz")]
    mod = next((m for m in MODALITIES if stem.endswith(f"_{m}")), None)
    if mod is None:
        return None
    rest = stem[: -(len(mod) + 1)]
    m = SUBJECT_RE.match(rest + "_")
    subject = m.group(1) if m else rest.split("_")[0]
    timepoint = rest[len(subject) + 1:] if rest.startswith(subject + "_") else rest
    return subject, timepoint, mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(S3))
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset names; default = every *_LESION dir found")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-stats", action="store_true",
                    help="skip voxel counts (much faster; loses the QC columns)")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "lesion_index.csv"

    lesion_dirs = ([root / f"{d}_LESION" for d in a.datasets] if a.datasets
                   else sorted(root.glob("*_LESION")))
    lesion_dirs = [d for d in lesion_dirs if d.is_dir()]
    if not lesion_dirs:
        print(f"No *_LESION directories under {root}. Run create_longitudinal.py first.")
        return 1

    rows, problems = [], []
    for ldir in lesion_dirs:
        dataset = ldir.name[: -len("_LESION")]
        aligned = root / dataset / "ALIGNED"
        masks = sorted(ldir.glob("*_lesion.nii.gz"))
        print(f"{dataset}: {len(masks)} masks", flush=True)
        for mp in tqdm(masks, desc=dataset, unit="mask", leave=False):
            parsed = parse_name(mp)
            if parsed is None:
                problems.append(f"{mp.name}: unparseable name"); continue
            subject, timepoint, mod = parsed
            vp = aligned / mp.name.replace("_lesion.nii.gz", ".nii.gz")
            if not vp.exists():
                problems.append(f"{mp.name}: no matching volume {vp.name}"); continue

            row = dict(volume_path=str(vp), mask_path=str(mp), dataset=dataset,
                       subject=subject, timepoint=timepoint, modality=mod)
            if a.no_stats:
                rows.append({**row, "ok": True}); continue

            mi = nib.load(mp)
            if mi.shape != CANON_SHAPE:
                problems.append(f"{mp.name}: shape {mi.shape} != {CANON_SHAPE}"); continue
            msk = mi.get_fdata()
            labels = np.unique(msk)
            # The real test for nearest-neighbour preservation is INTEGRALITY, not
            # membership in a fixed list: linear interpolation produces fractional
            # values (2.4), whereas an unexpected-but-whole label is a property of
            # the source. Measured: BraTS-MET-01094-002 genuinely ships label 6
            # (129 voxels), 1 of 773 raw segs -- a source anomaly, not a bug.
            clean = bool(np.all(labels == np.round(labels)))
            if not clean:
                problems.append(f"{mp.name}: FRACTIONAL values {labels[:6]} "
                                f"-- interpolation blended labels")
            nonstd = sorted(set(labels.tolist()) - {0., 1., 2., 3., 4.})
            if nonstd:
                problems.append(f"{mp.name}: non-standard (but integer) labels "
                                f"{nonstd} -- check the source seg")
            brain = nib.load(vp).get_fdata() > 0
            n_les, n_brain = int((msk > 0).sum()), int(brain.sum())
            rows.append({**row,
                         "n_lesion_voxels": n_les,
                         "n_brain_voxels": n_brain,
                         "lesion_frac": round(n_les / n_brain, 6) if n_brain else np.nan,
                         "labels": "|".join(str(int(x)) for x in labels),
                         "ok": clean and n_brain > 0})

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df):,} rows -> {out}")
    if len(df):
        print(f"  datasets      : {df.dataset.value_counts().to_dict()}")
        print(f"  subjects      : {df.subject.nunique():,}")
        print(f"  timepoints    : {df.groupby(['dataset','subject','timepoint']).ngroups:,}")
        if not a.no_stats:
            empty = int((df.n_lesion_voxels == 0).sum())
            print(f"  empty masks   : {empty:,} ({100*empty/len(df):.1f}%)  [kept, flagged]")
            print(f"  lesion_frac   : median {df.lesion_frac.median():.5f}  "
                  f"p95 {df.lesion_frac.quantile(.95):.5f}  max {df.lesion_frac.max():.5f}")
            print(f"  label sets    : {df.labels.value_counts().head(5).to_dict()}")
            print(f"  ok            : {int(df.ok.sum()):,} / {len(df):,}")
    if problems:
        print(f"\n  {len(problems)} PROBLEM(S), first 10:")
        for p in problems[:10]:
            print(f"    {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
