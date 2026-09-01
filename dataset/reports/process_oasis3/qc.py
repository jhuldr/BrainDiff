#!/usr/bin/env python3
"""
Phase 6: quality control on the written OASIS-3 volumes.

Three questions, in the order they matter:

1. Is the geometry the contract the dataloaders expect? (193x229x193, MNI affine)
2. Did the skull strip behave? -- rim occupancy, brain volume, FOV fraction.
   The raw T1w measures 90.6% of its outer 8-voxel shell nonzero; a stripped
   volume in this tree should be under 1%, which is where YALE (0.06%) and
   BraTS-MET (0.09%) sit.
3. Did each modality's independent affine to MNI actually land? -- mutual
   information against the T1w of the same session, and against the template.

Question 3 is the one this pipeline's design makes load-bearing. Because every
modality reaches MNI on its own affine rather than being coregistered to the T1w
first, a bad T2w or FLAIR fit is not caught by anything upstream. OASIS-3's T2w
(1x1x4 mm) and FLAIR (5-6 mm, 24-35 slices) are the likely failures.

The metric has to be INTENSITY-based, not support-based. Phase 4 masks every
modality of a session with the same MNI-space mask, so their nonzero supports are
identical by construction -- support Dice measured exactly 1.0000 for both
modalities of the first processed session and would do so for a T2w rotated 90
degrees. `dice_vs_mask` is therefore kept only as a check that masking ran, never
as a registration check. `mi_vs_t1w` is the real one: a misregistered T2w still
fills the mask, but its intensities stop predicting the T1w's.

Thresholds for questions 1 and 2 are fixed, because they are contracts. The
registration cut is NOT hardcoded: the point is to produce the distribution so a
cut can be chosen from it. --blank-failing then applies whatever you chose.

    python qc.py                              # measure, print the distribution
    python qc.py --blank-failing --min-mi 0.20

Blanking clears the modality's cell in OUTPUT/longitudinal.csv and leaves the
.nii.gz on disk. An absent modality contributes no keys to the Perceiver
(models/MultiModal/model.py:457 presence masking), which is strictly better than a
misregistered one.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

import resources  # noqa: E402
from paths import (
    EXPECTED_SHAPE, MNI_REFERENCE, MODALITIES, OUTPUT_ROOT,
    aligned_path, mask_mni_path, output_dir,
)

RIM = 8


def rim_fraction(data):
    """Nonzero fraction of the outer RIM-voxel shell -- the skull-strip detector."""
    mask = np.zeros(data.shape, dtype=bool)
    mask[:RIM] = mask[-RIM:] = True
    mask[:, :RIM] = mask[:, -RIM:] = True
    mask[:, :, :RIM] = mask[:, :, -RIM:] = True

    return float((data[mask] > 0).mean())


def mutual_information(a, b, bins=64, support=None):
    """MI between two volumes, over `support` (default: where either is nonzero).

    MI rather than correlation because the two volumes are different contrasts --
    T2w against T1w, or either against the template -- so intensities are related
    but not linearly. That is the same reason ANTs optimises Mattes MI to produce
    these alignments in the first place.
    """
    if support is None:
        support = (a > 0) | (b > 0)

    if support.sum() < 1000:
        return float("nan")

    joint, _, _ = np.histogram2d(a[support].ravel(), b[support].ravel(), bins=bins)
    joint = joint / joint.sum()

    p_a = joint.sum(axis=1, keepdims=True)
    p_b = joint.sum(axis=0, keepdims=True)

    nonzero = joint > 0
    outer = (p_a @ p_b)[nonzero]

    return float(np.sum(joint[nonzero] * np.log(joint[nonzero] / outer)))


def dice(a, b):
    total = a.sum() + b.sum()

    return float(2.0 * (a & b).sum() / total) if total else float("nan")


def check_session(task):
    """Measure every written modality of one session."""
    subject_id, timepoint, output_root, reference_data, reference_affine = task

    mask_file = mask_mni_path(subject_id, timepoint, output_root)
    brain_mask = None
    if mask_file.is_file():
        brain_mask = np.asarray(nib.load(str(mask_file)).dataobj) != 0

    # The T1w is the reference every other modality is judged against, so it must
    # be read first even though it is reported like any other row.
    t1w_path = aligned_path(subject_id, timepoint, "T1w", output_root)
    t1w_data = (
        np.asarray(nib.load(str(t1w_path)).dataobj, dtype=np.float32)
        if t1w_path.is_file()
        else None
    )

    rows = []

    for modality in MODALITIES:
        path = aligned_path(subject_id, timepoint, modality, output_root)

        if not path.is_file():
            continue

        image = nib.load(str(path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        support = data > 0
        zooms = image.header.get_zooms()[:3]

        # Inside the brain only. Outside it every volume is identically zero, and
        # including those voxels inflates MI with agreement that masking created.
        if brain_mask is not None:
            inside = brain_mask
        else:
            inside = support

        mi_vs_t1w = float("nan")
        if t1w_data is not None and modality != "T1w":
            mi_vs_t1w = mutual_information(data, t1w_data, support=inside)

        rows.append(
            {
                "subject_id": subject_id,
                "Timepoint": timepoint,
                "modality": modality,
                "path": str(path),
                "shape_ok": tuple(image.shape[:3]) == EXPECTED_SHAPE,
                "affine_ok": bool(np.allclose(image.affine, reference_affine, atol=1e-4)),
                "all_zero": not bool(support.any()),
                "rim_fraction": rim_fraction(data),
                "brain_cm3": float(support.sum() * np.prod(zooms) / 1000.0),
                "fov_fraction": float(support.mean()),
                # Masking check, NOT a registration check -- see the module docstring.
                "dice_vs_mask": dice(support, brain_mask) if brain_mask is not None else float("nan"),
                # The real registration check for T2w/FLAIR.
                "mi_vs_t1w": mi_vs_t1w,
                "mi_vs_template": mutual_information(data, reference_data),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=resources.default_workers(),
                        help="Default: 75%% of cores.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Check only N sessions. Default: all of them.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blank-failing", action="store_true",
                        help="Clear failing T2w/FLAIR cells in longitudinal.csv.")
    parser.add_argument("--min-mi", type=float, default=None,
                        help="mi_vs_t1w cut for --blank-failing. Pick it from the "
                             "distribution this script prints; there is no safe default.")
    args = parser.parse_args()

    out_dir = output_dir(args.output_root)
    longitudinal_path = out_dir / "longitudinal.csv"

    if not longitudinal_path.is_file():
        raise SystemExit(f"{longitudinal_path} not found. Run build_longitudinal.py first.")

    longitudinal = pd.read_csv(longitudinal_path, dtype={"subject_id": str})

    sessions = list(zip(longitudinal["subject_id"], longitudinal["Timepoint"]))

    if args.sample and len(sessions) > args.sample:
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(len(sessions), args.sample, replace=False)
        sessions = [sessions[i] for i in picks]

    print(f"QC over {len(sessions):,} sessions, {args.workers} workers")

    reference = nib.load(str(MNI_REFERENCE))
    reference_data = np.asarray(reference.dataobj, dtype=np.float32)
    reference_affine = reference.affine

    tasks = [
        (subject_id, int(timepoint), args.output_root, reference_data, reference_affine)
        for subject_id, timepoint in sessions
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(check_session, tasks, chunksize=8),
                           total=len(tasks), unit="sess"):
            rows.extend(result)

    report = pd.DataFrame(rows)

    if report.empty:
        raise SystemExit("No written volumes found to check.")

    report_path = out_dir / "qc_report.csv"
    report.to_csv(report_path, index=False)

    print("\n=== 1. Geometry (the dataloader contract) ===")
    print(f"  volumes checked   : {len(report):,}")
    print(f"  shape ok          : {int(report['shape_ok'].sum()):,} / {len(report):,}")
    print(f"  affine ok         : {int(report['affine_ok'].sum()):,} / {len(report):,}")
    print(f"  all-zero volumes  : {int(report['all_zero'].sum()):,}")

    print("\n=== 2. Skull strip (raw OASIS-3 T1w measured 90.6% rim) ===")
    for modality, group in report.groupby("modality"):
        rim = 100 * group["rim_fraction"]
        print(f"  {modality:6s} rim% median {rim.median():6.3f}  p95 {rim.quantile(0.95):6.3f}  "
              f"max {rim.max():6.2f}   n>1%: {int((rim > 1).sum()):,}/{len(rim):,}")

    brain = report[report["modality"] == "T1w"]["brain_cm3"]
    if len(brain):
        quantiles = np.percentile(brain, [1, 5, 50, 95, 99])
        print("  T1w brain_cm3 p1/5/50/95/99 = " + " / ".join(f"{v:.0f}" for v in quantiles))
        print(f"  implausible (<800 or >2200 cm3): "
              f"{int(((brain < 800) | (brain > 2200)).sum()):,} / {len(brain):,}")

    # Masking sanity, not registration: every modality is masked with the same
    # MNI mask, so this is ~1.0 by construction. A value below 1 means a volume
    # has interior zeros; a NaN means the mask was missing.
    bad_mask = report[report["dice_vs_mask"] < 0.999]
    print(f"\n=== 3a. Masking applied (dice_vs_mask, ~1.0 by construction) ===")
    print(f"  below 0.999      : {len(bad_mask):,} / {len(report):,}")
    print(f"  mask missing     : {int(report['dice_vs_mask'].isna().sum()):,}")

    print("\n=== 3b. Registration (mi_vs_t1w -- the real check) ===")
    for modality, group in report.groupby("modality"):
        values = group["mi_vs_t1w"].dropna()
        if not len(values):
            continue
        quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95])
        print(f"  {modality:6s} mi_vs_t1w      p1/5/25/50/75/95 = "
              + " / ".join(f"{v:.3f}" for v in quantiles))

    for modality, group in report.groupby("modality"):
        values = group["mi_vs_template"].dropna()
        if not len(values):
            continue
        quantiles = np.percentile(values, [1, 5, 50, 95])
        print(f"  {modality:6s} mi_vs_template p1/5/50/95       = "
              + " / ".join(f"{v:.3f}" for v in quantiles))

    scored = report[report["mi_vs_t1w"].notna()]
    if len(scored):
        worst = scored.nsmallest(10, "mi_vs_t1w")[
            ["subject_id", "Timepoint", "modality", "mi_vs_t1w", "mi_vs_template", "brain_cm3"]
        ]
        print(f"\n=== 10 lowest mi_vs_t1w (inspect before choosing --min-mi) ===\n"
              f"{worst.to_string(index=False)}")

    print(f"\nFull report: {report_path}")

    if not args.blank_failing:
        print("\nNo cells changed. Re-run with --blank-failing --min-mi X to apply a cut.")
        return

    if args.min_mi is None:
        raise SystemExit(
            "--blank-failing needs an explicit --min-mi. Choose it from the "
            "distribution above; defaulting it here would encode a guess."
        )

    if args.sample:
        raise SystemExit("--blank-failing needs a full pass; drop --sample.")

    # T1w is never blanked: it is the mask source and the reference mi_vs_t1w is
    # measured against, so a session whose T1w failed should be dropped upstream,
    # not silently emptied.
    failing = report[
        (report["modality"] != "T1w")
        & (report["mi_vs_t1w"] < args.min_mi)
    ]

    keys = {(r.subject_id, int(r.Timepoint), r.modality) for r in failing.itertuples()}

    if not keys:
        print(f"\nNothing below mi_vs_t1w {args.min_mi}. longitudinal.csv unchanged.")
        return

    changed = 0
    for index, row in longitudinal.iterrows():
        for modality in MODALITIES:
            if (str(row["subject_id"]), int(row["Timepoint"]), modality) in keys:
                longitudinal.at[index, modality] = ""
                changed += 1

    longitudinal.to_csv(longitudinal_path, index=False)

    # Persist the decision, not just its effect. build_longitudinal.py rebuilds
    # the modality cells from what is on disk, so without this record a later
    # re-run would silently restore every volume blanked here.
    blanked_path = out_dir / "blanked.csv"
    record = failing[["subject_id", "Timepoint", "modality", "mi_vs_t1w"]].copy()
    record["reason"] = f"mi_vs_t1w < {args.min_mi}"

    if blanked_path.is_file():
        previous = pd.read_csv(blanked_path, dtype={"subject_id": str})
        record = pd.concat([previous, record], ignore_index=True)
        record = record.drop_duplicates(subset=["subject_id", "Timepoint", "modality"])

    record.to_csv(blanked_path, index=False)

    print(f"\nBlanked {changed:,} modality cells below mi_vs_t1w {args.min_mi}:")
    for modality, count in failing["modality"].value_counts().items():
        print(f"    {modality:6s} {count:,}")
    print("  .nii.gz files left on disk for inspection")
    print(f"  Rewrote {longitudinal_path}")
    print(f"  Recorded {len(record):,} blanked cells in {blanked_path}; "
          f"build_longitudinal.py honours it on re-run.")


if __name__ == "__main__":
    main()
