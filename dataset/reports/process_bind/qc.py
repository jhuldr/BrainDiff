#!/usr/bin/env python3
"""
Quality control for the processed BIND volumes.

Three questions, in the order they matter:

1. Is the geometry the contract the dataloaders expect? (193x229x193, S4's affine)
2. Did the affine to MNI actually land? -- mutual information against the MNI
   reference, and Dice between each volume's brain support and the reference's.
3. Did the skull strip behave? -- brain volume in cm^3, and the fraction of the
   field of view it claims.

Thresholds are deliberately NOT hardcoded. The point of this script is to produce
the distribution so a rejection cut can be chosen from it; picking a number in
advance would just encode a guess.

    python qc.py --sample 100
"""

import argparse
import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import EXPECTED_SHAPE, MODALITIES, OUTPUT_ROOT

MNI_REFERENCE = "/path/to/code/BrainDiff/dataset/ants_data/mni_reference.nii.gz"
S4_ROOT = "/home/data/BRAIN_DIFF_S4"


def mutual_information(a, b, bins=64):
    """
    Mutual information between two volumes, over voxels where either is nonzero.

    MI rather than correlation because the moving image and the MNI template are
    different contrasts -- intensities are related but not linearly, which is the
    same reason ANTs optimizes Mattes MI to produce these alignments.
    """
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
    """Dice overlap between two boolean supports."""
    total = a.sum() + b.sum()

    return float(2.0 * (a & b).sum() / total) if total else float("nan")


def reference_arrays():
    """Load the MNI reference volume and its brain support, plus S4's affine."""
    reference = nib.load(MNI_REFERENCE)
    reference_data = np.asarray(reference.dataobj, dtype=np.float32)

    s4_samples = glob.glob(f"{S4_ROOT}/*/*/t1w.nii.gz")
    s4_affine = nib.load(s4_samples[0]).affine if s4_samples else reference.affine

    return reference_data, reference_data > 0, s4_affine


def check_volume(path, reference_data, reference_support, s4_affine):
    """Measure one written volume."""
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)

    zooms = image.header.get_zooms()[:3]
    support = data > 0

    return {
        "path": str(path),
        "site": Path(path).parents[1].name,
        "session_uid": Path(path).parent.name,
        "modality": Path(path).name.replace(".nii.gz", ""),
        "shape_ok": tuple(image.shape[:3]) == EXPECTED_SHAPE,
        "affine_ok": bool(np.allclose(image.affine, s4_affine, atol=1e-6)),
        "brain_cm3": float(support.sum() * np.prod(zooms) / 1000.0),
        "fov_fraction": float(support.mean()),
        "mutual_information": mutual_information(data, reference_data),
        "dice_vs_mni": dice(support, reference_support),
        "all_zero": not bool(support.any()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--sample", type=int, default=100, help="Sessions to check.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path(__file__).parent / "data" / "qc_report.csv")
    args = parser.parse_args()

    sessions = sorted(
        d for d in args.output_root.glob("I*/*") if d.is_dir()
    )

    if not sessions:
        print(f"No processed sessions under {args.output_root}")
        return

    print(f"{len(sessions):,} processed sessions found")

    rng = np.random.default_rng(args.seed)
    if len(sessions) > args.sample:
        sessions = [sessions[i] for i in rng.choice(len(sessions), args.sample, replace=False)]

    reference_data, reference_support, s4_affine = reference_arrays()

    rows = []
    for session in tqdm(sessions, desc="QC", unit="sess"):
        for modality in MODALITIES:
            path = session / f"{modality}.nii.gz"
            if path.is_file():
                rows.append(check_volume(path, reference_data, reference_support, s4_affine))

    report = pd.DataFrame(rows)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report, index=False)

    print(f"\n=== Geometry (the dataloader contract) ===")
    print(f"  volumes checked : {len(report):,}")
    print(f"  shape ok        : {report['shape_ok'].sum():,} / {len(report):,}")
    print(f"  affine matches S4: {report['affine_ok'].sum():,} / {len(report):,}")
    print(f"  all-zero volumes : {report['all_zero'].sum():,}")

    print("\n=== Registration to MNI ===")
    for column in ("mutual_information", "dice_vs_mni"):
        values = report[column].dropna()
        quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95])
        print(f"  {column:20s} p1/5/25/50/75/95 = "
              + " / ".join(f"{v:.3f}" for v in quantiles))

    print("\n=== Skull strip ===")
    brain = report.groupby(["site", "session_uid"])["brain_cm3"].first()
    quantiles = np.percentile(brain, [1, 5, 50, 95, 99])
    print(f"  brain_cm3 p1/5/50/95/99 = " + " / ".join(f"{v:.0f}" for v in quantiles))
    print(f"  implausible (<800 or >2200 cm3): {((brain < 800) | (brain > 2200)).sum():,}"
          f" / {len(brain):,}")
    print(f"  fov_fraction > 0.9 (strip failed): "
          f"{(report['fov_fraction'] > 0.9).sum():,}")

    worst = report.nsmallest(10, "mutual_information")[
        ["site", "session_uid", "modality", "mutual_information", "dice_vs_mni", "brain_cm3"]
    ]
    print(f"\n=== 10 lowest-MI volumes (inspect these) ===\n{worst.to_string(index=False)}")

    print(f"\nFull report: {args.report}")


if __name__ == "__main__":
    main()
