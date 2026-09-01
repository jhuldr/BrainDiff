#!/usr/bin/env python3
"""
Walk the BIND BIDS tree once and cache one row per anatomical series.

The filesystem is slow -- a `find -maxdepth 2` over I0001 alone takes ~4 minutes --
so this walks once, in parallel over subjects, and everything downstream reads the
CSV instead of the tree.

Only `anat/` is read. DWI/SWI/perf/angio are outside the T1w/T1ce/T2w/FLAIR tuple
the model consumes (`nn.Embedding(4, 768)`), so walking them would cost time and
buy nothing.

Output: data/bind_series_manifest.csv
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from session_keys import BIND_ROOT, SITES, session_from_bids_dir, subject_from_bids_dir

OUTPUT_CSV = Path(__file__).parent / "data" / "bind_series_manifest.csv"

# Sidecar fields kept verbatim. The numeric ones matter most: I0004 has its
# SeriesDescription scrubbed to "Series 5", so TR/TE/TI are the only signal that
# survives de-identification at both sites.
SIDECAR_FIELDS = [
    "SeriesDescription",
    "ProtocolName",
    "StudyDescription",
    "SeriesNumber",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "FlipAngle",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Manufacturer",
    "MagneticFieldStrength",
]


def parse_bids_entities(filename):
    """
    Split a BIDS filename into its suffix and entity dict.

    `sub-1_ses-2_acq-SE_run-1_T1w.nii.gz` -> ('T1w', {'acq': 'SE', 'run': '1'}).
    Measured across 1,200 subjects: the only entities present in BIND's anat/ are
    `acq` and `run`. There is no `ce-` entity anywhere, so contrast state cannot be
    read off the filename.
    """
    stem = filename[: -len(".nii.gz")] if filename.endswith(".nii.gz") else Path(filename).stem

    parts = stem.split("_")
    suffix = parts[-1]

    entities = {}
    for part in parts[:-1]:
        if "-" in part:
            key, _, value = part.partition("-")
            entities[key] = value

    return suffix, entities


def acquisition_plane(orientation):
    """
    Derive the acquisition plane from the DICOM ImageOrientationPatient cosines.

    The slice normal is the cross product of the two in-plane direction vectors;
    whichever patient axis it aligns with most strongly names the plane. Returns
    "" when the field is missing, which is the same thing MR-RATE's plane_priority
    treats as lowest priority.
    """
    if orientation is None:
        return ""

    try:
        cosines = np.asarray(orientation, dtype=float).ravel()
    except (TypeError, ValueError):
        return ""

    if cosines.size != 6 or not np.all(np.isfinite(cosines)):
        return ""

    normal = np.cross(cosines[:3], cosines[3:])

    norm = np.linalg.norm(normal)
    if norm == 0:
        return ""

    # DICOM patient axes are (x=L/R, y=A/P, z=S/I): a normal along z is axial.
    return ["SAGITTAL", "CORONAL", "AXIAL"][int(np.argmax(np.abs(normal / norm)))]


def is_derived_series(image_type, suffix, entities):
    """
    Flag series that are reconstructions rather than acquisitions.

    Mirrors MR-RATE's is_derived/is_localizer filter: DERIVED/SECONDARY ImageType,
    minimum-intensity projections, and the phase half of an SWI pair.
    """
    image_type = [str(v).upper() for v in (image_type or [])]

    if "DERIVED" in image_type or "SECONDARY" in image_type or "PROJECTION IMAGE" in image_type:
        return True

    if entities.get("acq", "").lower() in {"mip", "mnip"}:
        return True

    if entities.get("part", "").lower() == "phase":
        return True

    return suffix in {"angio", "pwi", "swi"}


def read_series(nii_path):
    """
    Build one manifest row for a single NIfTI series.

    The NIfTI header is read but the voxels never are -- `nib.load` is lazy, so
    touching `.header`/`.shape` costs a header read, not a decompress of a
    quarter-gigabyte volume.
    """
    nii_path = Path(nii_path)
    filename = nii_path.name

    suffix, entities = parse_bids_entities(filename)

    json_path = nii_path.with_name(filename[: -len(".nii.gz")] + ".json")
    sidecar = {}
    if json_path.is_file():
        try:
            with open(json_path) as handle:
                sidecar = json.load(handle)
        except (json.JSONDecodeError, OSError):
            sidecar = {}

    try:
        header = nib.load(str(nii_path)).header
        shape = tuple(int(v) for v in header.get_data_shape()[:3])
        spacing = tuple(round(float(v), 4) for v in header.get_zooms()[:3])
    except Exception:
        shape, spacing = None, None

    row = {
        # .../Imaging/{site}/BIDS/sub-X/ses-Y/anat/file.nii.gz
        "site": nii_path.parents[4].name,
        "patient_id": subject_from_bids_dir(nii_path.parents[2].name),
        "session_id": session_from_bids_dir(nii_path.parents[1].name),
        "datatype": nii_path.parent.name,
        "nii_path": str(nii_path),
        "json_path": str(json_path) if json_path.is_file() else "",
        "bids_suffix": suffix,
        "acq": entities.get("acq", ""),
        "run": entities.get("run", ""),
        "part": entities.get("part", ""),
        "shape": str(shape) if shape else "",
        "spacing_mm": str(spacing) if spacing else "",
        "acquisition_plane": acquisition_plane(sidecar.get("ImageOrientationPatientDICOM")),
        "is_derived": is_derived_series(sidecar.get("ImageType"), suffix, entities),
    }

    for field in SIDECAR_FIELDS:
        row[field] = sidecar.get(field)

    return row


def scan_subject(subject_dir):
    """Return manifest rows for every anat series under one subject."""
    rows = []

    for session_dir in sorted(Path(subject_dir).glob("ses-*")):
        anat_dir = session_dir / "anat"
        if not anat_dir.is_dir():
            continue

        for nii_path in sorted(anat_dir.glob("*.nii.gz")):
            try:
                rows.append(read_series(nii_path))
            except Exception as error:
                print(f"Warning: failed on {nii_path}: {error}")

    return rows


def scan_site(site, workers, limit=None):
    """Walk every subject of one site in parallel."""
    bids_root = Path(BIND_ROOT) / "Imaging" / site / "BIDS"

    print(f"Listing subjects under {bids_root} ...")
    subjects = sorted(d for d in bids_root.glob("sub-*") if d.is_dir())

    if limit:
        subjects = subjects[: limit]

    print(f"{site}: {len(subjects):,} subjects")

    rows = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan_subject, str(d)) for d in subjects]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Scanning {site}", unit="subj"
        ):
            rows.extend(future.result())

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    parser.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 8)))
    parser.add_argument(
        "--limit", type=int, default=None, help="Only scan the first N subjects per site (smoke test)."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    rows = []
    for site in args.sites:
        rows.extend(scan_site(site, args.workers, args.limit))

    manifest = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"\nSaved {len(manifest):,} series to {args.output}")
    print(f"  sessions: {manifest.groupby(['site', 'session_id']).ngroups:,}")
    print(f"  patients: {manifest.groupby(['site', 'patient_id']).ngroups:,}")
    print(f"\nBIDS suffix counts:\n{manifest['bids_suffix'].value_counts().head(15)}")


if __name__ == "__main__":
    main()
