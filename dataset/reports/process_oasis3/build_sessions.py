#!/usr/bin/env python3
"""
Phase 2: choose one file per modality per session, and assign timepoints.

OASIS-3 gives several candidates per modality per session -- 2,656 `run-NN` files
across the tree, and T2w split between `_T2w` (796) and `acq-TSE_T2w` (941). The
selection rule, in order:

    1. If any candidate carries a run-NN tag, keep only the highest NN.
       run-02 beats run-01. This is an explicit instruction and it OVERRIDES the
       geometry heuristic below, even when the lower run has better spacing.
    2. Among what survives, prefer the smallest max voxel spacing. This is what
       settles _T2w vs acq-TSE_T2w: measured on the current tree, T1w is 1 mm
       isotropic 3D, while T2w is mostly 1x1x4 mm 2D TSE and FLAIR is 5-6 mm.
    3. Tie-break on voxel count, then filename, so the choice is reproducible.

Every choice is logged with the rule that made it, so rule 1's cost is visible if
it ever picks a worse volume than rule 2 would have.

    python build_sessions.py

Writes OUTPUT/sessions.csv.
"""

import argparse
import re
import uuid
from pathlib import Path

import nibabel as nib
import pandas as pd
from tqdm import tqdm

from paths import MODALITIES, OUTPUT_ROOT, RAW_ROOT, output_dir

# uuid5, not the uuid4 that dataset3D/create_longitudinal.py uses. A rerun of that
# script mints fresh StudyUIDs and orphans every row written by the previous one;
# deriving the UID from the subject makes this phase idempotent.
UID_NAMESPACE = uuid.NAMESPACE_URL

RUN_RE = re.compile(r"_run-(\d+)_")
DAY_RE = re.compile(r"^ses-d(\d+)$")


def study_uid(subject_id):
    return str(uuid.uuid5(UID_NAMESPACE, f"oasis3/{subject_id}"))


def modality_of(filename):
    """Map a BIDS anat filename to one of MODALITIES, or None.

    Matches on the BIDS suffix -- the last underscore-separated token before the
    extension -- so `acq-TSE_T2w` and `run-02_T1w` classify correctly and
    unrelated suffixes (T2star, angio, swi) fall through to None.
    """
    stem = filename.replace(".nii.gz", "")
    suffix = stem.split("_")[-1]

    return suffix if suffix in MODALITIES else None


def run_number(filename):
    match = RUN_RE.search(filename)

    return int(match.group(1)) if match else None


def geometry(path):
    """(max spacing, voxel count) from the header alone -- no voxels are read."""
    header = nib.load(str(path)).header
    zooms = header.get_zooms()[:3]
    shape = header.get_data_shape()[:3]

    return float(max(zooms)), int(shape[0] * shape[1] * shape[2])


def choose(candidates):
    """Apply the three-step rule. Returns (path, reason) or (None, reason)."""
    if not candidates:
        return None, "absent"

    runs = {path: run_number(path.name) for path in candidates}
    tagged = [path for path, run in runs.items() if run is not None]

    reason = "single"

    if tagged:
        highest = max(runs[path] for path in tagged)
        shortlist = [path for path in tagged if runs[path] == highest]
        # Rule 1 is a filter, not a full ordering: if only some candidates carry a
        # run tag, the untagged ones are dropped rather than competing on spacing.
        reason = f"run-{highest:02d}"
    else:
        shortlist = list(candidates)

    if len(shortlist) == 1 and reason == "single" and len(candidates) > 1:
        reason = "only-candidate"

    if len(shortlist) > 1:
        reason = f"{reason}+spacing" if reason.startswith("run-") else "spacing"

    def sort_key(path):
        max_spacing, voxels = geometry(path)
        return (max_spacing, -voxels, path.name)

    shortlist.sort(key=sort_key)

    return shortlist[0], reason


def scan_session(session_dir):
    """Pick one file per modality for one sub-*/ses-* directory."""
    anat = session_dir / "anat"

    if not anat.is_dir():
        return None

    by_modality = {modality: [] for modality in MODALITIES}

    for path in sorted(anat.glob("*.nii.gz")):
        modality = modality_of(path.name)
        if modality is not None:
            by_modality[modality].append(path)

    chosen, reasons = {}, {}
    for modality, candidates in by_modality.items():
        path, reason = choose(candidates)
        chosen[modality] = str(path) if path is not None else ""
        reasons[modality] = f"{reason}({len(candidates)})"

    return chosen, reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    session_dirs = sorted(args.raw_root.glob("sub-*/ses-*"))
    print(f"{len(session_dirs):,} sub-*/ses-* directories under {args.raw_root}")

    rows = []
    no_t1w = 0

    for session_dir in tqdm(session_dirs, desc="scanning", unit="sess"):
        result = scan_session(session_dir)
        if result is None:
            continue

        chosen, reasons = result

        # No T1w means no mask -- HD-BET is run on the T1w and its transform is
        # what carries the mask into MNI. Such a session cannot be skull-stripped
        # consistently with the rest, so it is dropped rather than half-processed.
        if not chosen["T1w"]:
            no_t1w += 1
            continue

        subject_id = session_dir.parent.name
        session_label = session_dir.name
        day_match = DAY_RE.match(session_label)

        rows.append(
            {
                "subject_id": subject_id,
                "session_label": session_label,
                "days": int(day_match.group(1)) if day_match else -1,
                "StudyUID": study_uid(subject_id),
                **{f"src_{m}": chosen[m] for m in MODALITIES},
                **{f"rule_{m}": reasons[m] for m in MODALITIES},
            }
        )

    sessions = pd.DataFrame(rows)

    if sessions.empty:
        raise SystemExit(f"No usable sessions found under {args.raw_root}")

    # Timepoint is the 1-based rank of the day offset within the subject. Ordinal
    # rather than the day itself so UIDs stay `<patient>_<small int>` for
    # dataloaders/diff_pairs.py::patient_of and the baseline-pairing rule can name
    # timepoint 1.
    sessions = sessions.sort_values(["subject_id", "days", "session_label"])
    sessions["Timepoint"] = sessions.groupby("subject_id").cumcount() + 1
    sessions["UID"] = sessions["StudyUID"] + "_" + sessions["Timepoint"].astype(str)

    columns = (
        ["subject_id", "session_label", "days", "Timepoint", "StudyUID", "UID"]
        + [f"src_{m}" for m in MODALITIES]
        + [f"rule_{m}" for m in MODALITIES]
    )
    sessions = sessions[columns].reset_index(drop=True)

    out_dir = output_dir(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sessions.csv"
    sessions.to_csv(out_path, index=False)

    per_subject = sessions.groupby("subject_id").size()

    print(f"\n=== sessions.csv ===")
    print(f"  sessions          : {len(sessions):,}")
    print(f"  subjects          : {sessions['subject_id'].nunique():,}")
    print(f"  dropped, no T1w   : {no_t1w:,}")
    print(f"  sessions/subject  : mean {per_subject.mean():.2f}, max {per_subject.max()}")
    print(f"  subjects with >1  : {(per_subject > 1).sum():,} "
          f"({100 * (per_subject > 1).mean():.1f}%)  <- pairable")

    print("\n  present per modality:")
    for modality in MODALITIES:
        present = (sessions[f"src_{modality}"] != "").sum()
        print(f"    {modality:6s} {present:6,} / {len(sessions):,}")

    print("\n  selection rule used (count of candidates in parens):")
    for modality in MODALITIES:
        counts = sessions[f"rule_{modality}"].value_counts()
        top = ", ".join(f"{rule} x{n}" for rule, n in counts.head(6).items())
        print(f"    {modality:6s} {top}")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
