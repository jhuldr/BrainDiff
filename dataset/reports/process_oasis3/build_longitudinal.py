#!/usr/bin/env python3
"""
Phase 5: emit the two S3-schema CSVs for OASIS-3.

    OUTPUT/longitudinal.csv   the S3 `image.csv` schema, for this site only
    OUTPUT/pairs.csv          the S3 `main.csv` schema, for this site only

Deliberately does NOT touch /home/data/BRAIN_DIFF_S3/{main,image}.csv. Merging the
five sites into the top-level pair is left to the caller.

Pairing reimplements create_timepoint_pairs from
process_mrrate/revise_dataframes.ipynb cells 15-17, which is what built the
existing 40,376 S3 pairs: per subject, sort by Timepoint, emit every consecutive
pair AND every (timepoint 1, timepoint k) pair, deduped. Keeping the rule identical
is the point -- a different rule here would make OASIS-3 pairs a different kind of
object from the other four sites' without anything in the CSV saying so.

Unlike the other sites, OASIS-3 sessions carry real dates: the `ses-dXXXX` label is
days since the subject's entry, so `duration_days` is a genuine interval rather
than an index difference. It is recorded but NOT filtered on, so that any later
gap threshold is a visible choice rather than something baked in here.

    python build_longitudinal.py
"""

import argparse
from pathlib import Path

import pandas as pd

from paths import (
    CSV_MODALITIES, MODALITIES, OUTPUT_ROOT, aligned_path, output_dir,
)


def create_timepoint_pairs(sessions):
    """Consecutive pairs plus baseline-with-every-later, deduped.

    Mirrors process_mrrate/revise_dataframes.ipynb::create_timepoint_pairs. The
    set is what makes the two rules composable: for a subject with timepoints
    1,2,3 the consecutive rule gives (1,2),(2,3) and the baseline rule gives
    (1,2),(1,3); (1,2) must not appear twice.
    """
    pairs = []

    for _, group in sessions.groupby("StudyUID", sort=False):
        group = group.sort_values("Timepoint").reset_index(drop=True)

        if len(group) < 2:
            continue

        seen = set()

        def add(first, second):
            key = (first["UID"], second["UID"])
            if key in seen:
                return
            seen.add(key)
            pairs.append(
                {
                    "UID_1": first["UID"],
                    "UID_2": second["UID"],
                    "duration_days": int(second["days"]) - int(first["days"]),
                }
            )

        for i in range(len(group) - 1):
            add(group.iloc[i], group.iloc[i + 1])

        baseline = group.iloc[0]
        for i in range(1, len(group)):
            add(baseline, group.iloc[i])

    return pd.DataFrame(pairs, columns=["UID_1", "UID_2", "duration_days"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    out_dir = output_dir(args.output_root)
    sessions_csv = args.sessions_csv or (out_dir / "sessions.csv")
    sessions = pd.read_csv(sessions_csv, dtype={"subject_id": str, "session_label": str})

    # Modality cells qc.py rejected. This script rebuilds cells from what is on
    # disk, and qc.py deliberately leaves rejected volumes there for inspection,
    # so without honouring this record a re-run would restore every one of them.
    blanked = set()
    blanked_path = out_dir / "blanked.csv"
    if blanked_path.is_file():
        record = pd.read_csv(blanked_path, dtype={"subject_id": str})
        blanked = {
            (r.subject_id, int(r.Timepoint), r.modality) for r in record.itertuples()
        }
        print(f"Honouring {len(blanked):,} cells blanked by qc.py ({blanked_path})")

    # A session counts only if its volumes actually landed. Phase 4 records its
    # failures, but the file system is the authority -- a row pointing at a
    # missing path would blow up in DiffPairDataset's constructor, not here.
    rows = []
    for _, row in sessions.iterrows():
        subject_id, timepoint = row["subject_id"], int(row["Timepoint"])

        written = {}
        for modality in MODALITIES:
            if (subject_id, timepoint, modality) in blanked:
                written[modality] = ""
                continue

            path = aligned_path(subject_id, timepoint, modality, args.output_root)
            written[modality] = str(path) if path.is_file() else ""

        if not written["T1w"]:
            continue

        rows.append(
            {
                "StudyUID": row["StudyUID"],
                "Timepoint": timepoint,
                "T1w": written["T1w"],
                "T1ce": "",            # OASIS-3 ships no post-contrast series
                "T2w": written["T2w"],
                "FLAIR": written["FLAIR"],
                "UID": row["UID"],
                "subject_id": subject_id,
                "session_label": row["session_label"],
                "days": int(row["days"]),
            }
        )

    # The first seven columns are S3's image.csv schema, in its order. The last
    # three are extras: diff_pairs.py indexes on UID and reads only the four
    # modality columns, so trailing columns are inert.
    longitudinal = pd.DataFrame(
        rows,
        columns=["StudyUID", "Timepoint", *CSV_MODALITIES, "UID",
                 "subject_id", "session_label", "days"],
    )

    if longitudinal.empty:
        raise SystemExit(
            f"No written volumes found under {args.output_root}/ALIGNED. "
            "Run DataProcessing/align_to_mni.py first."
        )

    # Timepoints must stay contiguous from 1: create_timepoint_pairs' baseline
    # rule names timepoint 1 explicitly, and diff_pairs.py::patient_of recovers
    # the subject by stripping the trailing index. If Phase 4 dropped a middle
    # session, renumber so both stay true.
    longitudinal = longitudinal.sort_values(["StudyUID", "Timepoint"]).reset_index(drop=True)
    renumbered = longitudinal.groupby("StudyUID").cumcount() + 1

    if not renumbered.equals(longitudinal["Timepoint"]):
        moved = int((renumbered != longitudinal["Timepoint"]).sum())
        print(f"Renumbering {moved:,} timepoints to stay contiguous after dropped sessions")
        longitudinal["Timepoint"] = renumbered
        longitudinal["UID"] = (
            longitudinal["StudyUID"] + "_" + longitudinal["Timepoint"].astype(str)
        )

    pairs = create_timepoint_pairs(longitudinal)

    out_dir.mkdir(parents=True, exist_ok=True)
    longitudinal_path = out_dir / "longitudinal.csv"
    pairs_path = out_dir / "pairs.csv"

    longitudinal.to_csv(longitudinal_path, index=False)
    pairs.to_csv(pairs_path, index=False)

    per_subject = longitudinal.groupby("StudyUID").size()

    # Every number below is counted from the built tree, not estimated.
    print("\n=== longitudinal.csv ===")
    print(f"  volumes rows      : {len(longitudinal):,}")
    print(f"  subjects          : {longitudinal['StudyUID'].nunique():,}")
    print(f"  timepoints/subject: mean {per_subject.mean():.2f}, max {per_subject.max()}")
    for modality in CSV_MODALITIES:
        present = int((longitudinal[modality] != "").sum())
        print(f"    {modality:6s} {present:6,} / {len(longitudinal):,}")

    print("\n=== pairs.csv ===")
    print(f"  pairs             : {len(pairs):,}")

    if len(pairs):
        gaps = pairs["duration_days"]
        print(f"  subjects paired   : {per_subject[per_subject > 1].size:,}")
        print(f"  duration_days     : median {gaps.median():.0f}, "
              f"IQR {gaps.quantile(0.25):.0f}-{gaps.quantile(0.75):.0f}, "
              f"min {gaps.min():.0f}, max {gaps.max():.0f}")
        print(f"  under 180 days    : {int((gaps < 180).sum()):,} "
              f"({100 * (gaps < 180).mean():.1f}%)  <- not filtered, recorded only")

    print(f"\nWrote {longitudinal_path}")
    print(f"Wrote {pairs_path}")
    print("\nTop-level BRAIN_DIFF_S3/{main,image}.csv deliberately untouched.")


if __name__ == "__main__":
    main()
