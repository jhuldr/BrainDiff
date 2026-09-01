#!/usr/bin/env python3
"""
Build consecutive same-patient session pairs from BIND reports.

The analogue of process_mrrate/process_long.ipynb cell 2/12: sort each patient's
studies by date and emit every consecutive pair. Nothing about imaging is decided
here -- this stage only answers "which two sessions of which patient, and what did
each report say". generate_dual_bind.py then intersects that with what is actually
on disk.

Output: data/bind_longitudinal_meta.csv
    patient_uid, site, study_uid1, study_uid2, duration,
    report1, report2, exam_type1, exam_type2, had_comparison1, had_comparison2
"""

import argparse
from pathlib import Path

import pandas as pd

from report_sections import has_contrast, is_mr_brain, strip_comparison
from session_keys import (
    BIND_ROOT,
    SITES,
    canonical_patient_series,
    canonical_session_series,
    load_patient_merge_map,
)

OUTPUT_CSV = Path(__file__).parent / "data" / "bind_longitudinal_meta.csv"

# Rows where the de-identification pipeline had nothing to hand over. Every one of
# these has "not available" in Report_txt as well, so there is no text to recover.
_MISSING = {"not available", "n/a", "none", ""}


def load_reports(site):
    """Load one site's reports, keyed by canonical session id."""
    reports = pd.read_csv(
        f"{BIND_ROOT}/Imaging/{site}/Clinical/{site}_reports.csv", dtype=str
    )

    reports["session_uid"] = canonical_session_series(reports["session_id"], site)
    reports["site"] = site

    before = len(reports)

    missing = (
        reports["Report_txt"].fillna("").str.strip().str.lower().isin(_MISSING)
        | reports["Type"].fillna("").str.strip().str.lower().isin(_MISSING)
    )
    reports = reports[~missing].copy()

    print(f"  {site}: {before:,} rows -> {len(reports):,} with usable text "
          f"({before - len(reports):,} dropped as 'not available')")

    return reports


def load_dates(site):
    """
    Load study dates, which live only in the demographics table.

    ShiftedStudyDate is date-shifted per patient, but the shift is constant within
    a patient, so intervals between that patient's studies are preserved -- which
    is all `duration` needs.
    """
    demographics = pd.read_csv(
        f"{BIND_ROOT}/Imaging/{site}/Clinical/{site}_demographics.csv", dtype=str
    )

    demographics["session_uid"] = canonical_session_series(
        demographics["Session_id"], site
    )

    dates = demographics[["session_uid", "ShiftedStudyDate"]].dropna()
    dates = dates.drop_duplicates(subset=["session_uid"])
    dates["study_date"] = pd.to_datetime(dates["ShiftedStudyDate"], errors="coerce")

    return dates[["session_uid", "study_date"]].dropna(subset=["study_date"])


def build_site_studies(site):
    """Return one row per usable MR-brain session for a site."""
    reports = load_reports(site)
    dates = load_dates(site)

    studies = reports.merge(dates, on="session_uid", how="left")

    with_date = studies["study_date"].notna()
    print(f"  {site}: {with_date.sum():,}/{len(studies):,} sessions resolved a study date")
    studies = studies[with_date].copy()

    # Keep structural brain MRI only. `Type` is authoritative for every surviving
    # row; the report body is only consulted for the contrast flag.
    studies["exam_type"] = studies["Type"].fillna("").str.upper().str.strip()

    is_brain = studies["exam_type"].map(is_mr_brain)
    print(f"  {site}: {is_brain.sum():,} MR-brain sessions of {len(studies):,}")
    studies = studies[is_brain].copy()

    stripped = studies["Report_txt"].map(strip_comparison)
    studies["report"] = stripped.map(lambda pair: pair[0])
    studies["had_comparison"] = stripped.map(lambda pair: pair[1])

    studies["study_contains_contrast"] = [
        has_contrast(exam, text)
        for exam, text in zip(studies["exam_type"], studies["Report_txt"])
    ]

    merge_map = load_patient_merge_map(site)
    studies["patient_uid"] = canonical_patient_series(
        studies["bdsp_patient_id"], merge_map
    )

    if merge_map:
        remapped = (studies["patient_uid"] != studies["bdsp_patient_id"].str.strip()).sum()
        print(f"  {site}: {remapped:,} rows remapped through the patient-merge table")

    return studies[
        [
            "site",
            "patient_uid",
            "session_uid",
            "study_date",
            "report",
            "exam_type",
            "had_comparison",
            "study_contains_contrast",
        ]
    ]


def pair_consecutive(studies):
    """
    Emit every consecutive (prior, current) pair per patient.

    Same rule as MR-RATE: one study per patient per date (ties broken by session
    id), then adjacent timepoints only. Non-adjacent pairs would let a report
    describe interval change the prior image cannot account for.
    """
    studies = studies.sort_values(["patient_uid", "study_date", "session_uid"])
    studies = studies.drop_duplicates(subset=["patient_uid", "study_date"], keep="first")

    pairs = []

    for patient_uid, group in studies.groupby("patient_uid", sort=False):
        if len(group) < 2:
            continue

        group = group.reset_index(drop=True)

        for i in range(len(group) - 1):
            prior, current = group.iloc[i], group.iloc[i + 1]

            pairs.append(
                {
                    "patient_uid": patient_uid,
                    "site": prior["site"],
                    "study_uid1": prior["session_uid"],
                    "study_uid2": current["session_uid"],
                    # Days, matching S4's `duration` column.
                    "duration": (current["study_date"] - prior["study_date"]).days,
                    "report1": prior["report"],
                    "report2": current["report"],
                    "exam_type1": prior["exam_type"],
                    "exam_type2": current["exam_type"],
                    "had_comparison1": prior["had_comparison"],
                    "had_comparison2": current["had_comparison"],
                }
            )

    return pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--min-duration",
        type=int,
        default=1,
        help="Drop pairs closer together than this many days (same-day rescans).",
    )
    args = parser.parse_args()

    frames = []
    for site in args.sites:
        print(f"\n=== {site} ===")
        frames.append(build_site_studies(site))

    studies = pd.concat(frames, ignore_index=True)

    pairs = pair_consecutive(studies)

    before = len(pairs)
    pairs = pairs[pairs["duration"] >= args.min_duration].copy()
    print(f"\nDropped {before - len(pairs):,} pairs under {args.min_duration} day(s) apart")

    pairs = pairs.dropna(subset=["report1", "report2"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)

    print(f"\nSaved {len(pairs):,} pairs to {args.output}")
    print(f"  patients: {pairs['patient_uid'].nunique():,}")
    print(f"  per site: {pairs['site'].value_counts().to_dict()}")
    print(f"  duration days: median {pairs['duration'].median():.0f}, "
          f"mean {pairs['duration'].mean():.0f}")
    print(f"  had a COMPARISON section stripped: "
          f"{(pairs['had_comparison1'] | pairs['had_comparison2']).mean():.1%} of pairs")


if __name__ == "__main__":
    main()
