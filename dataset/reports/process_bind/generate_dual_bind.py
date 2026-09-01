#!/usr/bin/env python3
"""
Select the T1w/T1ce/T2w/FLAIR tuple for each BIND session in a longitudinal pair.

The analogue of process_mrrate/generate_single.py, which picked one series per
modality per study from MR-RATE's `classified_modality` / `is_center_modality`
columns. BIND ships no such columns, so classification happens here.

Classification uses the BIDS suffix, NOT the sidecar TR/TE/TI. That is a
measurement, not a preference. A hand-written parameter rule
(TI>=1.5 -> FLAIR, TE<0.03 -> T1w, else T2w) agrees with dcm2niix's suffix on only
97.09% of 549,295 non-derived series, and inspecting the 16,002 disagreements
shows the suffix is right and the rule is wrong in the large majority:

    T1w -> T2w   14,106   "Post Ax T1 FLAIR PROPELLER" -- T1-weighted, long TE
    T1w -> FLAIR  1,389   T1-FLAIR/MPRAGE with TI just over the threshold
    T2w -> T1w      272   STIR, a short-TI fat-suppressed T2

So the parameter rule was dropped. What the sidecar is still used for: the
acquisition plane, the 2D/3D flag and slice thickness (to pick the registration
target), SeriesNumber (ordering), and SeriesDescription (contrast).

Output: data/bind_longitudinal_data.csv, one row per session:
    session_uid, site, patient_uid, t1w, t1ce, t2w, flair, centered_image, t1ce_source
Values are absolute .nii.gz paths.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from report_sections import has_contrast

MANIFEST_CSV = Path(__file__).parent / "data" / "bind_series_manifest.csv"
PAIRS_CSV = Path(__file__).parent / "data" / "bind_longitudinal_meta.csv"
OUTPUT_CSV = Path(__file__).parent / "data" / "bind_longitudinal_data.csv"
OUTPUT_PAIRS_CSV = Path(__file__).parent / "data" / "bind_longitudinal_meta_paired.csv"

MODALITY_SUFFIXES = ("T1w", "T2w", "FLAIR")

# STIR is a short-TI fat-suppressed T2 and MAGiC is a synthetic-contrast
# reconstruction; neither belongs in a plain T2w channel.
EXCLUDED_ACQ = {"STIR", "MAGIC"}

# Contrast markers in SeriesDescription. Intact at I0001; I0004 has the field
# scrubbed to "Series 5", so nothing matches there and the run-order fallback
# carries that site.
_POST_RE = re.compile(r"(?<![A-Za-z])(POST|GAD|GD)(?![A-Za-z])|\+\s*C(?![A-Za-z])", re.IGNORECASE)
_PRE_RE = re.compile(r"(?<![A-Za-z])(PRE|NON[- ]?CON)(?![A-Za-z])", re.IGNORECASE)

_PLANE_RANK = {"AXIAL": 1, "SAGITTAL": 2, "CORONAL": 3}


def plane_priority(plane):
    """
    Rank acquisition planes: axial > sagittal > coronal, unknown last.

    Same ordering as MR-RATE's generate_single.plane_priority, minus its
    center-modality term -- the centre is chosen separately here.
    """
    return _PLANE_RANK.get(str(plane).upper(), 4)


def load_series(manifest_csv):
    """Load the manifest and keep only usable structural series."""
    manifest = pd.read_csv(
        manifest_csv,
        dtype={"session_id": str, "patient_id": str, "run": str, "acq": str},
        low_memory=False,
    )

    before = len(manifest)

    series = manifest[
        (~manifest["is_derived"])
        & manifest["bids_suffix"].isin(MODALITY_SUFFIXES)
        & (~manifest["acq"].fillna("").str.upper().isin(EXCLUDED_ACQ))
    ].copy()

    print(f"Manifest: {before:,} series -> {len(series):,} structural non-derived")

    series["SeriesNumber"] = pd.to_numeric(series["SeriesNumber"], errors="coerce")
    series["SliceThickness"] = pd.to_numeric(series["SliceThickness"], errors="coerce")
    series["is_3d"] = series["MRAcquisitionType"].astype(str).str.upper().eq("3D")
    series["plane_rank"] = series["acquisition_plane"].map(plane_priority)

    description = series["SeriesDescription"].fillna("")
    series["is_post"] = description.map(lambda v: bool(_POST_RE.search(v)))
    series["is_pre"] = description.map(lambda v: bool(_PRE_RE.search(v)))

    series["session_key"] = series["site"] + "/" + series["session_id"]

    return series


def pick_best(candidates):
    """
    Pick one representative series: best plane, then earliest SeriesNumber.

    Mirrors MR-RATE's pick_best_plane tiebreak.
    """
    if candidates.empty:
        return None

    ordered = candidates.sort_values(
        ["plane_rank", "SeriesNumber"], na_position="last"
    )

    return ordered.iloc[0]


def pick_center(t1w_series):
    """
    Pick the registration target: the highest-resolution T1w in the session.

    Every other modality is rigidly coregistered onto this image and its affine to
    MNI is the one applied to all four, so resolution here sets the ceiling for the
    whole session. 3D wins over 2D (measured median slice thickness 1 mm vs 5 mm),
    then thinnest slice, then plane, then earliest series.

    Contrast status is deliberately ignored -- a post-contrast MPRAGE is still the
    better geometric target than a 5 mm 2D spin echo.
    """
    if t1w_series.empty:
        return None

    ordered = t1w_series.sort_values(
        ["is_3d", "SliceThickness", "plane_rank", "SeriesNumber"],
        ascending=[False, True, True, True],
        na_position="last",
    )

    return ordered.iloc[0]


def select_t1_pair(t1w_series, study_has_contrast):
    """
    Split the T1w series into (pre-contrast, post-contrast, source).

    1. `text`      -- SeriesDescription says POST / GAD / +C. Reliable, I0001 only.
    2. `run_order` -- only when the report says the exam was performed with AND
       without contrast: lowest SeriesNumber is pre, highest is post. This is
       MR-RATE's select_modality_no_and_with_contrast heuristic. It is an
       inference from acquisition order, not a label, which is why every row
       records which branch produced it.
    3. `none`      -- one T1w, no t1ce.
    """
    if t1w_series.empty:
        return None, None, "none"

    post = t1w_series[t1w_series["is_post"]]
    pre = t1w_series[t1w_series["is_pre"]]
    untagged = t1w_series[~t1w_series["is_post"] & ~t1w_series["is_pre"]]

    if not post.empty:
        # Prefer an explicitly pre-contrast series for the plain T1w channel;
        # fall back to untagged series acquired before the first post series.
        baseline = pre if not pre.empty else untagged
        return pick_best(baseline), pick_best(post), "text"

    if study_has_contrast and len(t1w_series) > 1:
        numbered = t1w_series.dropna(subset=["SeriesNumber"])

        if len(numbered) > 1:
            first, last = numbered["SeriesNumber"].min(), numbered["SeriesNumber"].max()

            if first != last:
                return (
                    pick_best(numbered[numbered["SeriesNumber"] == first]),
                    pick_best(numbered[numbered["SeriesNumber"] == last]),
                    "run_order",
                )

    return pick_best(t1w_series), None, "none"


def select_session(session_series, study_has_contrast):
    """Build one output row for a session, or None if it has no usable T1w."""
    by_suffix = {
        suffix: group for suffix, group in session_series.groupby("bids_suffix")
    }

    t1w_series = by_suffix.get("T1w", session_series.iloc[0:0])

    center = pick_center(t1w_series)
    if center is None:
        return None

    t1w, t1ce, source = select_t1_pair(t1w_series, study_has_contrast)
    t2w = pick_best(by_suffix.get("T2w", session_series.iloc[0:0]))
    flair = pick_best(by_suffix.get("FLAIR", session_series.iloc[0:0]))

    path = lambda row: None if row is None else row["nii_path"]

    return {
        "site": center["site"],
        "session_uid": center["session_id"],
        "patient_uid": center["patient_id"],
        "t1w": path(t1w),
        "t1ce": path(t1ce),
        "t2w": path(t2w),
        "flair": path(flair),
        "centered_image": center["nii_path"],
        "center_is_3d": bool(center["is_3d"]),
        "center_slice_mm": center["SliceThickness"],
        "t1ce_source": source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--pairs", type=Path, default=PAIRS_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-pairs", type=Path, default=OUTPUT_PAIRS_CSV)
    args = parser.parse_args()

    series = load_series(args.manifest)
    pairs = pd.read_csv(args.pairs, dtype={"study_uid1": str, "study_uid2": str})

    print(f"Pairs: {len(pairs):,}")

    # Only sessions that take part in a pair are worth selecting for.
    wanted = pd.concat(
        [
            pairs[["site", "study_uid1"]].rename(columns={"study_uid1": "session_uid"}),
            pairs[["site", "study_uid2"]].rename(columns={"study_uid2": "session_uid"}),
        ]
    ).drop_duplicates()
    wanted["session_key"] = wanted["site"] + "/" + wanted["session_uid"]

    wanted_keys = set(wanted["session_key"])
    print(f"Sessions referenced by pairs: {len(wanted_keys):,}")

    series = series[series["session_key"].isin(wanted_keys)].copy()
    print(f"  with imaging on disk: {series['session_key'].nunique():,}")

    # A session's contrast flag comes from its own report, reachable via either
    # pair column since a session appears as the current study of one pair and the
    # prior of the next.
    exam_lookup = pd.concat(
        [
            pairs[["site", "study_uid1", "exam_type1"]].rename(
                columns={"study_uid1": "session_uid", "exam_type1": "exam_type"}
            ),
            pairs[["site", "study_uid2", "exam_type2"]].rename(
                columns={"study_uid2": "session_uid", "exam_type2": "exam_type"}
            ),
        ]
    ).drop_duplicates(subset=["site", "session_uid"])

    exam_lookup["has_contrast"] = exam_lookup["exam_type"].map(has_contrast)
    contrast_flag = dict(
        zip(
            exam_lookup["site"] + "/" + exam_lookup["session_uid"],
            exam_lookup["has_contrast"],
        )
    )

    rows = []
    for session_key, group in series.groupby("session_key", sort=False):
        row = select_session(group, contrast_flag.get(session_key, False))
        if row is not None:
            rows.append(row)

    selected = pd.DataFrame(rows)
    print(f"  with a usable T1w (center modality): {len(selected):,}")

    # Keep a pair only if both timepoints survived selection.
    have = set(selected["site"] + "/" + selected["session_uid"])
    keep = pairs.apply(
        lambda r: f"{r['site']}/{r['study_uid1']}" in have
        and f"{r['site']}/{r['study_uid2']}" in have,
        axis=1,
    )
    final_pairs = pairs[keep].copy()

    # Drop sessions orphaned by that filter.
    used = set(final_pairs["site"] + "/" + final_pairs["study_uid1"]) | set(
        final_pairs["site"] + "/" + final_pairs["study_uid2"]
    )
    selected = selected[(selected["site"] + "/" + selected["session_uid"]).isin(used)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    final_pairs.to_csv(args.output_pairs, index=False)

    print("\n=== Yield ===")
    print(f"  candidate pairs          : {len(pairs):,}")
    print(f"  both timepoints selected : {len(final_pairs):,}")
    print(f"  unique sessions to process: {len(selected):,}")
    print(f"  patients                 : {final_pairs['patient_uid'].nunique():,}")

    print("\n=== Modality coverage (selected sessions) ===")
    for column in ("t1w", "t1ce", "t2w", "flair"):
        print(f"  {column:6s}: {selected[column].notna().sum():,} "
              f"({selected[column].notna().mean():.1%})")

    print(f"\n  centre is 3D: {selected['center_is_3d'].mean():.1%}, "
          f"median slice {selected['center_slice_mm'].median():.2g} mm")
    print(f"\n=== t1ce_source ===\n{selected['t1ce_source'].value_counts()}")
    print(f"\nby site:\n{pd.crosstab(selected['site'], selected['t1ce_source'])}")
    print(f"\nSaved {args.output} and {args.output_pairs}")


if __name__ == "__main__":
    main()
