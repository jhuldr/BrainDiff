#!/usr/bin/env python3
"""
Session and patient key normalization for BIND.

BIND spells the same session id three different ways depending on which file you
read it from, and the difference is silent -- a join on the raw strings returns
zero rows rather than an error. Everything that touches a session id goes through
this module.

Measured 2026-08-04:

    Imaging/I0001/Clinical/I0001_reports.csv    session_id   '0000156358'  padded
    Imaging/I0001/Clinical/I0001_demographics.csv Session_id '156358'      unpadded
    Imaging/I0001/Clinical/I0001_imaging_findings_De-id.csv Session_id     unpadded
    Imaging/I0001/BIDS/sub-*/ses-0000156358                                padded

    I0004 uses alphanumeric ids ('GG73ca2ea') at every one of those sites, and
    zero-padding them corrupts the key.

So the canonical form is the padded I0001 id (which is also the BIDS directory
label) and the verbatim I0004 id.
"""

import pandas as pd

SITES = ("I0001", "I0004")

BIND_ROOT = "/home/data/BIND"

# Only I0001 ships a patient-merge table.
PATIENT_MERGE_CSV = {
    "I0001": f"{BIND_ROOT}/PatientMergeHistory/I0001_patient_history_18thAugust2025.csv",
}

_I0001_WIDTH = 10


def canonical_session(value, site):
    """
    Return the canonical session id for a site, matching the BIDS `ses-` label.

    I0001 ids are numeric and zero-padded to 10 characters. I0004 ids are
    alphanumeric and are passed through untouched -- padding them corrupts them.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    if site == "I0001":
        return value.zfill(_I0001_WIDTH)

    return value


def canonical_session_series(values, site):
    """Vectorized `canonical_session` for a pandas Series."""
    values = values.astype("string").str.strip()

    if site == "I0001":
        values = values.str.zfill(_I0001_WIDTH)

    return values.replace("", pd.NA)


def session_from_bids_dir(name):
    """
    Extract the session id from a BIDS directory name.

    `ses-0000156358` -> `0000156358`. Already canonical by construction, so no
    padding is applied here.
    """
    name = str(name)

    return name[4:] if name.startswith("ses-") else name


def subject_from_bids_dir(name):
    """Extract the patient id from a BIDS directory name: `sub-111189586` -> `111189586`."""
    name = str(name)

    return name[4:] if name.startswith("sub-") else name


def load_patient_merge_map(site):
    """
    Return a {BDSPPatientID -> MergedBDSPPatientID} map for a site.

    BIND assigns a new patient id when duplicate records are merged, so the same
    person can appear under several ids. Resolving them before grouping is what
    keeps one person from landing on both sides of a patient-grouped split.
    Sites without a merge table return an empty map.
    """
    path = PATIENT_MERGE_CSV.get(site)

    if path is None:
        return {}

    merge_df = pd.read_csv(path, dtype=str)

    merge_df = merge_df.dropna(subset=["BDSPPatientID", "MergedBDSPPatientID"])

    return dict(
        zip(
            merge_df["BDSPPatientID"].str.strip(),
            merge_df["MergedBDSPPatientID"].str.strip(),
        )
    )


def canonical_patient_series(values, merge_map):
    """
    Map patient ids through a merge map, leaving unmapped ids unchanged.

    Applied once and not iterated: BIND's merge table is flat in the rows checked,
    so a single hop suffices.
    """
    values = values.astype("string").str.strip()

    if not merge_map:
        return values

    return values.map(lambda v: merge_map.get(v, v) if pd.notna(v) else v)


def _self_test():
    """Round-trip the padding rules against ids measured in the real files."""
    assert canonical_session("156358", "I0001") == "0000156358"
    assert canonical_session("0000156358", "I0001") == "0000156358"
    assert canonical_session(156358, "I0001") == "0000156358"
    assert canonical_session(" 156358 ", "I0001") == "0000156358"

    # I0004 ids are alphanumeric; padding them would corrupt the key.
    assert canonical_session("GG73ca2ea", "I0004") == "GG73ca2ea"
    assert canonical_session("GG7b491e2", "I0004") == "GG7b491e2"

    assert canonical_session(None, "I0001") is None
    assert canonical_session("", "I0001") is None

    assert session_from_bids_dir("ses-0000229635") == "0000229635"
    assert session_from_bids_dir("ses-GG7b491e2") == "GG7b491e2"
    assert subject_from_bids_dir("sub-111189586") == "111189586"

    padded = canonical_session_series(pd.Series(["156358", "213295"]), "I0001")
    assert list(padded) == ["0000156358", "0000213295"]

    passthrough = canonical_session_series(pd.Series(["GG73ca2ea"]), "I0004")
    assert list(passthrough) == ["GG73ca2ea"]

    mapped = canonical_patient_series(pd.Series(["1", "2"]), {"1": "9"})
    assert list(mapped) == ["9", "2"]

    print("session_keys self-test passed")


if __name__ == "__main__":
    _self_test()
