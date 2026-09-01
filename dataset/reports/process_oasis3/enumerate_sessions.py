#!/usr/bin/env python3
"""
Phase 0: list every MR session in OASIS-3 on NITRC IR.

Why this exists. /home/data/OASIS-3 was downloaded on 2026-07-13 from a manifest
(datadownload/kkpatel25_7_13_2026_19_58_9.csv) holding 1,378 experiment_ids -- one
MR session per subject. Measured on disk: 1,374 subjects, every one with exactly
one session. That yields ZERO longitudinal pairs, which is the only thing S3 wants
from this dataset. OASIS-3 publishes roughly 2,842 MR sessions; this script asks
the server for the real list instead of trusting that figure.

Credentials are prompted for and never written anywhere: no argv, no env, no file.

    python enumerate_sessions.py --username kkpatel25

Writes:
    data/oasis3_all_mr_sessions.csv   every MR session the server reports
    data/oasis3_missing_full.csv      the subset not yet on disk, all columns
    data/oasis3_missing.csv           the same subset, ONE column -- feed this one

The single-column form is not cosmetic. download_oasis_scans_bids.sh:360 reads the
manifest with

    sed 1d $INFILE | while IFS=, read -r EXPERIMENT_ID

and `read` with a single variable assigns the ENTIRE remaining line to it, commas
and all. Handed a multi-column CSV it builds URLs like
`.../subjects/CENTRAL02/experiments/CENTRAL02_E02646,CENTRAL_S05127,OAS30038,...`
and every download fails. It must get exactly one column.
"""

import argparse
import getpass
import io
import sys
from pathlib import Path

import pandas as pd
import requests

from paths import RAW_ROOT

BASE_URL = "https://www.nitrc.org/ir"
DATA_DIR = Path(__file__).resolve().parent / "data"

# xnat:mrSessionData restricts this to MR; OASIS3 also holds PET/CT sessions that
# download_oasis_scans_bids.sh would happily try to fetch T1w from and fail on.
EXPERIMENTS_PATH = (
    "/data/projects/{project}/experiments"
    "?xsiType=xnat:mrSessionData&format=csv"
    "&columns=ID,label,subject_label,date,xsiType"
)


def fetch_sessions(session, project):
    """GET the project's MR session table as a DataFrame."""
    url = BASE_URL + EXPERIMENTS_PATH.format(project=project)
    response = session.get(url, timeout=300)
    response.raise_for_status()

    frame = pd.read_csv(io.StringIO(response.text))

    if frame.empty:
        raise RuntimeError(f"{project}: server returned an empty session table")

    # XNAT names this column `label`; download_oasis_scans_bids.sh wants
    # `experiment_id` and the values are the same strings (OAS30001_MR_d0757).
    if "label" not in frame.columns:
        raise RuntimeError(
            f"{project}: no `label` column in server response; got {list(frame.columns)}"
        )

    frame = frame.rename(columns={"label": "experiment_id"})

    # NITRC types some administrative entries as xnat:mrSessionData -- measured on
    # OASIS3: `OASIS3_data_files` and `OASIS_cohort_files`, both under a pseudo
    # subject `0AS_data_files`. They are not scans, they can never download, and
    # left in they inflate the session count and show up forever as "missing".
    # A real session label is OAS<digits>_MR_d<digits>.
    real = frame["experiment_id"].astype(str).str.match(r"^OAS\d+_MR_d\d+$")
    dropped = int((~real).sum())

    if dropped:
        print(f"Dropping {dropped} non-session entries typed as MR: "
              f"{', '.join(frame.loc[~real, 'experiment_id'].astype(str)[:5])}")

    return frame[real].reset_index(drop=True)


def parse_experiment_id(experiment_id):
    """`OAS30001_MR_d0757` -> ('sub-OAS30001', 'ses-d0757').

    The trailing day token is zero-padded to 4 digits by move_to_bids() in
    download_oasis_scans_bids.sh, so pad it here too or the on-disk comparison
    below silently reports everything as missing.
    """
    parts = str(experiment_id).split("_")

    if len(parts) < 3:
        return None, None

    subject = f"sub-{parts[0]}"
    day_token = parts[-1]

    if not day_token[1:].isdigit():
        return subject, f"ses-{day_token}"

    return subject, f"ses-{day_token[0]}{int(day_token[1:]):04d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="NITRC IR username.")
    parser.add_argument("--project", default="OASIS3")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    password = getpass.getpass(f"NITRC IR password for {args.username}: ")

    with requests.Session() as http:
        http.auth = (args.username, password)

        # Establishing a JSESSION first means the credentials travel once rather
        # than on every subsequent request, and it fails loudly on a bad password
        # instead of returning an HTML login page that pd.read_csv would mangle.
        auth = http.get(f"{BASE_URL}/data/JSESSION", timeout=60)
        if auth.status_code != 200:
            sys.exit(f"Authentication failed ({auth.status_code}). Check username/password.")

        del password
        http.auth = None

        print(f"Authenticated. Querying {args.project} MR sessions...")
        sessions = fetch_sessions(http, args.project)

    parsed = sessions["experiment_id"].map(parse_experiment_id)
    sessions["bids_subject"] = [p[0] for p in parsed]
    sessions["bids_session"] = [p[1] for p in parsed]

    unparsed = sessions["bids_subject"].isna().sum()
    if unparsed:
        print(f"Warning: {unparsed} experiment_ids did not parse into sub-/ses- form")

    on_disk = sessions.apply(
        lambda r: (args.raw_root / str(r["bids_subject"]) / str(r["bids_session"])).is_dir(),
        axis=1,
    )
    missing = sessions[~on_disk].copy()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.data_dir / "oasis3_all_mr_sessions.csv"
    missing_full_path = args.data_dir / "oasis3_missing_full.csv"
    missing_path = args.data_dir / "oasis3_missing.csv"

    sessions.to_csv(all_path, index=False)
    missing.to_csv(missing_full_path, index=False)

    # ONE column, matching the manifest format the downloader was written for.
    # See the module docstring: it reads whole lines, so anything else breaks it.
    missing[["experiment_id"]].to_csv(missing_path, index=False)

    # These counts are the whole point of the phase: they replace the "~2,842"
    # publication figure with something measured before 35 GB gets downloaded.
    per_subject = sessions.groupby("bids_subject").size()

    print(f"\n=== {args.project} on NITRC IR ===")
    print(f"  MR sessions      : {len(sessions):,}")
    print(f"  subjects         : {sessions['bids_subject'].nunique():,}")
    print(f"  sessions/subject : mean {per_subject.mean():.2f}, "
          f"median {per_subject.median():.0f}, max {per_subject.max()}")
    print(f"  subjects with >1 : {(per_subject > 1).sum():,} "
          f"({100 * (per_subject > 1).mean():.1f}%)  <- the longitudinal yield")

    print(f"\n=== against {args.raw_root} ===")
    print(f"  already on disk  : {int(on_disk.sum()):,}")
    print(f"  still to download: {len(missing):,}")

    print(f"\nWrote {all_path}")
    print(f"Wrote {missing_full_path}  (all columns, for reference)")
    print(f"Wrote {missing_path}  (single column -- this is the one to download)")
    print(
        "\nNext:\n"
        f"  cd /path/to/code/BrainDiff/datadownload\n"
        f"  ./download_oasis_scans_bids.sh {missing_path} "
        f"{args.raw_root} {args.username} T1w,T2w,FLAIR"
    )


if __name__ == "__main__":
    main()
