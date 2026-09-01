#!/usr/bin/env python3
"""
Draw the external-validation sample: N longitudinal pairs, one per patient.

Sampling one pair per patient rather than N pairs from the pool is not a style
choice. A patient contributes a median of 2 and up to 45 consecutive pairs that
share a session and a disease course, so pair-level sampling would hand the
bootstrap correlated rows and understate every confidence interval. 9,128
eligible patients against N=1,000 means independence costs nothing.

Grouping is on the MERGED patient id (the `patient_uid` already written by
build_pairs.py, which passed bdsp_patient_id through PatientMergeHistory). The
image side of the join carries the raw BIDS `sub-` id, which disagrees on 133
pairs / 54 patients -- all of them merge-explained. Grouping on the image-side id
instead would put one person on both sides of a split.

Two exclusions, both measured, applied before the draw:

  1. Pairs whose two sessions resolve to different patients even after the merge
     map is applied to both sides. Exactly 1 of 29,557 (I0001, patient 122353916,
     sessions 0000234032 / 0000000631). The reports table calls them one person
     and the BIDS tree calls them two; one of the two volumes belongs to someone
     other than the report describing it.
  2. Pairs where either report is under 300 characters (409, 1.38%). These are
     stubs, not reports, and the reference generator would invent a comparison
     from nothing.

Nothing else is filtered. Selecting on findings, pathology, scan gap, or modality
count would choose the external set on the axis the evaluation is measuring.

Outputs:
  data/bind_eval_{N}.csv          generation-batch input (integer index, 0..N-1)
  data/bind_eval_{N}_sessions.csv the 2N sessions to run HD-BET + ANTs over,
                                  in bind_longitudinal_data.csv schema
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_keys import load_patient_merge_map

DATA = Path(__file__).resolve().parent / "data"
MIN_REPORT_CHARS = 300

# BIND has no source for the `pathology1` column the S4 prompt reads. Marked
# unavailable rather than guessed (user's call, 2026-08-22). Note this is a token
# the S4 reference generation never saw -- S4's missing marker was
# "no_pathology_label" -- so it is a documented difference between how the two
# reference sets were produced.
PATHOLOGY_FILL = "unavailable"


def merged_ids(series, merge_map):
    return series.map(lambda v: merge_map.get(v, v) if pd.notna(v) else v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--pairs", type=Path, default=DATA / "bind_longitudinal_meta_paired.csv")
    ap.add_argument("--sessions", type=Path, default=DATA / "bind_longitudinal_data.csv")
    args = ap.parse_args()

    pairs = pd.read_csv(args.pairs, dtype=str, low_memory=False)
    pairs["duration"] = pairs["duration"].astype(int)
    sessions = pd.read_csv(args.sessions, dtype=str, low_memory=False)

    print(f"pool: {len(pairs):,} pairs, {pairs.patient_uid.nunique():,} patients")

    # --- exclusion 1: patient identity must agree on both sides of the join ---
    merge_map = load_patient_merge_map("I0001")
    session_patient = dict(zip(sessions.site + "/" + sessions.session_uid,
                               sessions.patient_uid))
    report_pid = merged_ids(pairs.patient_uid, merge_map)

    inconsistent = pd.Series(False, index=pairs.index)
    for column in ("study_uid1", "study_uid2"):
        image_pid = merged_ids((pairs.site + "/" + pairs[column]).map(session_patient),
                               merge_map)
        inconsistent |= image_pid.values != report_pid.values

    print(f"  drop {inconsistent.sum()} pair(s): image patient != report patient")
    pairs = pairs[~inconsistent].copy()

    # --- exclusion 2: degenerate reports ---
    too_short = (pairs.report1.fillna("").str.strip().str.len() < MIN_REPORT_CHARS) | \
                (pairs.report2.fillna("").str.strip().str.len() < MIN_REPORT_CHARS)
    print(f"  drop {too_short.sum()} pairs: a report under {MIN_REPORT_CHARS} chars")
    pairs = pairs[~too_short].copy()

    print(f"eligible: {len(pairs):,} pairs, {pairs.patient_uid.nunique():,} patients")

    # --- site quota, proportional to eligible PATIENTS (we sample patients) ---
    patient_site = pairs.groupby("patient_uid")["site"].first()
    share = patient_site.value_counts(normalize=True)
    quota = {s: int(round(share[s] * args.n)) for s in share.index}
    quota[share.index[0]] += args.n - sum(quota.values())   # absorb rounding
    print(f"  patient share {share.round(4).to_dict()} -> quota {quota}")

    rng_seed = args.seed
    chosen = []
    for site, k in quota.items():
        pool = patient_site[patient_site == site].index.to_series()
        picked = pool.sample(n=k, random_state=rng_seed)
        chosen.extend(picked.tolist())
        rng_seed += 1

    # one pair per chosen patient, uniformly among that patient's eligible pairs
    sample = (pairs[pairs.patient_uid.isin(set(chosen))]
              .groupby("patient_uid", group_keys=False)
              .apply(lambda g: g.sample(n=1, random_state=args.seed))
              .sort_values(["site", "patient_uid"])
              .reset_index(drop=True))

    assert len(sample) == args.n, f"drew {len(sample)}, wanted {args.n}"
    assert sample.patient_uid.nunique() == args.n, "a patient was drawn twice"

    # --- provenance flags: recorded, never used to filter ---
    sample["pathology1"] = PATHOLOGY_FILL
    sample["pathology2"] = PATHOLOGY_FILL

    out_pairs = DATA / f"bind_eval_{args.n}.csv"
    sample.to_csv(out_pairs, index=True)

    # --- the sessions those pairs need ---
    keys = set(sample.site + "/" + sample.study_uid1) | \
           set(sample.site + "/" + sample.study_uid2)
    session_key = sessions.site + "/" + sessions.session_uid
    todo = sessions[session_key.isin(keys)].copy()
    assert len(todo) == len(keys), f"{len(keys)} sessions wanted, {len(todo)} found"

    out_sessions = DATA / f"bind_eval_{args.n}_sessions.csv"
    todo.to_csv(out_sessions, index=False)

    print(f"\nsample: {len(sample):,} pairs / {sample.patient_uid.nunique():,} patients")
    print(f"  site       {sample.site.value_counts().to_dict()}")
    print(f"  gap days   median {sample.duration.median():.0f}  "
          f"mean {sample.duration.mean():.0f}  "
          f"IQR {sample.duration.quantile(.25):.0f}-{sample.duration.quantile(.75):.0f}")
    print(f"  sessions   {len(todo):,} unique")
    for modality in ("t1w", "t1ce", "t2w", "flair"):
        print(f"    {modality:5s} {todo[modality].notna().mean():.3f}")
    already = todo[session_key[session_key.isin(keys)].index].copy() if False else None
    print(f"\nwrote {out_pairs}")
    print(f"wrote {out_sessions}")


if __name__ == "__main__":
    main()
