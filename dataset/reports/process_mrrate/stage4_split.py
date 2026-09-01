"""Split S4 main.csv into train/val/test, grouped by patient, with pinned studies.

    python process_mrrate/stage4_split.py \
        --csv /home/data/BRAIN_DIFF_S4/main.csv \
        --train 80 --val 10 --test 10 \
        --seed 0 \
        --force-train pinned_studies.txt \
        --out-dir /home/data/BRAIN_DIFF_S4/splits

Three things this is careful about, each of which has burned this project before:

1. PATIENT GROUPING, NOT ROW GROUPING. A patient contributes up to 29 rows here,
   and their studies are longitudinal near-duplicates. Splitting by row puts the
   same patient on both sides and inflates val. Every row for a patient_uid lands
   in exactly one split, and that is asserted at the end, not assumed.

2. PINNING A STUDY PINS ITS PATIENT. `--force-train` names study UIDs, but sending
   one study to train while its patient's other studies go to val would be exactly
   the leak we are avoiding. So the pinned study's whole patient goes to train.
   The report prints how many patients that pulled in beyond the studies named.Re

3. PROPORTIONS ARE BY ROW, NOT BY PATIENT. Patients have wildly different row
   counts (1 to 29), so splitting patients 80/10/10 does not split rows 80/10/10.
   Patients are walked in shuffled order and each is dropped into whichever split
   is furthest below its target ROW count -- groups stay intact, proportions still
   land close. Same approach as train_dual.split_csv.

A study UID is looked up in BOTH study_uid1 (prior) and study_uid2 (current), since
a pinned study can appear as either half of a pair. UIDs the CSV does not contain
are WARNED ABOUT AND SKIPPED, not fatal -- the pin list is maintained separately
from main.csv and legitimately names studies this CSV does not carry. A missing UID
is not quietly in val; it is in no split, because it is not in the input. Only a
UID that exists in the CSV and still fails to land in train is an error, since that
would be a bug in the split itself.
"""
import argparse
import os
import sys

import pandas as pd

SPLITS = ("train", "val", "test")


def load_forced(arg):
    """--force-train is a path to a file (one UID per line) or a comma-separated list."""
    if not arg:
        return []
    if os.path.exists(arg):
        with open(arg) as fh:
            return [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    return [s.strip() for s in arg.split(",") if s.strip()]


def resolve_patients(df, study_uids):
    """Map pinned study UIDs -> patient_uids. Returns (patients, missing).

    A UID absent from the CSV is EXPECTED and not fatal: the pin list is maintained
    independently of whatever main.csv currently holds, so it legitimately names
    studies this CSV does not carry. Those UIDs are reported and skipped.

    What a missing UID does NOT mean: it is not quietly in val. It is not in the
    split at all, because it is not in the input. The warning exists so that a pin
    list which has silently stopped matching gets noticed instead of assumed.
    """
    in1 = df.set_index("study_uid1")["patient_uid"]
    in2 = df.set_index("study_uid2")["patient_uid"]
    patients, missing = set(), []
    for uid in study_uids:
        if uid in in1.index:
            patients.update(pd.Series(in1.loc[uid]).tolist())
        elif uid in in2.index:
            patients.update(pd.Series(in2.loc[uid]).tolist())
        else:
            missing.append(uid)
    return patients, missing


def split(df, fracs, seed, forced_patients):
    counts = df["patient_uid"].value_counts()
    total = len(df)
    targets = {s: fracs[s] * total for s in SPLITS}
    current = {s: 0 for s in SPLITS}
    assign = {s: set() for s in SPLITS}

    # Pinned patients first, so their rows count against train's quota and the
    # greedy pass below compensates by favouring val/test until they catch up.
    for pid in forced_patients:
        assign["train"].add(pid)
        current["train"] += counts[pid]

    rest = [p for p in counts.index if p not in forced_patients]
    rest = pd.Series(rest).sample(frac=1, random_state=seed).tolist()
    for pid in rest:
        s = max(SPLITS, key=lambda s: targets[s] - current[s])
        assign[s].add(pid)
        current[s] += counts[pid]

    return {s: df[df["patient_uid"].isin(assign[s])].reset_index(drop=True) for s in SPLITS}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="/home/data/BRAIN_DIFF_S4/main.csv")
    p.add_argument("--train", type=float, required=True, help="percent or fraction")
    p.add_argument("--val", type=float, required=True)
    p.add_argument("--test", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force-train", default=None,
                   help="file of study UIDs (one per line) or a comma-separated list; "
                        "their PATIENTS are pinned to train")
    p.add_argument("--out-dir", default=None, help="omit to report without writing")
    p.add_argument("--prefix", default="s4")
    a = p.parse_args()

    raw = {"train": a.train, "val": a.val, "test": a.test}
    tot = sum(raw.values())
    if abs(tot - 100.0) < 1e-6:
        fracs = {s: v / 100.0 for s, v in raw.items()}
    elif abs(tot - 1.0) < 1e-6:
        fracs = dict(raw)
    else:
        raise SystemExit(f"--train/--val/--test sum to {tot}, expected 100 (percent) "
                         f"or 1.0 (fraction)")

    df = pd.read_csv(a.csv)
    for col in ("patient_uid", "study_uid1", "study_uid2"):
        if col not in df.columns:
            raise SystemExit(f"{a.csv} has no '{col}' column; got {list(df.columns)}")

    forced_uids = load_forced(a.force_train)
    forced_patients, missing_uids = resolve_patients(df, forced_uids)
    if missing_uids:
        print(f"WARNING: {len(missing_uids)} of {len(set(forced_uids))} --force-train study "
              f"UID(s) are not in this CSV (checked both study_uid1 and study_uid2). "
              f"They are skipped -- they are in NO split, because they are not in the "
              f"input at all.\n  first few: {missing_uids[:5]}\n")
    parts = split(df, fracs, a.seed, forced_patients)

    # ---- verification: assert, do not trust ------------------------------------
    seen = {}
    for s in SPLITS:
        for pid in parts[s]["patient_uid"].unique():
            if pid in seen:
                raise SystemExit(f"PATIENT LEAK: {pid} in both {seen[pid]} and {s}")
            seen[pid] = s
    assert sum(len(parts[s]) for s in SPLITS) == len(df), "rows lost or duplicated"

    # Only UIDs that actually exist in the CSV can be expected in train. Missing
    # ones were already reported above; including them here would turn every
    # skipped UID into a spurious failure.
    train_uids = set(parts["train"]["study_uid1"]) | set(parts["train"]["study_uid2"])
    matched = [u for u in forced_uids if u not in missing_uids]
    stray = [u for u in matched if u not in train_uids]
    if stray:
        raise SystemExit(f"{len(stray)} pinned study UID(s) exist in the CSV but did not "
                         f"land in train -- this is a bug in the split, not a bad pin "
                         f"list: {stray[:5]}")

    # ---- report ----------------------------------------------------------------
    print(f"{a.csv}: {len(df)} rows, {df['patient_uid'].nunique()} patients, seed {a.seed}")
    if forced_uids:
        pinned_rows = len(parts["train"][parts["train"]["patient_uid"].isin(forced_patients)])
        # Pinning is transitive through the patient, so the rows dragged into train
        # exceed the studies named. Report that explicitly -- it is the number that
        # explains why train can overshoot its target share.
        n_matched = len(set(forced_uids)) - len(set(missing_uids))
        print(f"pinned: {n_matched} of {len(set(forced_uids))} studies matched"
              + (f" ({len(set(missing_uids))} not in this CSV, skipped)" if missing_uids else "")
              + f" -> {len(forced_patients)} patients -> {pinned_rows} rows in train "
              f"({pinned_rows - n_matched} pulled in by patient grouping)")
    print(f"\n{'split':<7} {'rows':>7} {'row %':>8} {'target':>8} {'patients':>9}")
    for s in SPLITS:
        n = len(parts[s])
        print(f"{s:<7} {n:>7} {100*n/len(df):7.2f}% {100*fracs[s]:7.2f}% "
              f"{parts[s]['patient_uid'].nunique():>9}")
    print("\nno patient appears in more than one split (verified)")

    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        for s in SPLITS:
            path = os.path.join(a.out_dir, f"{a.prefix}_{s}.csv")
            parts[s].to_csv(path, index=False)
            print(f"wrote {path}")
    else:
        print("\n(--out-dir not given; nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
