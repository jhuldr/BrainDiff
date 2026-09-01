"""Freeze the 64-study benchmark subset: deterministic sample from the S4 TEST split.
All systems (ours, base NeuroVFM, Claude, GPT) read this exact list, so the comparison is
on identical studies. Run once; commit subset_64.csv."""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_TEST_CSV_DEFAULT = _os.path.join(_REPO, "paper", "cache", "s4_test.csv")
if not _os.path.exists(_TEST_CSV_DEFAULT):
    _TEST_CSV_DEFAULT = "/home/data/BRAIN_DIFF_S4/splits_extended/s4_test.csv"
import csv, os, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = _TEST_CSV_DEFAULT
SEED = 1
N = 64
KEEP = ["patient_uid", "study_uid1", "study_uid2", "report1", "report2",
        "generated_report", "classification", "duration"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=os.path.join(HERE, "subset_64.csv"))
    a = ap.parse_args()

    csv.field_size_limit(10**7)
    with open(TEST_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    # Eligible pool = studies our model actually generated for (in the nodelta_10 dump).
    # These are exactly the dataloader-kept rows (usable images at BOTH timepoints), so all
    # 64 work for every system. Sampling from the raw 1464 can hit the ~20 incomplete
    # studies the dataloader drops, which have no images.
    import json
    dump = json.load(open(_REPO + "/logs/test/nodelta10/rger_real.merged.json"))
    eligible_refs = set(dump["refs"])
    rows = [r for r in rows if r.get("generated_report", "") in eligible_refs]
    n_all = len(rows)
    print(f"eligible pool (in nodelta_10 dump): {n_all}")
    rng = np.random.default_rng(a.seed)
    idx = rng.choice(n_all, size=min(a.n, n_all), replace=False)
    idx.sort()
    picked = [rows[i] for i in idx]

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP)
        w.writeheader()
        for r in picked:
            w.writerow({k: r.get(k, "") for k in KEEP})

    # provenance / sanity
    from collections import Counter
    cls = Counter(r.get("classification", "?") for r in picked)
    print(f"sampled {len(picked)}/{n_all} test rows (seed {a.seed}) -> {a.out}")
    print("classification balance:", dict(cls))
    empties = sum(1 for r in picked if not r.get("generated_report", "").strip())
    print(f"rows with empty generated_report (reference): {empties}")

if __name__ == "__main__":
    main()
