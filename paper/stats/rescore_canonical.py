"""Re-score the CANONICAL dumps the manuscript's tables are built from.

Provenance: RESULTS.md (2026-08-24 "unified recompute", 2026-08-26 reconciliation)
names `logs/recompute/pp` as the canonical production dump for Tables 1/2/3, with the
other arms recomputed the same way under logs/recompute_*. This script re-derives
rg_er / BLEU-4 / METEOR from those dumps only, so every table cell traces to one run.

Requires data that is NOT in this repository: the raw generation dumps under logs/, and the
FULL S4 test table (with `generated_report`) to map each reference to its study_uid2. The
shipped `assets/s4_test.csv` carries identifiers only -- the MR-RATE report text is not
redistributable. Point S4_TEST_FULL at the original table to run this.

It writes the id-keyed cache schema the readers expect:
    {rg_er: [...], uids: [...], bleu4: float, meteor: float, n: int, source: str}
No report text is written.

Run:  conda activate radgraph_env && RG_CUDA=2 python paper/stats/rescore_canonical.py
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, _REPO)
from radgraph import F1RadGraph
from braindiff.eval import nlg_score

OUT = _REPO + "/paper/cache/perreport"
os.makedirs(OUT, exist_ok=True)

# The full S4 test table, WITH report text -- not shipped. See the module docstring.
S4_TEST_FULL = "/home/data/BRAIN_DIFF_S4/splits_extended/s4_test.csv"


def _uid_resolver():
    """reference text -> study_uid2, duplicates resolved by order of appearance."""
    import collections, csv
    csv.field_size_limit(10 ** 7)
    if not os.path.exists(S4_TEST_FULL):
        raise SystemExit(
            f"{S4_TEST_FULL} not found. Re-scoring needs the full S4 test table, which "
            "carries the MR-RATE report text and is not redistributable; the copy in "
            "assets/s4_test.csv has identifiers only.")
    by_ref = collections.defaultdict(list)
    for r in csv.DictReader(open(S4_TEST_FULL)):
        by_ref[r["generated_report"].strip()].append(r["study_uid2"])
    seen = collections.Counter()

    def resolve(refs):
        out = []
        for k in (x.strip() for x in refs):
            u = by_ref.get(k, [])
            out.append(u[min(seen[k], len(u) - 1)] if u else k)
            seen[k] += 1
        return out
    return resolve

ARMS = {
    # BrainDiff production arm (delta OFF) -- Tables 2,3,4,6,7,9
    "C_braindiff_real":        "logs/recompute/pp/rger_real.rank*.json",
    "C_braindiff_visroll":     "logs/recompute/pp/rger_vis_roll.rank*.json",
    "C_braindiff_noreport":    "logs/recompute/nr/rger_real.noreport.rank*.json",
    "C_braindiff_nr_visroll":  "logs/recompute/nr/rger_vis_roll.noreport.rank*.json",
    "C_braindiff_scanzero":    "logs/recompute_sz/rger_scan_zero.rank*.json",
    "C_braindiff_sz_noreport": "logs/recompute_sz_nr/rger_scan_zero.noreport.rank*.json",
    # no-counterfactual arm -- Table 4 row 1
    "C_nocf_real":             "logs/recompute_nocf/pp/rger_real.rank*.json",
    "C_nocf_visroll":          "logs/recompute_nocf/pp/rger_vis_roll.rank*.json",
    # no-curriculum arm -- Tables 5, 9
    "C_nocurric_real":         "logs/recompute_nc/pp/rger_real.rank*.json",
    "C_nocurric_visroll":      "logs/recompute_nc/pp/rger_vis_roll.rank*.json",
    "C_nocurric_noreport":     "logs/recompute_nc/nr/rger_real.noreport.rank*.json",
    "C_nocurric_scanzero":     "logs/recompute_nc_sz/rger_scan_zero.rank*.json",
    # delta-ON arm -- Table 7
    "C_deltaon_real":          "logs/rger_AB/A/rger_real.merged.json",
    "C_deltaon_visroll":       "logs/rger_AB/A/rger_delta_roll.merged.json",
}


def load(spec):
    hyps, refs = [], []
    for p in sorted(glob.glob(spec)):
        d = json.load(open(p))
        hyps += d["hyps"]; refs += d["refs"]
    return hyps, refs


def main():
    f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl",
                    cuda=int(os.environ.get("RG_CUDA", "0")))

    def per_report(hyps, refs):
        keep = [i for i, (h, r) in enumerate(zip(hyps, refs))
                if h and h.strip() and r and r.strip()]
        _, rl, *_ = f1(hyps=[hyps[i] for i in keep], refs=[refs[i] for i in keep])
        a = np.full(len(hyps), np.nan)
        for j, i in enumerate(keep):
            a[i] = rl[1][j]
        return a

    print(f"{'arm':26s} {'n':>5s} {'rg_er':>8s} {'BLEU-4':>8s} {'METEOR':>8s}")
    for arm, spec in ARMS.items():
        if not glob.glob(spec):
            print(f"{arm:26s}  MISSING: {spec}")
            continue
        hyps, refs = load(spec)
        rg = per_report(hyps, refs)
        nlg = nlg_score.score(hyps, refs)
        json.dump({"rg_er": rg.tolist(), "uids": _uid_resolver()(refs),
                   "bleu4": nlg["bleu4"], "meteor": nlg["meteor"], "n": len(hyps),
                   "source": spec},
                  open(f"{OUT}/{arm}.json", "w"))
        print(f"{arm:26s} {len(hyps):5d} {np.nanmean(rg):8.4f} "
              f"{nlg['bleu4']:8.4f} {nlg['meteor']:8.4f}", flush=True)
    print("CANONICAL_DONE")


if __name__ == "__main__":
    main()
