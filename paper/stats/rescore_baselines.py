"""Re-score the non-BrainDiff rows of Table 2, the metric floor, and Table 10.

Separate from rescore_all.py because these come from CSV/JSONL rather than the
trainer's {hyps, refs} dumps, and because they need the subset_64 study list.

Run:  conda activate radgraph_env && RG_CUDA=1 python paper/stats/rescore_baselines.py

*Requires data that is NOT in this repository:* the raw generation dumps under `logs/` and the
full S4/subset tables with report text. The shipped CSVs carry identifiers only -- the MR-RATE
report text is not redistributable.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_TEST_CSV_DEFAULT = _os.path.join(_REPO, "paper", "cache", "s4_test.csv")
if not _os.path.exists(_TEST_CSV_DEFAULT):
    _TEST_CSV_DEFAULT = "/home/data/BRAIN_DIFF_S4/splits_extended/s4_test.csv"
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, _REPO)
csv.field_size_limit(10 ** 7)
from radgraph import F1RadGraph
from braindiff.eval import nlg_score

HERE = _REPO
B = f"{HERE}/benchmarks/mrrate_proprietary"
OUT = f"{HERE}/../cache/perreport"
os.makedirs(OUT, exist_ok=True)

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


def report(tag, hyps, refs, uids=None):
    rg = per_report(hyps, refs)
    nlg = nlg_score.score(hyps, refs)
    json.dump({"rg_er": rg.tolist(), "uids": list(uids) if uids is not None else list(refs),
               "bleu4": nlg["bleu4"], "meteor": nlg["meteor"], "n": len(hyps)},
              open(f"{OUT}/{tag}.json", "w"))
    print(f"{tag:34s} n={len(hyps):5d} rg_er={np.nanmean(rg):.4f} "
          f"BLEU-4={nlg['bleu4']:.4f} METEOR={nlg['meteor']:.4f}", flush=True)
    return rg


# ---- NeuroVFM base, full test (Table 2 row 4) -------------------------------
rows = list(csv.DictReader(open(f"{B}/cache/outputs/neurovfm_base/fulltest_longitudinal_meta_reports.csv")))
report("neurovfm_base_fulltest",
       [r["generated_report"] for r in rows], [r["gt_generated_report"] for r in rows])

# ---- frontier models, 64-study subset (Table 2 rows 2-3) --------------------
for tag, path in (("claude_opus5_subset64", f"{B}/outputs/claude/reports.jsonl"),
                  ("gpt56sol_subset64", f"{B}/outputs/openai/reports.jsonl")):
    d = [json.loads(l) for l in open(path)]
    report(tag, [x["hyp"] for x in d], [x["ref"] for x in d])

# ---- ours + NeuroVFM restricted to that same 64 (Table 2 footnote b) --------
sub = list(csv.DictReader(open(f"{B}/cache/subset_64.csv")))
sub_refs = {s["generated_report"].strip() for s in sub}
sub_uids = {s["study_uid2"] for s in sub}

od = json.load(open(f"{HERE}/logs/test/nodelta10/rger_real.merged.json"))
pairs = [(h, r) for h, r in zip(od["hyps"], od["refs"]) if r.strip() in sub_refs]
report("braindiff_subset64", [h for h, _ in pairs], [r for _, r in pairs])

nv = [r for r in rows if r["study_uid2"] in sub_uids]
report("neurovfm_subset64", [r["generated_report"] for r in nv],
       [r["gt_generated_report"] for r in nv])

# ---- metric floor: reference vs ANOTHER patient's reference (Table 6) -------
refs = [r for r in od["refs"] if r and r.strip()]
rolled = refs[7:] + refs[:7]                       # derangement, 0 self-pairs
report("metric_floor_ref_vs_other_ref", rolled, refs)

# ---- Table 10: real radiologist Impression as reference --------------------
test = {r["study_uid2"]: r for r in
        csv.DictReader(open(_TEST_CSV_DEFAULT))}
by_ref = {}
for uid, r in test.items():
    by_ref.setdefault(r["generated_report"].strip(), []).append(uid)


def impression(t):
    t = (t or "")
    i = t.rfind("Impression:")
    return t[i:].strip() if i >= 0 else ""


ours_imp, nv_imp, real_imp = [], [], []
nv_by_uid = {r["study_uid2"]: r["generated_report"] for r in rows}
for h, r in zip(od["hyps"], od["refs"]):
    uids = by_ref.get(r.strip(), [])
    if len(uids) != 1:
        continue
    uid = uids[0]
    real = impression(test[uid]["report2"])
    if not real or uid not in nv_by_uid:
        continue
    oh, nh = impression(h), impression(nv_by_uid[uid])
    if not oh or not nh:
        continue
    ours_imp.append(oh); nv_imp.append(nh); real_imp.append(real)

print(f"\nTable 10 subset: {len(real_imp)} studies with an explicit report2 Impression "
      f"({len(real_imp)/len(od['refs'])*100:.1f}% of {len(od['refs'])})   [claim n=1239, '90%']")
report("table10_braindiff_vs_real_impression", ours_imp, real_imp)
report("table10_neurovfm_vs_real_impression", nv_imp, real_imp)
print("BASELINES_DONE")
