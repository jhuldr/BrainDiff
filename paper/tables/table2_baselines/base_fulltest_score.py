"""Score base-NeuroVFM FULL-TEST longitudinal reports (batch synthesis) on rg_er, and pair
vs ours (nodelta_10) full test. GPU RadGraph (radgraph_env, RG_CUDA=0). Writes result txt.

Pairing is by study_uid2, NOT by reference text: one normal target is duplicated across two
studies (A5QGT6KZV3 / K2FVY4JJJM share an identical synthesized report), so keying ours by ref
collapses them and drops a study (n=1443). The base CSV carries study_uid2 directly; ours is
keyed via the loader-order sidecar written by recover_test_uids.py (validated refs 1:1), giving
the honest n=1444. Point OURS_DUMP at the canonical aligned run (logs/recompute/pp) and SIDECAR
at its uids.json (loader order is identical across dumps, so one sidecar serves any 2-shard
test dump; refs are re-checked below)."""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import csv, os, json
import numpy as np
from radgraph import F1RadGraph
csv.field_size_limit(10**7)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = _REPO
f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=int(os.environ.get("RG_CUDA", "0")))
rng = np.random.default_rng(0)

def per(h, r):
    keep = [i for i, (x, y) in enumerate(zip(h, r)) if isinstance(x, str) and isinstance(y, str) and x.strip() and y.strip()]
    _, rl, *_ = f1(hyps=[h[i] for i in keep], refs=[r[i] for i in keep])
    a = np.full(len(h), np.nan)
    for j, i in enumerate(keep):
        a[i] = rl[1][j]
    return a

# canonical aligned ours dump (2-shard rank0++rank1) + its uid sidecar
OURS_DUMP = os.path.join(REPO, "logs/recompute/pp")            # rger_real.rank{0,1}.json
SIDECAR = os.path.join(REPO, "logs/rger_AB/A/rger_real.uids.json")

# base: keyed by study_uid2
base = {}
with open(os.path.join(HERE, "outputs/neurovfm_base/fulltest_longitudinal_meta_reports.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        base[r["study_uid2"]] = (r.get("gt_generated_report", ""), r.get("generated_report", ""))

# ours: merge the 2 shards in rank order (= sidecar order), then key by study_uid2
ours_hyps, ours_refs = [], []
for rk in (0, 1):
    d = json.load(open(os.path.join(OURS_DUMP, f"rger_real.rank{rk}.json")))
    ours_hyps += d["hyps"]; ours_refs += d["refs"]
side = json.load(open(SIDECAR))
assert ours_refs == side["refs"], "ours dump refs do not match the uid sidecar order"
ours = {u: h for u, h in zip(side["uids"], ours_hyps)}

uids = [u for u in base if u in ours]                          # 1444
ref = [base[u][0] for u in uids]
base_hyp = [base[u][1] for u in uids]
ours_hyp = [ours[u] for u in uids]

rg_base = per(base_hyp, ref)
rg_ours = per(ours_hyp, ref)
base_mean = float(np.nanmean(rg_base)); n_base = int(np.isfinite(rg_base).sum())
empties = sum(1 for x in base_hyp if not (isinstance(x, str) and x.strip()))

# paired by uid: rows where BOTH base and ours scored (ours - base, + => ours better)
both = np.isfinite(rg_base) & np.isfinite(rg_ours)
d = rg_ours[both] - rg_base[both]
ix = np.arange(len(d))
s = np.array([d[rng.choice(ix, len(ix), True)].mean() for _ in range(10000)])
lo, hi = np.percentile(s, [2.5, 97.5])
line = (f"base NeuroVFM FULL-TEST rg_er={base_mean:.4f} n={n_base} empties={empties} | "
        f"paired-by-uid n={both.sum()}: ours {rg_ours[both].mean():.4f} - base {rg_base[both].mean():.4f} "
        f"= {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]")
print(line)
open(os.path.join(HERE, "base_fulltest_result.txt"), "w").write(line + "\n")
print("BASE_FULLTEST_SCORED")
