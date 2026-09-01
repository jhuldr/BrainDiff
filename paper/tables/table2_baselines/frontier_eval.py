"""Frontier + NeuroVFM comparison on the seed-1 64-subset: rg_er, BLEU-4, METEOR, and the
paired margin against ours -- in one pass.

Merges the former frontier_score.py (rg_er) and frontier_nlg.py (BLEU-4/METEOR); the numbers
are unchanged. Runs in radgraph_env, which carries both radgraph and nltk.

  RG_CUDA=0 python benchmarks/mrrate_proprietary/frontier_eval.py
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import json, os, sys
import numpy as np
sys.path.insert(0, _REPO)
from radgraph import F1RadGraph
from braindiff.eval import nlg_score

f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl",
                cuda=int(os.environ.get("RG_CUDA", "0")))
rng = np.random.default_rng(0)


def load(p):
    H, R = [], []
    for ln in open(p):
        d = json.loads(ln); H.append(d.get("hyp") or ""); R.append(d.get("ref") or "")
    return H, R


def per(H, R):
    keep = [i for i, (h, r) in enumerate(zip(H, R)) if h and h.strip() and r and r.strip()]
    _, rl, *_ = f1(hyps=[H[i] for i in keep], refs=[R[i] for i in keep])
    a = np.full(len(H), np.nan)
    for j, i in enumerate(keep):
        a[i] = rl[1][j]
    return a


# ours on the 64, from the full-test dump, keyed by reference text
od = json.load(open("logs/test/nodelta10/rger_real.merged.json"))
ours = {r: h for h, r in zip(od["hyps"], od["refs"]) if r and r.strip()}

for name, path in [("Claude (Opus 5)", "paper/cache/outputs/claude/reports.jsonl"),
                   ("GPT-5.6 Sol", "paper/cache/outputs/openai/reports.jsonl")]:
    H, R = load(path)
    rg = per(H, R)
    nlg = nlg_score.score(H, R)
    empties = sum(1 for h in H if not h.strip())

    idx = [i for i in range(len(H)) if R[i] in ours and H[i].strip()]
    ph = [H[i] for i in idx]; po = [ours[R[i]] for i in idx]; pr = [R[i] for i in idx]
    rs, ro = per(ph, pr), per(po, pr)
    both = np.isfinite(rs) & np.isfinite(ro)
    d = rs[both] - ro[both]
    s = np.array([d[rng.choice(len(d), len(d), True)].mean() for _ in range(10000)])
    lo, hi = np.percentile(s, [2.5, 97.5])
    o_nlg = nlg_score.score(po, pr)

    print(f"{name}: rg_er={np.nan_to_num(rg).mean():.4f}  BLEU-4={nlg['bleu4']:.4f}  "
          f"METEOR={nlg['meteor']:.4f}  n={len(H)} empties={empties}")
    print(f"    paired vs ours: {name.split()[0]} {rs[both].mean():.4f}  ours {ro[both].mean():.4f}  "
          f"sys-ours {d.mean():+.4f} CI[{lo:+.4f},{hi:+.4f}]  n={both.sum()}")
    print(f"    ours on the same rows: BLEU-4={o_nlg['bleu4']:.4f} METEOR={o_nlg['meteor']:.4f}")
print("FRONTIER_EVAL_DONE")
