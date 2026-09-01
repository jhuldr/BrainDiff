"""Standalone replication of Table 7 (difference-pathway ablation) from the raw generation
dumps -- no perreport caches, no API calls, no regeneration.

Re-derives rg_er / BLEU-4 / METEOR and the two paired bootstrap intervals, and prints the
provenance of the delta-ON arm (which S3 objective produced its change map) from the
checkpoint and the training log rather than from the YAML.

  conda activate radgraph_env && RG_CUDA=0 python paper/stats/replicate_table7.py

*Requires data that is NOT in this repository:* the raw generation dumps under `logs/` and the
full S4/subset tables with report text. The shipped CSVs carry identifiers only -- the MR-RATE
report text is not redistributable.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, _REPO)
os.chdir(_REPO)
from radgraph import F1RadGraph
from braindiff.eval import nlg_score

ARMS = {
    "delta on":                    "logs/rger_AB/A/rger_real.merged.json",
    "delta off":                   "logs/recompute/pp/rger_real.rank*.json",
    "wrong patient's scans":       "logs/recompute/pp/rger_vis_roll.rank*.json",
}

def load(spec):
    hyps, refs = [], []
    for f in sorted(glob.glob(spec)):
        d = json.load(open(f)); hyps += d["hyps"]; refs += d["refs"]
    return hyps, refs

f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl",
                cuda=int(os.environ.get("RG_CUDA", "0")))

def per_report(hyps, refs):
    keep = [i for i, (h, r) in enumerate(zip(hyps, refs)) if h and h.strip() and r and r.strip()]
    _, rl, *_ = f1(hyps=[hyps[i] for i in keep], refs=[refs[i] for i in keep])
    a = np.full(len(hyps), np.nan)
    for j, i in enumerate(keep):
        a[i] = rl[1][j]
    return a

R = {}
print(f"{'arm':26s} {'n':>5s} {'rg_er':>8s} {'BLEU-4':>8s} {'METEOR':>8s}   source")
for arm, spec in ARMS.items():
    h, r = load(spec)
    rg = per_report(h, r)
    nlg = nlg_score.score(h, r)
    R[arm] = dict(rg=rg, refs=r, bleu4=nlg["bleu4"], meteor=nlg["meteor"])
    print(f"{arm:26s} {len(h):5d} {np.nanmean(rg):8.4f} {nlg['bleu4']:8.4f} "
          f"{nlg['meteor']:8.4f}   {spec}", flush=True)

def paired(a, b, label):
    """Index-paired, with the reference-match assertion PAPER_CIS uses."""
    ra, rb = R[a]["refs"], R[b]["refs"]
    assert len(ra) == len(rb) and all(x == y for x, y in zip(ra, rb)), \
        f"{label}: dumps are not in the same order -- index pairing invalid"
    d = R[a]["rg"] - R[b]["rg"]
    d = d[np.isfinite(d)]
    rng = np.random.default_rng(0)
    s = np.array([d[rng.choice(len(d), len(d), True)].mean() for _ in range(10000)])
    lo, hi = np.percentile(s, [2.5, 97.5])
    return d.mean(), lo, hi, len(d)

print()
for label, a, b in [("delta contribution", "delta on", "delta off"),
                    ("image contribution", "delta off", "wrong patient's scans")]:
    m, lo, hi, n = paired(a, b, label)
    db = R[a]["bleu4"] - R[b]["bleu4"]; dm = R[a]["meteor"] - R[b]["meteor"]
    print(f"{label:20s} rg_er {m:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
          f"BLEU-4 {db:+.4f}  METEOR {dm:+.4f}   n={n}")

# ---- provenance of the delta-ON arm, read from artifacts not from the YAML -------------
print("\n--- delta-ON provenance (checkpoint + log, not curriculum.yaml) ---")
import torch, collections
sd = torch.load("checkpoints/nv_stage4_priorreport_5.pt", map_location="cpu")
pre = collections.Counter(k.split(".")[0] for k in sd)
cm = sum(v.numel() for k, v in sd.items() if k.startswith("change_map"))
print(f"S4 delta-on ckpt   nv_stage4_priorreport_5.pt")
print(f"  change_map       {pre['change_map']} tensors / {cm:,} params")
print(f"  connector.delta  {sum(1 for k in sd if k.startswith('connector.delta'))} tensors")
print(f"  diff_encoder     {sum(1 for k in sd if k.startswith('diff_encoder'))} tensors")
for pat, f in [(r"Loaded checkpoints/nv_stage3.*", "logs/nv_stage4_priorreport_changemap.log"),
               (r"note: delta_ckpt.*", "logs/nv_stage3_deltaunsup_extended_newarch_train.log")]:
    for ln in open(f, errors="ignore"):
        if re.match(pat, ln.strip()):
            print(f"  {os.path.basename(f)}: {ln.strip()}")
# Read the objective as text -- importing models/ needs flash-attn, which radgraph_env lacks.
src = open("models/change_map_pretrain.py").read()
hits = [f"{i}: {l.strip()}" for i, l in enumerate(src.splitlines(), 1)
        if re.search(r"LABEL-FREE|label-free|UNUSED", l)]
print(f"  objective        {'LABEL-FREE (unsupervised)' if hits else 'USES LABELS'}"
      f"   <- models/change_map_pretrain.py")
for h in hits:
    print(f"    change_map_pretrain.py:{h}")
print("REPLICATE_TABLE7_DONE")
