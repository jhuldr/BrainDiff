import json, os, numpy as np
from radgraph import F1RadGraph
f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=int(os.environ.get("RG_CUDA", "0")))
rng = np.random.default_rng(0)
d = json.load(open("logs/rger_AB/A/rger_real.merged.json"))
refs = [r for r in d["refs"] if r and r.strip()]
# floor: each ref vs a DIFFERENT patient's ref (roll by 1)
rolled = refs[1:] + refs[:1]
_, rl, *_ = f1(hyps=rolled, refs=refs)   # hyp = other patient's ref, ref = this patient's
a = np.array(rl[1])
m = float(a.mean())
bs = np.array([a[rng.choice(len(a), len(a), True)].mean() for _ in range(10000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
line = "TABLE2 FLOOR (ref vs other patient's ref) rg_er=%.4f CI[%.4f,%.4f] n=%d" % (m, lo, hi, len(a))
print(line)
open("benchmarks/mrrate_proprietary/table2_floor.txt", "w").write(line + "\n")
print("TABLE2_FLOOR_DONE")
