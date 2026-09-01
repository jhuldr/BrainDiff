import json, os, numpy as np
from radgraph import F1RadGraph
f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=int(os.environ.get("RG_CUDA", "0")))
d = json.load(open("paper/cache/outputs/table2_cell/rger_vis_roll.noreport.json"))
h, r = d["hyps"], d["refs"]
keep = [i for i, (x, y) in enumerate(zip(h, r)) if x and x.strip() and y and y.strip()]
_, rl, *_ = f1(hyps=[h[i] for i in keep], refs=[r[i] for i in keep])
m = float(np.mean(rl[1]))
line = "TABLE2 prior-withheld x other-patient (no-report + vis_roll, nodelta_10) rg_er=%.4f n=%d" % (m, len(keep))
print(line)
open("benchmarks/mrrate_proprietary/table2_cell_rger.txt", "w").write(line + "\n")
print("TABLE2_CELL_SCORED")
