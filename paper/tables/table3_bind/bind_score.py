import glob, json, os, numpy as np
from radgraph import F1RadGraph
f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=int(os.environ.get("RG_CUDA", "0")))
rng = np.random.default_rng(0)
D = "logs/bind_rger"
def merged(mode):
    H, R = [], []
    for f in sorted(glob.glob(f"{D}/rger_{mode}.rank*.json")):
        o = json.load(open(f)); H += o["hyps"]; R += o["refs"]
    return H, R
def per(H, R):
    keep = [i for i, (x, y) in enumerate(zip(H, R)) if x and x.strip() and y and y.strip()]
    _, rl, *_ = f1(hyps=[H[i] for i in keep], refs=[R[i] for i in keep])
    a = np.full(len(H), np.nan)
    for j, i in enumerate(keep): a[i] = rl[1][j]
    return a
hr, rr = merged("real"); hv, rv = merged("vis_roll")
rg_real = per(hr, rr); rg_vis = per(hv, rv)
real_m = float(np.nanmean(rg_real)); n_real = int(np.isfinite(rg_real).sum())
vis_m = float(np.nanmean(rg_vis))
INTERNAL = 0.3843
fin = rg_real[np.isfinite(rg_real)]
bs = np.array([fin[rng.choice(len(fin), len(fin), True)].mean() for _ in range(10000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
line = ("BIND external rg_er(real)=%.4f n=%d CI[%.4f,%.4f] | vis_roll=%.4f image_worth=%+.4f | "
        "internal=%.4f DELTA(BIND-internal)=%+.4f |abs|=%.4f" %
        (real_m, n_real, lo, hi, vis_m, real_m - vis_m, INTERNAL, real_m - INTERNAL, abs(real_m - INTERNAL)))
print(line)
open("paper/tables/table3_bind/bind_result.txt", "w").write(line + "\n")
print("BIND_SCORED")
