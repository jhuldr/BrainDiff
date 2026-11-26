#!/usr/bin/env python3
"""Paper Table 2, brain block: point estimate, 95% CI and paired significance vs BrainDiff for every system x internal /
external test.
  macro-MCC      mean over NEW / WORSENED / IMPROVED / RESOLVED of MCC(Rad-Delta presence >= 0.5, original-report statuses Y)
  agreement      finding-level Rad-Delta agreement vs the original current report (ref_changes > 0)
  rg_er, F1temp  per pair vs the edited target (rg_er = RadGraph entity+relation F1)
  readers        blinded LLM readers (paper/readers.py), first read of each item, change_score 1-5
  CI             percentile patient bootstrap, 2,000 resamples, seed 0; sig: '*' = paired CI excludes 0
Modes:
  full   (--full) every cell from the private scorer outputs ($BRAINDIFF_CKPT) and reader answers ($READER_DIR)
  cache  (default) paper/cache holds no reference labels, so only rg_er and F1temp are recomputed; macro-MCC, agreement
         and reader cells are copied from table2_golden.json (marked 'g', listed under "from_golden")
    python paper/table2.py [--full] [--out paper/table2.json]     (CPU)
    python paper/table2.py --build_cache                          full-mode inputs -> paper/cache"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import matthews_corrcoef

H = os.path.dirname(os.path.abspath(__file__)); C = f"{H}/cache"; SPLITS = ("test", "external_test")
CK = os.environ.get("BRAINDIFF_CKPT", "checkpoints")
# key, label, source run dir (checkpoints/table2_baselines/<dir>), reader batches for the cell, reader batches for the paired difference
SYSTEMS = [("copy", "Copy prior report", None, (), ()),
           ("neurovfm14b", "NeuroVFM -> Qwen3-14B", "neurovfm14b_rp110", ("b14",), ("b14",)),
           ("qwen3vl8b_8", "Qwen3-VL-8B, 8 sl.", "qwen3vl8b_imginstr_sl8img_rp110", ("b13",), ("b13",)),
           ("qwen3vl8b_48", "Qwen3-VL-8B, 48 sl.", "qwen3vl8b_imginstr_s48_rp110", ("b19",), ("b19",)),
           ("medgemma4b_vol", "MedGemma-1.5-4B, volume", "medgemma_vol85_rp110", ("b22",), ("b21", "b22")),
           ("medgemma27b_8", "MedGemma-27B, 8 sl.", "medgemma27b_sl8img_rp110", ("b22",), ("b21", "b22")),
           ("medgemma27b_48", "MedGemma-27B, 48 sl.", "medgemma27b_s48_rp110", ("b21",), ("b21",)),
           ("qwen3vl8b_ft_sf", "Qwen3-VL-8B FT, status-first", "qwen3vl8b_ft-sf1_imginstr_sl8img_rp110", ("b20",), ("b20",)),
           ("ours", "BrainDiff (ours)", "ours_rp110", ("b13", "b14", "b16", "b19"), ())]
METRICS = ("mcc", "agree_change", "agree_stable", "readers_change", "readers_stable", "rger", "f1temp")
CACHED = ("rger", "f1temp")


def mcc(Y, H):
    return float(np.mean([matthews_corrcoef(Y[:, j], H[:, j]) if 0 < H[:, j].sum() < len(H) else 0 for j in range(4)]))


def load_full():
    """Scorer outputs -> per-pair dicts, Y.any(1) and reader reads per split. Rows = score-npz order (sorted pair_id)."""
    import pandas as pd
    import readers
    G, B = f"{CK}/stage2_A0_s1d_sf", f"{CK}/table2_baselines"; dirs = {d: k for k, _, d, *_ in SYSTEMS if d}; D, CHG, RR, rowof, pid = {}, {}, {}, {}, {}
    for sp in SPLITS:
        g = np.load(f"{G}/score_{sp}_epoch20.npz", allow_pickle=True); ga = np.load(f"{G}/gen_{sp}_epoch20.agree.npz", allow_pickle=True)
        assert (g["pair_id"] == ga["pair_id"]).all() and (np.diff(g["pair_id"]) > 0).all()
        _, grp = np.unique(g["patient"], return_inverse=True); rowof[sp] = {int(x): i for i, x in enumerate(g["pair_id"])}; pid[sp] = g["patient"]
        Y, ch, CHG[sp] = g["Y"].astype(bool), ga["ref_changes"] > 0, g["Y"].astype(bool).any(1)
        pp = pd.read_parquet(f"{CK}/table2/perpair_{sp}.parquet").set_index("pair_id").loc[g["pair_id"]]
        D[sp] = {"copy": dict(Y=Y, H=g["copy_prior"] >= 0.5, A=g["agree_copy_prior"], ch=ch, pat=grp, rg=pp.rger_copy.values, ft=pp.f1temp_copy.values)}
        for d, k in dirs.items():
            s = np.load(f"{B}/{d}/score_{sp}_epoch20.npz", allow_pickle=True); a = np.load(f"{B}/{d}/gen_{sp}_epoch20.agree.npz", allow_pickle=True)
            pp = pd.read_parquet(f"{B}/{d}/perpair_{sp}.parquet").set_index("pair_id").loc[s["pair_id"]]
            assert (s["pair_id"] == g["pair_id"]).all() and (a["pair_id"] == g["pair_id"]).all(), d
            assert (s["Y"] == g["Y"]).all() and (s["patient"] == g["patient"]).all() and (a["ref_changes"] == ga["ref_changes"]).all(), d
            D[sp][k] = dict(Y=s["Y"].astype(bool), H=s["model"] >= 0.5, A=a["agree"], ch=a["ref_changes"] > 0, pat=grp, rg=pp.rger.values, ft=pp.f1temp.values)
    for b in sorted({b for *_, rb, sb in SYSTEMS for b in rb + sb}):
        for k, a, _ in readers.first_reads(b)[0]:
            if k["system"] not in dirs: continue   # systems not in Table 2
            r = rowof[k["split"]][k["pair_id"]]; assert pid[k["split"]][r] == k["patient"]
            RR.setdefault(k["split"], {}).setdefault(b, {}).setdefault(dirs[k["system"]], []).append([r, a["change_score"]])
    return D, CHG, RR


def load_cache():
    D = {}
    for sp in SPLITS:
        grp = np.load(f"{C}/groups_{sp}.npz")["group"]
        D[sp] = {k: (lambda s: dict(pat=grp, rg=s["rger"], ft=s["f1temp"]))(np.load(f"{C}/{k}_{sp}.npz")) for k, *_ in SYSTEMS}
    return D


def stats(d, ix):
    r = dict(rger=float(np.nanmean(d["rg"][ix])), f1temp=float(np.nanmean(d["ft"][ix])))
    if "Y" not in d:   # cache mode
        return r
    c = d["ch"][ix]
    return dict(mcc=mcc(d["Y"][ix], d["H"][ix]), agree_change=float(d["A"][ix][c].mean()), agree_stable=float(d["A"][ix][~c].mean())) | r


def boot(pat, f, n=2000):
    """f(index) -> dict of metrics; -> {metric: [2.5, 97.5] percentiles} over n patient resamples."""
    up = np.unique(pat); idx = {u: np.where(pat == u)[0] for u in up}; rng = np.random.default_rng(0)
    v = [f(np.concatenate([idx[u] for u in rng.choice(up, len(up))])) for _ in range(n)]
    return {k: [round(float(x), 4) for x in np.nanpercentile([z[k] for z in v], [2.5, 97.5])] for k in v[0]}


def reader_cell(RR, batches, key, chg, pat):
    """Per-pair mean of the system's first reads over its batches."""
    per = {}
    for b in batches:
        for row, cs in RR[b].get(key, []):
            per.setdefault(row, []).append(cs)
    rows = sorted(per); cs = np.array([np.mean(per[r]) for r in rows]); pt = pat[rows]; c = chg[rows]
    ci = boot(pt, lambda ix: {"readers_change": cs[ix][c[ix]].mean(), "readers_stable": cs[ix][~c[ix]].mean()})
    r = {nm: [round(float(cs[m].mean()), 3), ci[nm], int(m.sum())] for nm, m in (("readers_change", c), ("readers_stable", ~c))}
    r["reader_reads_per_pair"] = round(float(np.mean([len(per[x]) for x in rows])), 2)
    return r


def reader_sig(RR, batches, key, chg, pat):
    """Paired BrainDiff - system over pairs read by every system in the batches, change / stable."""
    rows = {}
    for b in batches:
        for s, v in RR[b].items():
            for row, cs in v:
                rows[(s, row)] = cs
    systems = sorted({s for s, _ in rows})
    pids = sorted({r for _, r in rows if all((z, r) in rows for z in systems)})
    pt = pat[pids]; up = np.unique(pt); idx = {u: np.where(pt == u)[0] for u in up}
    rng = np.random.default_rng(0); B = [np.concatenate([idx[u] for u in rng.choice(up, len(up))]) for _ in range(2000)]
    c = chg[pids]; dd = np.array([rows[("ours", r)] for r in pids], float) - np.array([rows[(key, r)] for r in pids], float); out = {}
    for nm, m in (("readers_change", c), ("readers_stable", ~c)):
        ci = np.percentile([dd[b][m[b]].mean() for b in B], [2.5, 97.5])
        out[nm] = [round(dd[m].mean(), 3), [round(ci[0], 3), round(ci[1], 3)]]
    return out


def table(full):
    D, CHG, RR = load_full() if full else (load_cache(), None, None); cells, sig = {}, {}
    for sp in SPLITS:
        o = D[sp]["ours"]; al = np.arange(len(o["rg"])); po = stats(o, al)
        for key, _, _, rb, sb in SYSTEMS:
            d = D[sp][key]; pt = stats(d, al)
            ci = boot(o["pat"], lambda ix: stats(d, ix)); r = {k: [round(v, 4), ci[k]] for k, v in pt.items()}
            if rb and full:
                r |= reader_cell(RR[sp], rb, key, CHG[sp], o["pat"])
            cells.setdefault(sp, {})[key] = r
            if key == "ours":
                continue
            cd = boot(o["pat"], lambda ix: (lambda a, b: {k: a[k] - b[k] for k in a})(stats(o, ix), stats(d, ix)))
            s = {k: [round(po[k] - pt[k], 4), cd[k]] for k in pt}
            if sb and full:
                s |= reader_sig(RR[sp], sb, key, CHG[sp], o["pat"])
            sig.setdefault(sp, {})[key] = s
            print(sp, key, "done", flush=True)
    if full:
        return {"mode": "full", "cells": cells, "sig": sig, "from_golden": []}
    gold = json.load(open(f"{H}/table2_golden.json")); fg = []
    for part, t in (("cells", cells), ("sig", sig)):
        for sp in gold[part]:
            for k, v in gold[part][sp].items():
                for m, x in v.items():
                    if m not in t[sp][k]:
                        t[sp][k][m] = x; fg.append(f"{part}/{sp}/{k}/{m}")
    return {"mode": "cache: rger / f1temp recomputed, the rest copied from table2_golden.json", "cells": cells, "sig": sig, "from_golden": fg}


def show(t):
    f = lambda m, v: f"{v[0]:.2f} [{v[1][0]:.2f}, {v[1][1]:.2f}]" if m.startswith("readers") else f"{v[0]:.3f} [{v[1][0]:.3f}, {v[1][1]:.3f}]"
    fg = set(t["from_golden"])
    for sp in SPLITS:
        print(f"\n{sp} ('*' = BrainDiff - system CI excludes 0; 'g' = copied from table2_golden.json, not recomputed)\n| System | "
              + " | ".join(METRICS) + " |\n|---|" + "---|" * len(METRICS))
        for key, lab, *_ in SYSTEMS:
            r, s = t["cells"][sp][key], t["sig"][sp].get(key, {})
            star = lambda m: "*" if m in s and (s[m][1][0] > 0 or s[m][1][1] < 0) else ""
            g = lambda m: " g" if f"cells/{sp}/{key}/{m}" in fg else ""
            print(f"| {lab} | " + " | ".join(f(m, r[m]) + star(m) + g(m) if m in r else "-" for m in METRICS) + " |")


def build_cache():
    """Full-mode inputs -> paper/cache: per split groups_<split>.npz (patient index) and <system>_<split>.npz (rger, f1temp) only."""
    D, _, _ = load_full(); os.makedirs(C, exist_ok=True)
    for sp in SPLITS:
        np.savez_compressed(f"{C}/groups_{sp}.npz", group=D[sp]["ours"]["pat"].astype(np.int32))
        for k, d in D[sp].items():
            np.savez_compressed(f"{C}/{k}_{sp}.npz", rger=d["rg"].astype(np.float64), f1temp=d["ft"].astype(np.float64))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true"); ap.add_argument("--build_cache", action="store_true"); ap.add_argument("--out", default=f"{H}/table2.json")
    a = ap.parse_args()
    if a.build_cache:
        return build_cache()
    t = table(a.full); json.dump(t, open(a.out, "w"), indent=1); show(t)


if __name__ == "__main__":
    main()
