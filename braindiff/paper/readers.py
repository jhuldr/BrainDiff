#!/usr/bin/env python3
"""Table 2 reader columns: blinded LLM readers grade each system's report with the change-question brief (reader_brief.txt).
  build   500 pairs per site; systems mixed and shuffled within a batch, 10% duplicates (reliability); ~73 items per packet.
  score   change_score (1-5) per system and site on pairs read by every system, change / stable split; REF - system
          paired, patient bootstrap 2,000 (seed 0). First read of each item scored.
    python paper/readers.py build <batch> <system> [<system> ...]    systems = run dirs under $BRAINDIFF_CKPT/table2_baselines
    REF=ours_rp110 python paper/readers.py score <batch> [<batch> ...]  -> $READER_DIR/result_<batches>.json
Env: BRAINDIFF_CKPT (default ./checkpoints), READER_DIR (default $BRAINDIFF_CKPT/readers/table2_baselines_cq), READER_SAMPLE.
Packets and answers contain report text and are not released; Table 2 is rebuilt from paper/cache instead."""
import glob
import json
import os
import sys

import numpy as np

CK = os.environ.get("BRAINDIFF_CKPT", "checkpoints"); D = os.environ.get("READER_DIR", f"{CK}/readers/table2_baselines_cq")
SAMPLE = os.environ.get("READER_SAMPLE", f"{CK}/readers/table2_random_2026-11-11/key/key.json")
GREEDY = f"{CK}/stage2_A0_s1d_sf"; CH = ("new", "worse", "better", "resolved")


def build(name, systems):
    rng = np.random.default_rng(4 + len(systems))   # seed fixed by system count: reproduces the stored batches
    k276 = json.load(open(SAMPLE))["items"]
    pids = {sp: sorted({v["pair_id"] for v in k276.values() if v["split"] == sp}) for sp in ("test", "external_test")}
    if os.environ.get("PAIRS"):   # subset run: only the pairs in this json {split: [pair_id]}
        sub = json.load(open(os.environ["PAIRS"])); pids = {sp: [p for p in pids[sp] if p in set(sub[sp])] for sp in pids}
    items = []
    for sp in ("test", "external_test"):
        if not pids[sp]: continue   # subset runs may cover one split only
        ours = {r["pair_id"]: r for r in map(json.loads, open(f"{GREEDY}/gen_{sp}_epoch20.jsonl"))}
        gens = {s: ({p: r["generated"] for p, r in ours.items()} if s == "ours" else
                    {r["pair_id"]: r["generated"] for r in map(json.loads, open(f"{CK}/table2_baselines/{s}/gen_{sp}_epoch20.jsonl"))}) for s in systems}
        for p in pids[sp]:
            r = ours[p]
            for s in systems:
                items.append(dict(split=sp, pair_id=p, patient=r["patient"], system=s, PRIOR=r["prior"] or ".", REFERENCE=r["current_orig"], CANDIDATE=gens[s][p] or "."))
    O = f"{D}/{name}"; os.makedirs(f"{O}/packets", exist_ok=True); os.makedirs(f"{O}/answers", exist_ok=True)
    P = round(len(items) * 1.1 / int(os.environ.get("PACKET", 73))); order = list(rng.permutation(len(items))); packets = [order[k * len(order) // P:(k + 1) * len(order) // P] for k in range(P)]
    for j, x in enumerate(rng.choice(len(items), len(items) // 10, replace=False)):
        home = next(k for k in range(P) if x in packets[k]); packets[(home + 1 + j % (P - 1)) % P].append(x)
    key = {}; c = 0
    for k, pk in enumerate(packets):
        with open(f"{O}/packets/packet_{k}.jsonl", "w") as f:
            for x in rng.permutation(pk):
                it = items[x]; iid = f"{name[0]}{c:05d}"; c += 1; key[iid] = {kk: it[kk] for kk in ("split", "pair_id", "patient", "system")} | {"item": int(x)}
                f.write(json.dumps({"id": iid, "PRIOR": it["PRIOR"], "REFERENCE": it["REFERENCE"], "CANDIDATE": it["CANDIDATE"]}) + "\n")
    json.dump({"systems": systems, "items": key}, open(f"{O}/key.json", "w"), indent=1)
    print(name, systems, len(items), "items;", P, "packets", [len(p) for p in packets][:5], "...")


def first_reads(b):
    """[(key of the item's first answered copy, its answer, answer of the second copy or None)] per item of batch b, as score.py."""
    key = json.load(open(f"{D}/{b}/key.json"))["items"]; ans = {}
    for f in glob.glob(f"{D}/{b}/answers/answers_p*.jsonl"):
        for l in open(f):
            if l.strip(): a = json.loads(l); ans[a["id"]] = a
    by = {}
    for i in sorted(key): by.setdefault(key[i]["item"], []).append(i)
    out = []
    for v in by.values():
        v = [i for i in v if i in ans]
        if v: out.append((key[v[0]], ans[v[0]], ans[v[1]] if len(v) > 1 else None))
    return out, len(ans), len(key)


def score(batches):
    REF = os.environ.get("REF", "ours")   # reference system for the paired difference
    rows = {}; dup = []
    for b in batches:
        reads, na, nk = first_reads(b); print(f"{b}: answers {na} / {nk}")
        for k, a, a2 in reads:
            cl = [c for c in a["claims"] if c.get("type") in CH]
            rows[(k["split"], k["system"], k["pair_id"])] = dict(cs=a["change_score"], inv=a["invented_findings"], om=a["omitted_findings"], miss=a["missed_changes"],
                                                                ncl=len(cl), ok=sum(c["label"] == "correct" for c in cl), pat=k["patient"])
            if a2 is not None: dup.append((a["change_score"], a2["change_score"]))
    d = np.array(dup); print(f"duplicates {len(d)}: exact {np.mean(d[:,0]==d[:,1]):.2f}, within 1 {np.mean(abs(d[:,0]-d[:,1])<=1):.2f}")
    systems = sorted({s for _, s, _ in rows}, key=lambda s: (s != REF, s)); out = {}
    for sp in ("test", "external_test"):
        s = np.load(f"{GREEDY}/score_{sp}_epoch20.npz", allow_pickle=True); chg = dict(zip(s["pair_id"].tolist(), s["Y"].astype(bool).any(1).tolist()))
        pids = sorted({p for (x, y, p) in rows if x == sp and all((sp, z, p) in rows for z in systems)})
        pat = np.array([rows[(sp, REF, p)]["pat"] for p in pids]); up = np.unique(pat); idx = {u: np.where(pat == u)[0] for u in up}
        rng = np.random.default_rng(0); B = [np.concatenate([idx[u] for u in rng.choice(up, len(up))]) for _ in range(2000)]
        c = np.array([chg[p] for p in pids])
        print(f"\n{sp}: {len(pids)} pairs with every system read ({c.sum()} change / {(~c).sum()} stable refs)")
        for sy in systems:
            v = {f: np.array([rows[(sp, sy, p)][f] for p in pids], float) for f in ("cs", "inv", "om", "miss", "ncl", "ok")}
            r = dict(change_score=round(v["cs"].mean(), 3), change_refs=round(v["cs"][c].mean(), 3), stable_refs=round(v["cs"][~c].mean(), 3),
                     invented=round(v["inv"].mean(), 3), omitted=round(v["om"].mean(), 3), missed_change_refs=round(v["miss"][c].mean(), 3),
                     claim_precision=round(v["ok"].sum() / max(v["ncl"].sum(), 1), 3))
            if sy != REF:
                o = np.array([rows[(sp, REF, p)]["cs"] for p in pids], float); dd = o - v["cs"]
                for nm, m in (("all", np.ones(len(pids), bool)), ("change", c), ("stable", ~c)):
                    ci = np.percentile([dd[b][m[b]].mean() for b in B], [2.5, 97.5])
                    r[f"ours_minus_{nm}"] = [round(dd[m].mean(), 3), [round(ci[0], 3), round(ci[1], 3)]]
            out.setdefault(sp, {})[sy] = r; print(f"  {sy:9s} {r}")
    json.dump(out, open(f"{D}/result_{'_'.join(batches)}.json", "w"), indent=1)


if __name__ == "__main__":
    {"build": lambda a: build(a[0], a[1:]), "score": score}[sys.argv[1]](sys.argv[2:])
