"""Score one generation file for paper Table 2 (ours and baselines). Scoring environment, one GPU.

    python -m braindiff.eval.score <dir>/gen_<tag>.jsonl [--copy_prior]
    python -m braindiff.eval.score assemble <baseline raw gen.jsonl> <our gen.jsonl> <out gen.jsonl>

Per pair, against:
  the ORIGINAL current report's statuses   status presence (noisy-OR of Rad-Δ unit probs) -> macro-MCC / macro-F1 / AUC
  the ORIGINAL current report              finding-level Rad-Δ change agreement, tp / fp / fn
  the edited target                        rg_er = index [1] of F1RadGraph(reward_level="all", model_type="radgraph-xl");
                                           F1temp = Libra's temporal F1
-> score_<tag>.json, score_<tag>.npz, gen_<tag>.agree.npz, perpair_<split>.parquet next to the file (read by
   paper/table2.py --build_cache). --copy_prior also scores the edited prior report as the generation.
Rad-Δ checkpoint: --rgd_ckpt, else $RADDELTA_CKPT, else raddelta's default.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from braindiff.eval.temporal_score import f1temp
from raddelta.metric import CHANGES, RadDelta, annotate_long, change_agreement, change_counts, units

LML, COLS = ["NEW", "WORSENED", "IMPROVED", "RESOLVED"], [0, 1, 2, 4]


def presence(ann):
    """Status presence [4] = 1 - prod(1 - p) over Observation units (p rounded to 4 decimals, as the Stage-1 labels)."""
    ents = ann["entities"]
    probs = [np.mean([x["probs"] for x in ents.values() if x.get("unit") == k], axis=0).round(4).tolist()
             for k, e in ents.items() if e["head"]]
    if not probs:
        return np.zeros(4)
    p = np.clip(np.array(probs)[:, COLS], 0, 1 - 1e-12)
    return 1 - np.exp(np.log1p(-p).sum(0))


def change_metrics(P, Y):
    """P [n,4] generated probabilities, Y [n,4] bool reference presence."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    H = P >= 0.5; ha, ya, pa = H.any(1), Y.any(1), 1 - np.prod(1 - P, 1)
    out = {s: dict(f1=round(f1_score(Y[:, j], H[:, j], zero_division=0), 3),
                   auc=round(roc_auc_score(Y[:, j], P[:, j]), 3) if 0 < Y[:, j].sum() < len(Y) else None,
                   pred_rate=round(float(H[:, j].mean()), 3)) for j, s in enumerate(LML)}
    out["macro_f1"] = round(float(np.mean([out[s]["f1"] for s in LML])), 3)
    out["any_change"] = dict(f1=round(f1_score(ya, ha, zero_division=0), 3), bal_acc=round(balanced_accuracy_score(ya, ha), 3),
                             auc=round(roc_auc_score(ya, pa), 3), pred_rate=round(float(ha.mean()), 3),
                             true_rate=round(float(ya.mean()), 3))
    return out


def rger_scorer():
    from radgraph import F1RadGraph
    f = F1RadGraph(reward_level="all", model_type="radgraph-xl", cuda=0)

    def rger(refs, hyps):
        _, rewards, _, _ = f(refs=refs, hyps=hyps)
        return list(rewards[1])                    # reward_level "all" -> (rg_e, rg_er, rg_bar_er) lists
    return rger


def safe(fn, refs, hyps):
    """Score a chunk; on GPU out-of-memory one pair at a time, NaN for pairs that still fail."""
    import torch
    try:
        return fn(refs, hyps)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        out = []
        for r, h in zip(refs, hyps):
            try:
                out.append(fn([r], [h])[0])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); out.append(float("nan"))
        return out


def score(path, copy_prior=False, rgd_ckpt=None):
    rows = [json.loads(l) for l in open(path)]
    Y = np.array([r["statuses"] for r in rows], bool)
    ckpt = rgd_ckpt or os.environ.get("RADDELTA_CKPT")
    m = RadDelta(ckpt, cuda=0) if ckpt else RadDelta(cuda=0)
    rger = rger_scorer()
    refs = [r["target"] for r in rows]
    cache, res, arr = {}, {"n": len(rows)}, {}
    pp = {"pair_id": [r["pair_id"] for r in rows]}
    ag = {k: [] for k in ("pair_id", "patient", "agree", "n_pairs", "ref_changes", "tp", "fp", "fn")}
    for r in rows:
        ref = annotate_long(m, r["current_orig"], cache)
        h = annotate_long(m, r["generated"], cache)
        a, n = change_agreement(h, ref)
        for k, v in zip(("pair_id", "patient", "ref_changes", "agree", "n_pairs", "tp", "fp", "fn"),
                        (r["pair_id"], r["patient"], sum(u[0] in CHANGES for u in units(ref)), a, n) + change_counts(h, ref)):
            ag[k].append(v)
    hs = [("model", [r["generated"] for r in rows], "")] + \
        ([("copy_prior", [r["prior"] for r in rows], "_copy")] if copy_prior else [])
    for k, h, suf in hs:
        P = arr[k] = np.stack([presence(annotate_long(m, t or "", cache)) for t in h])
        out = {"all": change_metrics(P, Y)}
        H = P >= 0.5
        out["recall"] = {s: round(float(H[Y[:, j], j].mean()), 3) if Y[:, j].any() else None for j, s in enumerate(LML)}
        arr[f"agree_{k}"] = (np.array(ag["agree"]) if k == "model" else
                             np.array([change_agreement(annotate_long(m, x, cache), annotate_long(m, r_["current_orig"], cache))[0]
                                       for x, r_ in zip(h, rows)]))
        out["agree"] = round(float(arr[f"agree_{k}"].mean()), 4)
        hyp = [x or "." for x in h]
        sc = []
        for i in range(0, len(rows), 64):
            sc += list(safe(rger, refs[i:i + 64], hyp[i:i + 64]))
        pp[f"rger{suf}"] = np.array(sc, float)
        pp[f"f1temp{suf}"] = np.array(f1temp(refs, hyp), float)
        out["rg_er"] = round(float(np.nanmean(pp[f"rger{suf}"])), 4)
        out["mean_words"] = round(float(np.mean([len(x.split()) for x in h])), 1)
        res[k] = out
        a = out["all"]
        print(f"{k:10s} agree {out['agree']:.4f}  rg_er {out['rg_er']:.4f}  any-change AUC {a['any_change']['auc']:.3f} "
              f"macro-F1 {a['macro_f1']:.3f}  F1 " + "  ".join(f"{s} {a[s]['f1']:.2f}" for s in LML), flush=True)
    out = path.replace("gen_", "score_").replace(".jsonl", ".json")
    json.dump(res, open(out, "w"), indent=1)
    np.savez(out.replace(".json", ".npz"), pair_id=np.array([r["pair_id"] for r in rows]),
             patient=np.array([r.get("patient", "") for r in rows]), Y=Y, **arr)
    np.savez(path.replace(".jsonl", ".agree.npz"), **{k: np.array(v) for k, v in ag.items()})
    d, tag = os.path.split(path)[0], os.path.basename(path)[len("gen_"):-len(".jsonl")]
    split = re.sub(r"_epoch\d+$", "", tag)
    pd.DataFrame(pp).to_parquet(os.path.join(d, f"perpair_{split}.parquet"))
    print("SCORES", out)


def assemble(raw, ours, out):
    """A baseline's raw generations (pair_id, generated) in our generation-file format, on our rows."""
    g = {json.loads(l)["pair_id"]: json.loads(l)["generated"] for l in open(raw)}
    miss = 0
    with open(out, "w") as f:
        for l in open(ours):
            r = json.loads(l)
            t = (g.get(r["pair_id"]) or "").strip(); miss += not t
            f.write(json.dumps(r | {"generated": t or ".", "status_line": ""}) + "\n")
    print(out, miss, "missing / empty")


def main():
    if sys.argv[1:2] == ["assemble"]:
        assemble(*sys.argv[2:5])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gen"); ap.add_argument("--copy_prior", action="store_true"); ap.add_argument("--rgd_ckpt")
    a = ap.parse_args()
    score(a.gen, a.copy_prior, a.rgd_ckpt)


if __name__ == "__main__":
    main()
