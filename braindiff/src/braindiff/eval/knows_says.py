"""Knows vs says for Fig. 2.

    torchrun --standalone --nproc_per_node=4 -m braindiff.eval.knows_says knows --ckpt <ckpt> [--init <stage1>] \
        --out <dir>/tf_direction_canon_epochK
    python -m braindiff.eval.knows_says committed <spec.json> <out.json>

knows: on BIND val pairs with exactly one of WORSENED / IMPROVED, score = mean over CANONICAL wordings of
NLL(improving) - NLL(worsening); AUC per image condition, paired patient bootstrap. -> <out>.json + <out>.npz.
says: W-vs-I AUC of Rad-Δ P[W] - P[I] on the greedy report of the same pairs.
committed: spec = {model: [{epoch, knows_file, says_file}]}; knows / says / gap on all pairs and on pairs whose report
commits to a direction (max(P[W], P[I]) >= 0.5).
"""
import argparse
import json
import os
import sys
from datetime import timedelta

import numpy as np
from sklearn.metrics import roc_auc_score

CONDS = ("correct", "same_cur", "same_prior", "swap")
CANONICAL = [("Findings: Interval worsening compared with the prior study.",
              "Findings: Interval improvement compared with the prior study."),
             ("Findings: The abnormality has increased since the prior study.",
              "Findings: The abnormality has decreased since the prior study."),
             ("Findings: There is interval progression.", "Findings: There is interval regression.")]
TH = 0.5


def knows():
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from braindiff.data.datasets import PairSampler, Stage2Dataset, collate
    from braindiff.models.braindiff import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--init", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32, help="global")
    ap.add_argument("--conds", default=",".join(CONDS), help="comma-separated image conditions (default all four)")
    a = ap.parse_args(sys.argv[2:])
    conds = tuple(a.conds.split(",")); assert conds[0] == "correct" and set(conds) <= set(CONDS), conds
    dist.init_process_group("nccl", timeout=timedelta(hours=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"]); dev = f"cuda:{local}"; torch.cuda.set_device(dev)
    model, _ = load_model(a.ckpt, a.init, is_main=rank == 0)
    model.to(dev).eval()
    res = {}
    for cond in conds:
        ds = Stage2Dataset("val", augment=False, sources=("BIND",)).set_condition(cond)
        ds.pairs = ds.pairs[ds.pairs.st_WORSENED != ds.pairs.st_IMPROVED].reset_index(drop=True)
        dl = DataLoader(ds, batch_sampler=PairSampler(len(ds.pairs), a.batch, world, rank, shuffle=False, drop_last=False),
                        collate_fn=collate, num_workers=8)
        rows = []
        for b in dl:
            b = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            n = len(b["text"])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                prefix, pattn = model.prefix(*model.visual(b), b["prior"])
                s = 0
                for w, i in CANONICAL:
                    nw, kw = model.nll(prefix, pattn, [w] * n)
                    ni, ki = model.nll(prefix, pattn, [i] * n)
                    s = s + ((ni * ki).sum(1) - (nw * kw).sum(1)).float() / len(CANONICAL)
            rows += list(zip(b["pair_id"], s.cpu().tolist()))
        allr = [None] * world
        dist.all_gather_object(allr, rows)
        if rank == 0:
            res[cond] = {p: sc for x in allr for p, sc in x}
            print(f"[knows] {cond}: {len(res[cond])} pairs", flush=True)
        dist.barrier()
    if rank == 0:
        meta = ds.pairs.set_index("pair_id")
        ids = np.array(sorted(res["correct"]))
        y = meta.loc[ids, "st_WORSENED"].astype(bool).values
        pat = meta.loc[ids, "patient"].values
        S = {c: np.array([res[c][p] for p in ids]) for c in conds}
        up = np.unique(pat); by = {u: np.flatnonzero(pat == u) for u in up}
        rng = np.random.default_rng(0)
        boot = [np.concatenate([by[u] for u in rng.choice(up, len(up))]) for _ in range(1000)]
        auc = lambda c, ix=slice(None): roc_auc_score(y[ix], S[c][ix])
        out = {"ckpt": a.ckpt, "n": f"{int(y.sum())}/{int((~y).sum())}", "auc": {c: round(auc(c), 3) for c in conds}}
        for c in conds[1:]:
            g = [auc("correct", ix) - auc(c, ix) for ix in boot if 0 < y[ix].sum() < len(ix)]
            out[f"correct-{c}"] = [round(auc("correct") - auc(c), 3)] + [round(float(v), 3) for v in np.percentile(g, [2.5, 97.5])]
        json.dump(out, open(a.out + ".json", "w"), indent=1)
        np.savez(a.out + ".npz", pair_id=ids, patient=pat, y=y, **{c: S[c] for c in conds})
        print(json.dumps(out), flush=True)
    dist.barrier(); dist.destroy_process_group()


def auc_ci(y, kn, sy, pat, rng):
    A = roc_auc_score
    up = np.unique(pat); by = {u: np.flatnonzero(pat == u) for u in up}
    bs = [ix for ix in (np.concatenate([by[u] for u in rng.choice(up, len(up))]) for _ in range(2000)) if 0 < y[ix].sum() < len(ix)]
    q = lambda v: np.round(np.percentile(v, [2.5, 97.5]), 3).tolist()
    r = {"n": int(len(y)), "n_W": int(y.sum()), "n_I": int((~y).sum()), "knows": [round(A(y, kn), 3), q([A(y[ix], kn[ix]) for ix in bs])]}
    if sy is not None:
        bk, bsy = np.array([A(y[ix], kn[ix]) for ix in bs]), np.array([A(y[ix], sy[ix]) for ix in bs])
        r |= {"says": [round(A(y, sy), 3), q(bsy)], "gap": [round(A(y, kn) - A(y, sy), 3), q(bk - bsy)]}
    return r


def committed(spec, out_path):
    out = {"threshold": TH}
    for nm, rows in json.load(open(spec)).items():
        out[nm] = []
        for row in rows:
            rng = np.random.default_rng(0)
            t = np.load(row["knows_file"], allow_pickle=True); z = np.load(row["says_file"], allow_pickle=True)
            ids, y, pat, kn = t["pair_id"], t["y"].astype(bool), t["patient"], t["correct"]
            P = dict(zip(z["pair_id"], z["model"])); P = np.array([P[p] for p in ids]); sy = P[:, 1] - P[:, 2]
            cm = np.maximum(P[:, 1], P[:, 2]) >= TH
            r = {"epoch": row["epoch"], "commit_rate": round(float(cm.mean()), 3),
                 "commit_rate_W": round(float(cm[y].mean()), 3), "commit_rate_I": round(float(cm[~y].mean()), 3),
                 "all": auc_ci(y, kn, sy, pat, rng), "committed": auc_ci(y[cm], kn[cm], sy[cm], pat[cm], rng),
                 "uncommitted_knows": auc_ci(y[~cm], kn[~cm], None, pat[~cm], rng),
                 "says_correct_dir_committed": round(float(((sy[cm] > 0) == y[cm]).mean()), 3)}
            out[nm].append(r); a, c = r["all"], r["committed"]
            print(f"{nm:20s} ep{r['epoch']:<3d} commit {r['commit_rate']:.2f}  ALL k/s/gap {a['knows'][0]:.3f}/{a['says'][0]:.3f}/"
                  f"{a['gap'][0]:+.3f}  COMMITTED n{c['n']} k/s/gap {c['knows'][0]:.3f}/{c['says'][0]:.3f}/{c['gap'][0]:+.3f}",
                  flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["knows"]:
        knows()
    elif sys.argv[1:2] == ["committed"]:
        committed(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
