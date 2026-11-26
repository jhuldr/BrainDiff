"""Generate comparison reports and pick the Stage 2 epoch.

    torchrun --standalone --nproc_per_node=4 -m braindiff.eval.generate --ckpt braindiff.pt --split test \
        --repetition_penalty 1.1 --out out/gen_test.jsonl
    python -m braindiff.eval.generate pick <save dir> [--full]

Greedy, prior report present, every BIND pair of the split. --ckpt = the release file, or epochK.pt with --init <its
Stage 1-D checkpoint>. --cond: an image condition of Stage2Dataset.set_condition. Output rows -> eval/score.py.

pick (macro-MCC over NEW / WORSENED / IMPROVED / RESOLVED):
  default  top 3 epochs by the 256-pair in-loop subset (<save>/score_epochK.npz)    -> generate those on full BIND val
  --full   the best of those by full val (<save>/score_val_epochK.npz)
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import timedelta

import numpy as np

STATUS = ("NEW", "WORSENED", "IMPROVED", "RESOLVED")


def mcc(f):
    z = np.load(f, allow_pickle=True)
    Y, H = z["Y"].astype(bool), z["model"] >= 0.5
    out = []
    for j in range(Y.shape[1]):
        y, h = Y[:, j], H[:, j]
        tp, tn, fp, fn = (y & h).sum(), (~y & ~h).sum(), (~y & h).sum(), (y & ~h).sum()
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out.append((tp * tn - fp * fn) / den if den else 0.0)
    return float(np.mean(out))


def top3(d):
    ks = sorted(int(re.findall(r"score_epoch(\d+)\.npz", f)[0]) for f in glob.glob(f"{d}/score_epoch*.npz"))
    s = {k: mcc(f"{d}/score_epoch{k}.npz") for k in ks}
    return sorted(s, key=s.get, reverse=True)[:3]


def chosen(d):
    s = {k: mcc(f"{d}/score_val_epoch{k}.npz") for k in top3(d)}
    return max(s, key=s.get)


def generate():
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from braindiff.data.datasets import PairSampler, Stage2Dataset, collate
    from braindiff.models.braindiff import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--init", default=None)
    ap.add_argument("--split", default="val"); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32, help="global"); ap.add_argument("--max_new_tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="first n pairs only")
    ap.add_argument("--cond", default=None, choices=("correct", "same_cur", "same_prior", "swap"))
    ap.add_argument("--repetition_penalty", type=float, default=None, help="1.1 = the reported decode; default greedy")
    ap.add_argument("--subset", default=None, help="run.json whose eval_pair_ids restrict the split")
    a = ap.parse_args()
    # ranks finish shards at different times; NCCL's 10-min default timeout would kill the wait
    dist.init_process_group("nccl", timeout=timedelta(hours=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"]); dev = f"cuda:{local}"; torch.cuda.set_device(dev)
    model, ck = load_model(a.ckpt, a.init, is_main=rank == 0)
    model.to(dev).eval()
    ds = Stage2Dataset(a.split, augment=False, sources=("BIND",))
    if a.cond:
        ds.set_condition(a.cond)
    if a.subset:
        keep = set(json.load(open(a.subset))["eval_pair_ids"])
        ds.pairs = ds.pairs[ds.pairs.pair_id.isin(keep)].reset_index(drop=True)
    if a.limit:
        ds.pairs = ds.pairs.iloc[:a.limit].reset_index(drop=True)
    dl = DataLoader(ds, batch_sampler=PairSampler(len(ds.pairs), a.batch, world, rank, shuffle=False, drop_last=False),
                    collate_fn=collate, num_workers=8)
    kw = {"repetition_penalty": a.repetition_penalty} if a.repetition_penalty is not None else {}
    rows = []
    for b in dl:
        b = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(b, b["prior"], a.max_new_tokens, **kw)
        st = model.last_status if model.status_first else [None] * len(gen)
        rows += [{"pair_id": p, "generated": g} | ({"status_line": x} if x is not None else {})
                 for p, g, x in zip(b["pair_id"], gen, st)]
    allr = [None] * world
    dist.all_gather_object(allr, rows)
    if rank == 0:
        meta = ds.pairs.set_index("pair_id")
        with open(a.out, "w") as f:
            for r in (r for x in allr for r in x):
                m = meta.loc[r["pair_id"]]
                f.write(json.dumps(r | {"cond": a.cond or "correct", "patient": m.patient, "source": m.source,
                                        "target": m.target, "prior": m.prior_edit, "current_orig": m.current_orig,
                                        "statuses": [bool(m[f"st_{s}"]) for s in STATUS]}) + "\n")
        print(f"[gen] {sum(len(x) for x in allr)} pairs ({a.split}, epoch {ck.get('epoch')}) -> {a.out}", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1:2] == ["pick"]:
        print(chosen(sys.argv[2]) if "--full" in sys.argv else " ".join(map(str, top3(sys.argv[2]))))
    else:
        generate()
