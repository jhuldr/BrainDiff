"""Stage 1-D training and epoch selection (fmt "status").

    torchrun --standalone --nproc_per_node=4 -m braindiff.training.train_stage1 --config configs/stage1.yaml
    python -m braindiff.training.train_stage1 --select checkpoints/stage1        # prints the selected epoch

Change task + single-study tasks (round(task_ratio * n_pairs) task items per epoch, own batches). Eval (BIND val, every
ent_ok pair) at step 0 and after every epoch: auc_nll = any-change AUC from the summed NLL of "No interval change.",
plus per-status recall / macro-F1 from greedy generations; metrics_tasks is a health check only.
Selection: highest val auc_nll, ties to the earlier epoch.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from braindiff.data.datasets import EntTextDataset, MixedBatchSampler, PairSampler, collate, sub_batch
from braindiff.models.prompts import NONE_TEXT, STATUSES
from braindiff.models.stage1 import Stage1Text


def statuses(text):
    return {s for s in STATUSES if re.search(rf"\b{s}\b", text)}


def metrics(rows):
    y = np.array([bool(r["true"]) for r in rows])
    s = np.array([r["score"] for r in rows])
    g = np.array([bool(r["pred"]) for r in rows])
    out = {"n": len(rows), "true_any": round(float(y.mean()), 3), "auc_nll": round(float(roc_auc_score(y, s)), 3),
           "auc_gen": round(float(roc_auc_score(y, g)), 3), "pred_any": round(float(g.mean()), 3),
           "parse_ok": round(float(np.mean([r["gen"].rstrip().endswith("change.") for r in rows])), 3)}
    f1s = []
    for st in STATUSES:
        t = np.array([st in r["true"] for r in rows]); p = np.array([st in r["pred"] for r in rows])
        tp = float((t & p).sum())
        rec, prec = tp / max(t.sum(), 1), tp / max(p.sum(), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
        out[f"rec_{st}"] = round(rec, 3)
    out["macro_f1"] = round(float(np.mean(f1s)), 3)
    return out


def box_of(text):
    m = re.search(r"\[\s*(\d+(?:\s*,\s*\d+){5})\s*\]", text or "")
    return [int(x) for x in m.group(1).split(",")] if m else None


def iou(a, b):
    """3D IoU of [x1, x2, y1, y2, z1, z2] boxes (inclusive integer percent coordinates); 0 for an unparsed box."""
    if a is None or b is None:
        return 0.0
    inter = 1.0
    for k in range(3):
        lo, hi = max(a[2 * k], b[2 * k]), min(a[2 * k + 1], b[2 * k + 1])
        inter *= max(hi - lo + 1, 0)
    va = np.prod([max(a[2 * k + 1] - a[2 * k] + 1, 0) for k in range(3)])
    vb = np.prod([max(b[2 * k + 1] - b[2 * k] + 1, 0) for k in range(3)])
    return float(inter / max(va + vb - inter, 1))


def metrics_tasks(rows):
    out = {}
    for t in ("aref", "gcap", "caref", "pcls"):
        r = [x for x in rows if x["task"] == t]
        if not r:
            continue
        o = {"n": len(r)}
        if t in ("aref", "caref"):
            io = np.array([iou(box_of(x["gen"]), box_of(x["target"])) for x in r])
            les = np.array([x["lesion"] for x in r])
            o["parse_ok"] = round(float(np.mean([box_of(x["gen"]) is not None for x in r])), 3)
            for name, m in (("lesion", les), ("all", np.ones(len(r), bool))):
                if m.any():
                    o[f"iou_{name}"] = round(float(io[m].mean()), 3)
                    o[f"iou25_{name}"] = round(float((io[m] >= .25).mean()), 3)
                    o[f"iou50_{name}"] = round(float((io[m] >= .5).mean()), 3)
        if t in ("gcap", "caref"):
            cap = lambda x: re.sub(r"\s*\[.*", "", x).strip()
            o["exact"] = round(float(np.mean([cap(x["gen"]) == cap(x["target"]) for x in r])), 3)
            o["lesion_acc"] = round(float(np.mean([("unremarkable" in x["gen"]) == (not x["lesion"]) for x in r])), 3)
        if t == "pcls":
            lab = lambda s: {w.strip() for w in s.split(",") if w.strip()}
            P, T = [lab(x["gen"]) for x in r], [lab(x["target"]) for x in r]
            names = sorted(set().union(*T))
            tp = sum(len(p & q) for p, q in zip(P, T)); fp = sum(len(p - q) for p, q in zip(P, T))
            fn = sum(len(q - p) for p, q in zip(P, T))
            o["micro_f1"] = round(2 * tp / max(2 * tp + fp + fn, 1), 3)
            f1 = []
            for c in names:
                a = sum(c in p and c in q for p, q in zip(P, T)); b = sum(c in p for p in P); d = sum(c in q for q in T)
                f1.append(2 * a / max(b + d, 1))
            o["macro_f1"] = round(float(np.mean(f1)), 3)
            o["exact"] = round(float(np.mean([p == q for p, q in zip(P, T)])), 3)
        out[t] = o
    return out


def select(save):
    """Selected epoch: max val auc_nll in <save>/eval_history.json, ties to the earlier epoch."""
    h = json.load(open(os.path.join(save, "eval_history.json")))
    ep = [(v["auc_nll"], -int(k[len("epoch"):])) for k, v in h.items() if k.startswith("epoch") and v]
    return -max(ep)[1]


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--select")
    a = ap.parse_args()
    if a.select:
        print(select(a.select))
        return
    cfg = yaml.safe_load(open(a.config))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = f"cuda:{int(os.environ['LOCAL_RANK'])}"; torch.cuda.set_device(dev)
    torch.manual_seed(cfg["seed"])
    save = os.path.join(cfg["save_dir"], cfg["save_name"]); os.makedirs(save, exist_ok=True)
    say = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    tasks = tuple(cfg["tasks"])
    tr = EntTextDataset("train", augment=True, tasks=tasks)
    va = EntTextDataset("val", augment=False, tasks=tasks)
    say(f"[data] tasks {tasks}: train {tr.tasks.task.value_counts().to_dict()}, val {va.tasks.task.value_counts().to_dict()}, "
        f"task_ratio {cfg['task_ratio']}")
    if cfg.get("eval_n"):                                   # smoke tests only
        va.pairs = va.pairs.iloc[:cfg["eval_n"]].reset_index(drop=True)
    say(f"[data] train {len(tr.pairs)} pairs ({tr.pairs.source.value_counts().to_dict()}, {tr.n_not_ok} ent_ok-False "
        f"dropped), val {len(va.pairs)} BIND pairs ({va.n_not_ok} dropped); train any-change {tr.pairs.any_change.mean():.3f}, "
        f"val {va.pairs.any_change.mean():.3f}")
    true_st = {int(p): set(t.split(",")) - {""} for p, t in zip(va.pairs.pair_id, va.pairs.text_statuses)}
    smp = MixedBatchSampler(tr, cfg["batch"], world_size=world, rank=rank, seed=cfg["seed"], task_ratio=cfg["task_ratio"])
    say(f"[data] {len(smp)} steps/epoch")
    dl = DataLoader(tr, batch_sampler=smp, collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    vdl = DataLoader(va, batch_sampler=PairSampler(len(va.pairs), cfg["eval_batch"], world, rank, shuffle=False,
                                                   drop_last=False),
                     collate_fn=collate, num_workers=cfg["num_workers_eval"])
    rng = np.random.default_rng(0)                          # fixed val subset per task, sharded over ranks
    tix = np.concatenate([rng.permutation(np.where(va.tasks.task == t)[0])[:cfg["task_eval_n"]] for t in tasks])
    keys = [("task", int(i), 0, 0) for i in tix[rank::world]]
    eb = cfg["eval_batch"] // world
    tdl = DataLoader(va, batch_sampler=[keys[lo:lo + eb] for lo in range(0, len(keys), eb)], collate_fn=collate,
                     num_workers=cfg["num_workers_eval"])

    model = Stage1Text(tuple(cfg["lr"]), is_main=rank == 0).to(dev)
    groups = [{"params": model.groups[k], "lr": lr} for k, lr in cfg["lr"].items()]
    params = [p for g in groups for p in g["params"]]
    say(f"[model] {sum(p.numel() for p in params) / 1e6:.1f}M trainable; decoder LoRA {len(model.groups['lora'])} tensors")
    ddp = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False)
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    total = min(len(smp) * cfg["epochs"], cfg.get("max_steps") or 10 ** 9)
    sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_frac"] * total), total)
    mb = cfg["micro_batch"]
    say(f"[train] global batch {cfg['batch']} ({cfg['batch'] // world} per rank, micro {mb}), {len(smp)} steps/epoch, "
        f"{total} steps")

    def evaluate(tag):
        model.eval(); rows = []
        for b in vdl:
            b = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sc, gen = model.score(b, NONE_TEXT, cfg["max_new_tokens"])
            for pid, pt, s_, g_ in zip(b["pair_id"], b["patient"], sc.tolist(), gen):
                rows.append({"pair_id": pid, "patient": pt, "score": s_, "gen": g_,
                             "true": sorted(true_st[pid]), "pred": sorted(statuses(g_))})
        trows = []
        for b in tdl:
            b = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(b, cfg["task_max_new_tokens"])
            trows += [{"task": t, "target": x, "gen": g_, "lesion": les}
                      for t, x, g_, les in zip(b["task"], b["text"], gen, b["lesion"])]
        allr, allt = [None] * world, [None] * world
        dist.all_gather_object(allr, rows)
        dist.all_gather_object(allt, trows)
        model.train()
        if rank:
            return None
        rows, trows = [r for x in allr for r in x], [r for x in allt for r in x]
        with open(f"{save}/eval_tasks_{tag}.jsonl", "w") as f:
            for r in trows:
                f.write(json.dumps(r) + "\n")
        say(f"[eval {tag}] tasks " + json.dumps(metrics_tasks(trows)))
        with open(f"{save}/eval_{tag}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        m = metrics(rows)
        say(f"[eval {tag}] " + json.dumps(m))
        return m

    t0 = time.time()
    hist = {"step0": evaluate("step0")}
    say(f"[eval] {(time.time() - t0) / 60:.1f} min")
    step, t0, run = 0, time.time(), []
    model.train()
    for ep in range(cfg["epochs"]):
        smp.set_epoch(ep)
        tw = time.time(); wait = 0.0
        for b in dl:
            wait += time.time() - tw
            b = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            n = b["n_items"]
            parts = [list(range(lo, min(lo + mb, n))) for lo in range(0, n, mb)]
            for j, ix in enumerate(parts):
                ctx = ddp.no_sync() if j < len(parts) - 1 else _null()
                with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = ddp(sub_batch(b, ix))
                    (loss * len(ix) / n).backward()
                run.append(float(loss))
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % cfg["log_every"] == 0:
                say(f"[train] step {step}/{total} ep {ep} loss {np.mean(run[-cfg['log_every'] * len(parts):]):.4f} "
                    f"gnorm {float(gn):.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f} s/step "
                    f"(loader wait {wait / max(step - ep * len(smp), 1):.2f} s/step)")
            tw = time.time()
            if step >= total:
                break
        if rank == 0:
            sd = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save({"model": sd, "step": step, "epoch": ep + 1, "cfg": cfg}, f"{save}/epoch{ep + 1}.pt")
            say(f"[save] {save}/epoch{ep + 1}.pt: {len(sd)} tensors")
        hist[f"epoch{ep + 1}"] = evaluate(f"epoch{ep + 1}")
        say(f"[epoch {ep + 1}] train CE {np.mean(run[-len(smp) * len(parts):]):.4f}  {(time.time() - t0) / 60:.1f} min")
    if rank == 0:
        json.dump(hist, open(f"{save}/eval_history.json", "w"), indent=1)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
