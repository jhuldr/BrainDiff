"""Stage 2 training with in-loop validation.

    torchrun --standalone --nproc_per_node=4 -m braindiff.training.train_stage2 --config configs/stage2.yaml

Init: the selected Stage 1-D epoch (`init_vision`). Train = BIND I0001 train + every MR-RATE pair; captions
>= max_caption tokens dropped. SGDR: each restart_every-epoch cycle gets its own warm-up + cosine, AdamW state kept.
Each epoch: caption CE (health check only) and greedy generations on an eval_n-pair val subset, scored in the background
by eval/score.py with $BRAINDIFF_SCORE_PYTHON (default this interpreter). Selection: eval/generate.py pick.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from braindiff.data.datasets import PairSampler, Stage2Dataset, collate, sub_batch
from braindiff.models.braindiff import BrainDiff
from braindiff.training.train_stage1 import _null


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    ap.add_argument("--num_workers", type=int, default=None, help="loader workers per rank")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["num_workers"] = a.num_workers or cfg["num_workers"]
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    dev = f"cuda:{local}"; torch.cuda.set_device(dev)
    torch.manual_seed(cfg["seed"])
    save = os.path.join(cfg["save_dir"], cfg["save_name"]); os.makedirs(save, exist_ok=True)
    say = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    model = BrainDiff(cfg["init_vision"], cfg["lora_r"], cfg["lora_alpha"], cfg["max_prompt"], cfg["max_report_tokens"],
                      is_main=rank == 0, train_connector=cfg["train_connector"], status_first=cfg["status_first"]).to(dev)
    tok = model.decoder.tokenizer
    tr = Stage2Dataset("train", augment=True, max_target_tokens=cfg["max_caption"], tokenizer=tok,
                       status_first=cfg["status_first"])
    va = Stage2Dataset("val", augment=False, sources=("BIND",), max_target_tokens=cfg["max_caption"], tokenizer=tok,
                       status_first=cfg["status_first"])
    ev = np.sort(np.random.default_rng(0).permutation(len(va.pairs))[:cfg["eval_n"]])
    va.pairs = va.pairs.iloc[ev].reset_index(drop=True)
    say(f"[data] train {len(tr.pairs)} pairs ({tr.pairs.source.value_counts().to_dict()}; {tr.n_long} with targets >= "
        f"{cfg['max_caption']} tokens dropped), change pairs {tr.pairs.any_change.mean():.3f}; eval {len(va.pairs)} BIND "
        f"val pairs (change {va.pairs.any_change.mean():.3f})")
    if cfg.get("max_steps_smoke"):                          # smoke tests only
        tr.pairs = tr.pairs.iloc[:cfg["batch"] * cfg["max_steps_smoke"]].reset_index(drop=True)
    smp = PairSampler(len(tr.pairs), cfg["batch"], world, rank, cfg["seed"])
    dl = DataLoader(tr, batch_sampler=smp, collate_fn=collate, num_workers=cfg["num_workers"], pin_memory=True,
                    persistent_workers=True, prefetch_factor=4)
    vdl = DataLoader(va, batch_sampler=PairSampler(len(va.pairs), cfg["eval_batch"], world, rank, shuffle=False,
                                                   drop_last=False),
                     collate_fn=collate, num_workers=cfg["num_workers_eval"])
    if rank == 0:
        json.dump({"eval_pair_ids": va.pairs.pair_id.tolist(), "cfg": cfg}, open(f"{save}/run.json", "w"))

    params = [p for p in model.parameters() if p.requires_grad]
    ddp = DDP(model, device_ids=[local], find_unused_parameters=False)
    vis = lambda n: n.startswith("connector.") or n.startswith("modality_type_embedding.")
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and not vis(n)], "lr": cfg["lr"]},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("connector.")],
               "lr": cfg["lr_connector"]},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("modality_type_embedding.")],
               "lr": cfg["lr_modality"]}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0)
    total = len(smp) * cfg["epochs"]
    cyc = len(smp) * cfg["restart_every"]                   # SGDR cycle; AdamW state kept across restarts
    warm = int(cfg["warmup_frac"] * cyc)
    lam = lambda s: (s % cyc / max(1, warm) if s % cyc < warm else
                     0.5 * (1 + math.cos(math.pi * (s % cyc - warm) / max(1, cyc - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam] * len(opt.param_groups))
    mb = cfg["micro_batch"]
    say(f"[train] global batch {cfg['batch']} ({cfg['batch'] // world} per rank, micro {mb}), {len(smp)} steps/epoch, "
        f"{total} steps; init_vision {cfg['init_vision']}; train_connector {cfg['train_connector']} "
        f"(lr {cfg['lr_connector']}, modality {cfg['lr_modality']})")

    def evaluate(tag, generate=True):
        model.eval(); rows, s, n = [], 0.0, 0.0
        for b in vdl:
            b = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, y = model.caption_nll(b, b["prior"])
                s, n = s + x, n + y
                gen = model.generate(b, b["prior"], cfg["max_new_tokens"]) if generate else [""] * b["n_items"]
            st = model.last_status if (generate and model.status_first) else [None] * len(gen)
            rows += [{"pair_id": p, "patient": pt, "generated": g} | ({"status_line": x} if x is not None else {})
                     for p, pt, g, x in zip(b["pair_id"], b["patient"], gen, st)]
        t = torch.tensor([s, n], device=dev); dist.all_reduce(t)
        allr = [None] * world
        dist.all_gather_object(allr, rows)
        model.train()
        ce = float(t[0] / t[1])
        say(f"[eval {tag}] val caption CE {ce:.4f} (health check only)")
        if rank or not generate:
            return ce
        meta = va.pairs.set_index("pair_id")
        path = f"{save}/gen_{tag}.jsonl"
        with open(path, "w") as f:
            for r in (r for x in allr for r in x):
                m = meta.loc[r["pair_id"]]
                f.write(json.dumps(r | {"source": m.source, "target": m.target, "prior": m.prior_edit,
                                        "current_orig": m.current_orig,
                                        "statuses": [bool(m[f"st_{s_}"]) for s_ in ("NEW", "WORSENED", "IMPROVED", "RESOLVED")]})
                        + "\n")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", str(local)).split(",")[local]
        subprocess.Popen([os.environ.get("BRAINDIFF_SCORE_PYTHON", sys.executable), "-m", "braindiff.eval.score", path],
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                         stdout=open(f"{save}/score_{tag}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        say(f"[eval {tag}] scoring in the background -> {save}/score_{tag}.npz")
        return ce

    hist = {"step0_ce": evaluate("step0", generate=False)}
    step, t0, run = 0, time.time(), []
    model.train()
    for ep in range(cfg["epochs"]):
        smp.set_epoch(ep)
        if ep and ep % cfg["restart_every"] == 0:
            say(f"[restart] epoch {ep + 1}: lr re-warms over {warm} steps (AdamW state kept, as in SGDR)")
        tw, wait = time.time(), 0.0
        for b in dl:
            wait += time.time() - tw
            b = {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            n = b["n_items"]
            parts = [list(range(lo, min(lo + mb, n))) for lo in range(0, n, mb)]
            for j, ix in enumerate(parts):
                s_ = sub_batch(b, ix)
                ctx = ddp.no_sync() if j < len(parts) - 1 else _null()
                with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, st = ddp(s_, s_["prior"])
                    (loss * len(ix) / n).backward()
                run.append(st["ce"])
            gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % cfg["log_every"] == 0:
                say(f"[train] step {step}/{total} ep {ep} CE {np.mean(run[-cfg['log_every'] * len(parts):]):.4f} "
                    f"gnorm {float(gn):.3f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f} s/step "
                    f"(loader wait {wait / max(step - ep * len(smp), 1):.2f} s/step) "
                    f"mem {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GB")
            tw = time.time()
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        if rank == 0:
            sd = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save({"model": sd, "step": step, "epoch": ep + 1, "cfg": cfg}, f"{save}/epoch{ep + 1}.pt")
            say(f"[save] {save}/epoch{ep + 1}.pt: {len(sd)} tensors")
        hist[f"epoch{ep + 1}_ce"] = evaluate(f"epoch{ep + 1}")
        say(f"[epoch {ep + 1}] train CE {np.mean(run[-len(smp) * len(parts):]):.4f}  {(time.time() - t0) / 60:.1f} min")
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break
    if rank == 0:
        json.dump(hist, open(f"{save}/eval_history.json", "w"), indent=1)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
