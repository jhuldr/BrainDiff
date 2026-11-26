#!/usr/bin/env python3
"""Status-first fine-tuned Qwen3-VL-8B, the Table 2 "Qwen3-VL-8B FT, status-first" row:
a status warm-up, then status-first fine-tuning.
  train     LoRA fine-tune on the Table 2 8-slice input (generate.py qwen3vl8b_8). Same training pairs and recipe as our Stage 2
            decoder: LoRA r16 / alpha 32 / dropout 0.05 on the LLM's q/k/v/o (vision tower and merger frozen), AdamW lr 1e-4,
            3% warm-up + cosine, global batch 32, grad clip 1.0, bf16, seed 0, loss on target tokens only.
            --target status = status line alone (warm-up), sf = "<status line>\n<report>"; --init continues
            a warm-up adapter with a fresh optimiser. Adapter saved every epoch (<out>/epoch<k>/), optimiser in <out>/state_last.pt.
  assemble  <k>: val generations of epoch k (status line cut by sf_strip) -> our generation-file format on the 1,000 val pairs
  pick      pre-specified epoch pick: best val macro-MCC, ties -> earlier epoch -> $BRAINDIFF_CKPT/$FT_OUT/select.json
    torchrun --nproc_per_node 4 finetune.py train --out checkpoints/qwen3vl8b_sfw --epochs 2 --target status --accum 8      (BrainDiff env)
    torchrun --nproc_per_node 4 finetune.py train --out checkpoints/qwen3vl8b_sf --epochs 3 --target sf --init checkpoints/qwen3vl8b_sfw/epoch2 --accum 8
    python finetune.py assemble 1 ; ... ; python finetune.py pick                                                              (scoring env)
Paths: BRAINDIFF_PAIRS, BASELINE_DIR (generate.py), BRAINDIFF_CKPT (checkpoints: stage2_A0_s1d_sf/gen_val_epoch20.jsonl), VAL_IDS
(checkpoints/decoding_check/val1000_ids.txt). The prompt strings come from generate.py."""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import INSTR_IMG, OUT as B, PAIRS, SYSTEM, matched  # noqa: E402
import generate  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"; CK = os.environ.get("BRAINDIFF_CKPT", "checkpoints")
VAL_IDS = os.environ.get("VAL_IDS", f"{CK}/decoding_check/val1000_ids.txt")
STATUSES, CLOSE, NONE_TEXT = ("NEW", "WORSENED", "IMPROVED", "RESOLVED"), "No other interval change.", "No interval change."


def status_line(r):
    """Stage-1 status format from a pair's st_* labels."""
    on = [s for s in STATUSES if r[f"st_{s}"]]
    return " ".join(f"{s}." for s in on) + f" {CLOSE}" if on else NONE_TEXT


def messages(r):
    """build_sl8 order and labels (current study first), cap 96 images, then prior report + INSTR_IMG."""
    from PIL import Image
    content = []
    for t, tag in ((2, "Current"), (1, "Prior")):
        for m in matched(r):
            h = hashlib.md5(r[f"{m}_{t}"].encode()).hexdigest(); n = int(open(f"{generate.MON}/{h}_sl8_n.txt").read())
            for k in range(n):
                content += [{"type": "text", "text": f"{tag} study - {m}, slice {k + 1}/{n}"},
                            {"type": "image", "image": Image.open(f"{generate.MON}/{h}_sl8_{k}.png").convert("RGB")}]
    content = content[:192] + [{"type": "text", "text": f"Prior report:\n{(r['prior_edit'] or 'Not available.').strip()}\n\n{INSTR_IMG}"}]
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]


def train(argv):
    import pandas as pd
    import torch
    import torch.distributed as dist

    class Pairs(torch.utils.data.Dataset):
        def __init__(self, rows):
            from transformers import AutoProcessor
            self.rows, self.P = rows, AutoProcessor.from_pretrained(MODEL)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows.iloc[i]; msgs = messages(r)
            n_prompt = self.P.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")["input_ids"].shape[1]
            x = self.P.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": r.caption}]}], tokenize=True,
                                           return_dict=True, return_tensors="pt")
            x["labels"] = x["input_ids"].clone(); x["labels"][:, :n_prompt] = -100
            return dict(x)

    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--accum", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--max_target", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N training pairs"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--target", default="sf", choices=["status", "sf"]); ap.add_argument("--init", help="warm-up adapter dir (--target sf)")
    a = ap.parse_args(argv)
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr_); dev = f"cuda:{lr_}"; torch.manual_seed(0); main_ = rank == 0
    say = (lambda *s: print(f"[{time.strftime('%a %H:%M')}]", *s, flush=True)) if main_ else (lambda *s: None)

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    p = pd.read_parquet(PAIRS); p = p[p.split == "train"].sort_values("pair_id").reset_index(drop=True)
    have = lambda v: os.path.exists(f"{generate.MON}/{hashlib.md5(v.encode()).hexdigest()}_sl8_n.txt")
    ok = [len(matched(r)) > 0 and all(have(r[f"{m}_{t}"]) for m in matched(r) for t in (1, 2)) for _, r in p.iterrows()]
    say(f"train pairs without matched series or with a failed slice render (dropped): {len(p) - sum(ok)}"); p = p[ok]
    tok = AutoTokenizer.from_pretrained(MODEL); n = np.array([len(x) for x in tok(p.target.tolist(), add_special_tokens=False).input_ids])
    say(f"train pairs {len(p):,} ({p.source.value_counts().to_dict()}); targets >= {a.max_target} tokens dropped: {(n >= a.max_target).sum()}")
    p = p[n < a.max_target].reset_index(drop=True)
    if a.limit:
        p = p.iloc[:a.limit]
    p["caption"] = {"status": p.apply(status_line, axis=1), "sf": p.apply(status_line, axis=1) + "\n" + p.target}[a.target]
    say(f"target {a.target}" + (f", init {a.init}" if a.init else ""))
    ds = Pairs(p)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, world, rank, shuffle=True, seed=0, drop_last=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, num_workers=8, prefetch_factor=4, persistent_workers=True)
    steps_ep = len(sampler) // a.accum; total = steps_ep * a.epochs; warm = int(0.03 * total)
    say(f"{len(p):,} pairs, {world} GPUs x accum {a.accum} = global batch {world * a.accum}; {steps_ep} optimiser steps / epoch, {total} total, warm-up {warm}")

    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    for x in model.parameters():
        x.requires_grad_(False)
    start = 0
    if a.resume and os.path.exists(f"{a.out}/state_last.pt"):
        st = torch.load(f"{a.out}/state_last.pt", map_location="cpu", weights_only=False); start = st["epoch"]
        model = PeftModel.from_pretrained(model, f"{a.out}/epoch{start}", is_trainable=True)
    elif a.init:
        model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads(); model.train()
    params = [x for x in model.parameters() if x.requires_grad]
    say(f"trainable {sum(x.numel() for x in params):,} (LoRA q/k/v/o of the LLM only: {all('language_model' in k for k, x in model.named_parameters() if x.requires_grad)})")
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_])
    opt = torch.optim.AdamW(params, lr=a.lr)
    lam = lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    if start:
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); say(f"resumed after epoch {start}")

    for ep in range(start, a.epochs):
        sampler.set_epoch(ep); t0 = time.time(); acc, run, k = 0, [], 0
        for i, b in enumerate(dl):
            if i >= steps_ep * a.accum:
                break
            b = {kk: v.to(dev, non_blocking=True) for kk, v in b.items()}
            sync = (i + 1) % a.accum == 0
            with (ddp.no_sync() if not sync else torch.enable_grad()):
                loss = ddp(**b).loss / a.accum; loss.backward()
            acc += float(loss.detach())
            if sync:
                torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); k += 1
                run.append(acc); acc = 0
                if k % 20 == 0 or k == steps_ep:
                    l = torch.tensor(np.mean(run[-20:]), device=dev); dist.all_reduce(l); el = time.time() - t0
                    say(f"epoch {ep + 1} step {k}/{steps_ep} loss {float(l) / world:.4f} lr {sched.get_last_lr()[0]:.2e} "
                        f"{el / k:.2f} s/step, epoch ETA {(steps_ep - k) * el / k / 60:.0f} min")
        if main_:
            model.save_pretrained(f"{a.out}/epoch{ep + 1}")
            torch.save({"epoch": ep + 1, "opt": opt.state_dict(), "sched": sched.state_dict()}, f"{a.out}/state_last.pt")
            say(f"EPOCH_DONE {ep + 1}: mean loss {np.mean(run):.4f}, {(time.time() - t0) / 60:.1f} min, saved {a.out}/epoch{ep + 1}")
        dist.barrier()
    say("TRAIN_DONE"); dist.destroy_process_group()


TAG, FT_OUT = os.environ.get("FT_TAG", "sf"), os.environ.get("FT_OUT", "qwen3vl8b_sf")
d = lambda k: f"{B}/qwen3vl8b_ft-{TAG}{k}_imginstr_sl8img_rp110" if k else f"{B}/qwen3vl8b_imginstr_sl8img_rp110"   # 0 = untuned (reference only)


def assemble(k):
    ids = {int(x) for x in open(VAL_IDS).read().split()}
    g = {json.loads(l)["pair_id"]: json.loads(l)["generated"] for l in open(f"{d(k)}/gen_val_sub.jsonl")}
    rows = [r for r in map(json.loads, open(f"{CK}/stage2_A0_s1d_sf/gen_val_epoch20.jsonl")) if r["pair_id"] in ids]
    miss = 0
    with open(f"{d(k)}/gen_val1000_epoch20.jsonl", "w") as f:
        for r in rows:
            t = (g.get(r["pair_id"]) or "").strip(); miss += not t
            f.write(json.dumps(r | {"generated": t or ".", "status_line": ""}) + "\n")
    print(f"epoch {k}: {len(rows)} val pairs, {miss} missing / empty", flush=True)


def mcc(Y, H):
    from sklearn.metrics import matthews_corrcoef
    return float(np.mean([matthews_corrcoef(Y[:, j], H[:, j]) if 0 < H[:, j].sum() < len(H) else 0 for j in range(4)]))


def pick():
    res = {}
    for k in range(0, 6):
        f = f"{d(k)}/score_val1000_epoch20.npz"
        if os.path.exists(f):
            s = np.load(f, allow_pickle=True); res[k] = round(float(mcc(s["Y"].astype(bool), s["model"] >= 0.5)), 4)
    best = max((k for k in res if k), key=lambda k: (res[k], -k), default=None)
    json.dump({"val_macro_mcc": res, "chosen_epoch": best, "n_pairs": 1000}, open(f"{CK}/{FT_OUT}/select.json", "w"), indent=1)
    print("val macro-MCC by epoch", res, "-> CHOSEN epoch", best, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train": train(sys.argv[2:])
    elif cmd == "assemble": assemble(int(sys.argv[2]))
    else: pick()
