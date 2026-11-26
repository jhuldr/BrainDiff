#!/usr/bin/env python3
"""
MAIRA-2 status-first fine-tune (recipe 4, "no-vision"), seed 42 -- the fine-tuned MAIRA-2 of the chest X-ray block.
Plain PyTorch DDP around the released model + processor (Microsoft released no training code).

data    MIMIC-CXR train pairs (libra_sf/train_L1.json) with MAIRA-2 inputs built as in cxr_generate.py; targets = current
        FINDINGS + Rad-Delta status line (libra_sf/labels_train.parquet). -> out/cxr_arm/maira_sf/train.parquet
train   RAD-DINO frozen; projector fully trained (fp32); LLM LoRA on q/k/v/o. Loss on target + </s> only.
        Arms: M3W warm-up (status line only, Stage 1 status question); M3 = status line + FINDINGS from M3W (--init);
        M1 = FINDINGS from the released model. Saves <out>/epoch<k>/ (peft adapter) + projector.pt.
select  per-epoch val generation scored by macro-MCC of Rad-Delta status; prints BEST <arm> <epoch>.
        Paper run: M3 epoch 2, M1 epoch 2.

    python baselines/cxr/maira2_finetune.py data --reports <mimic-cxr-reports.zip>
    torchrun --nproc_per_node 3 baselines/cxr/maira2_finetune.py train --arm M3W --epochs 2 --micro 5 --batch 30 --seed 42 --out <d>/M3W_lora
    torchrun --nproc_per_node 3 baselines/cxr/maira2_finetune.py train --arm M3 --epochs 3 --micro 5 --batch 30 --seed 42 --out <d>/M3_lora --init <d>/M3W_lora/epoch2
    torchrun --nproc_per_node 6 baselines/cxr/maira2_finetune.py train --arm M1 --epochs 3 --micro 5 --batch 30 --seed 42 --out <d>/M1_lora
    CUDA_VISIBLE_DEVICES=0 python baselines/cxr/maira2_finetune.py select --arm M3
"""
import argparse
import glob
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(os.environ.get("CXR_WORK", "."))
CXR = WORK / "out" / "cxr_arm"
O = CXR / "maira_sf"
IMG = Path(os.environ.get("MIMIC_CXR_JPG", "data/mimic-cxr-jpg/2.1.0"))
# M3W: BrainDiff's Stage 1 status question replaces MAIRA-2's task sentence
STATUS_Q = ("Which interval changes are present between the prior and current studies? Answer with the statuses "
            "present (NEW, WORSENED, IMPROVED, RESOLVED), then end with 'No other interval change.', e.g. 'NEW. "
            "IMPROVED. No other interval change.' If nothing changed, answer 'No interval change.'")
TASK = " Provide a description of the findings in the radiology study in comparison to the prior frontal image."
LINE = re.compile(r"^\s*((?:(?:NEW|WORSENED|IMPROVED|RESOLVED)\.\s*)+No other interval change\.|No interval change\.)\s*\n?", re.S)


# ---- data ----
def data(a):
    import zipfile
    from cxr_generate import sections
    T = pd.read_parquet(CXR / "libra_sf_train.parquet")
    keep = {tuple(r["image"]) for r in json.load(open(CXR / "libra_sf" / "train_L1.json"))}          # (current, prior), both images
    T = T[[(c, p) in keep for c, p in zip(T.current_frontal, T.prior_frontal)]].reset_index(drop=True)
    L = pd.read_parquet(CXR / "libra_sf" / "labels_train.parquet").set_index("patient")
    zf = zipfile.ZipFile(a.reports); names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".txt")}
    rep = lambda sid: sections(zf.read(names[sid]).decode("utf-8", "replace")) if sid in names else {}  # noqa: E731
    rows = []
    for r in T.itertuples():
        c = rep("s" + re.search(r"_s(\d+)$", r.patient).group(1)); p = rep(re.search(r"/(s\d+)/", r.prior_frontal).group(1))
        rows.append({"patient": r.patient, "current_frontal": r.current_frontal, "prior_frontal": r.prior_frontal,
                     "prior_report": " ".join(x for x in (f"FINDINGS: {p['FINDINGS']}" if p.get("FINDINGS") else "",
                                                          f"IMPRESSION: {p['IMPRESSION']}" if p.get("IMPRESSION") else "") if x),
                     "indication": c.get("INDICATION") or c.get("HISTORY") or "", "technique": c.get("TECHNIQUE", ""),
                     "comparison": c.get("COMPARISON", ""), "findings": r.findings, "status_line": L.loc[r.patient, "status_line"]})
    M = pd.DataFrame(rows)
    assert (M.findings == T.findings).all() and M.findings.str.len().gt(0).all()
    O.mkdir(parents=True, exist_ok=True)
    M.to_parquet(a.out or O / "train.parquet")
    print(len(M), "pairs;", {c: f"{(M[c].str.len() > 0).mean():.0%}" for c in ("prior_report", "indication", "technique", "comparison")})


# ---- train ----
def img(rel):   # .jpg listed; a PNG download keeps the same paths otherwise
    from PIL import Image
    q = IMG / rel
    return Image.open(q if q.exists() else q.with_suffix(".png")).convert("RGB")


try:
    import torch
    Dataset = torch.utils.data.Dataset
except ImportError:      # data / select need no torch at import time
    Dataset = object


class Pairs(Dataset):
    def __init__(self, df, arm, proc):
        self.df, self.arm, self.proc = df.reset_index(drop=True), arm, proc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        tgt = r.findings if self.arm == "M1" else r.status_line if self.arm == "M3W" else r.status_line + "\n" + r.findings
        kw = dict(current_frontal=img(r.current_frontal), current_lateral=None, prior_frontal=img(r.prior_frontal),
                  indication=r.indication, technique=r.technique, comparison=r.comparison, prior_report=r.prior_report, return_tensors="pt")
        # = inference: generation prompt + target as the tokenizer splits it + </s>; the chat template would merge the
        # prompt's trailing "▁" into the first target token
        x = self.status_prompt(kw) if self.arm == "M3W" else self.proc.format_and_preprocess_reporting_input(assistant_text=None, **kw)
        t = torch.tensor(self.proc.tokenizer(tgt, add_special_tokens=False).input_ids + [self.proc.tokenizer.eos_token_id])
        p = x.input_ids[0]; n = len(p)
        return torch.cat([p, t]), torch.cat([torch.full((n,), -100), t]), x.pixel_values, n

    def status_prompt(self, kw):
        """M3W: MAIRA-2's reporting messages with the task sentence swapped for STATUS_Q."""
        P = self.proc
        msgs = P._construct_chat_messages_reporting(has_prior=True, has_lateral=False, indication=kw["indication"], technique=kw["technique"],
                                                    comparison=kw["comparison"], prior_report=kw["prior_report"], get_grounding=False,
                                                    assistant_text=None)
        last = msgs[0]["content"][-1]; assert last["text"].startswith(TASK), last["text"][:80]
        last["text"] = " " + STATUS_Q + last["text"][len(TASK):]
        text = P.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        imgs = P._normalize_and_stack_images(current_frontal=kw["current_frontal"], current_lateral=None, prior_frontal=kw["prior_frontal"])
        return P(text=text, images=imgs, return_tensors="pt")


def collate(batch, pad):
    L = max(len(b[0]) for b in batch)
    ids = torch.full((len(batch), L), pad, dtype=torch.long); lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y, _, _) in enumerate(batch):
        ids[i, :len(x)] = x; lab[i, :len(y)] = y; att[i, :len(x)] = 1
    return ids, att, lab, torch.cat([b[2] for b in batch]), [b[3] for b in batch]


def projector(model):
    return next(m for n, m in model.named_modules() if n.endswith("multi_modal_projector"))


def train(a):
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoProcessor, get_cosine_schedule_with_warmup
    assert bool(a.init) == (a.arm == "M3"), "--init exactly for M3"
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local); dev = torch.device("cuda", local)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    acc = a.batch // (a.micro * world); assert acc * a.micro * world == a.batch, "batch must be micro x GPUs x accumulation"
    proc = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             bias="none", task_type="CAUSAL_LM"))
    proj = projector(model)
    if a.init:   # M3: continue the warm-up's LLM LoRA and projector; fresh optimizer
        sd = torch.load(Path(a.init) / "adapter_model.bin", map_location="cpu"); set_peft_model_state_dict(model, sd)
        k = next(k for k in sd if "lora_B" in k); got = dict(model.named_parameters())
        q = next(n for n in got if n.endswith(k.replace(".weight", ".default.weight")) or n == k)
        assert torch.equal(got[q].data.cpu(), sd[k].to(got[q].dtype)) and sd[k].abs().sum() > 0, "init LLM adapter not loaded"
        ps = torch.load(Path(a.init) / "projector.pt", map_location="cpu")
        before = proj.layers[0].weight.detach().float().clone(); proj.load_state_dict(ps)
        assert not torch.equal(before, proj.layers[0].weight.detach().float()), "init projector identical to the released one"
        if rank == 0:
            print(f"[init] {len(sd)} LLM LoRA tensors + projector ({len(ps)} tensors) from {a.init}", flush=True)
    for n, p in model.named_parameters():   # trainable: LLM LoRA + projector
        p.requires_grad = ("lora_" in n and ".language_model." in n) or ("multi_modal_projector" in n)
        if p.requires_grad:
            p.data = p.data.float()
    proj.register_forward_pre_hook(lambda m, args: tuple(x.float() for x in args))     # fp32 projector inside a bf16 model
    proj.register_forward_hook(lambda m, args, out: out.to(torch.bfloat16))
    model.to(dev)
    g_llm = [p for n, p in model.named_parameters() if p.requires_grad and ".language_model." in n]
    g_proj = [p for n, p in model.named_parameters() if p.requires_grad and "multi_modal_projector" in n]
    assert len(g_llm) + len(g_proj) == sum(p.requires_grad for p in model.parameters()) and g_llm and len(g_proj) == 8
    assert not any(p.requires_grad for n, p in model.named_parameters() if "vision_tower" in n)
    if rank == 0:
        print(f"[{a.arm}] world {world} micro {a.micro} acc {acc} (effective {a.batch}); trainable LLM LoRA "
              f"{sum(p.numel() for p in g_llm) / 1e6:.2f} M (lr {a.lr}), projector {sum(p.numel() for p in g_proj) / 1e6:.2f} M "
              f"(lr {a.lr_projector}), vision 0", flush=True)
    ddp = DDP(model, device_ids=[local])
    df = pd.read_parquet(O / "train.parquet")
    if a.smoke:
        df = df.iloc[:a.smoke * a.batch]
    ds = Pairs(df, a.arm, proc)
    samp = torch.utils.data.distributed.DistributedSampler(ds, world, rank, shuffle=True, seed=a.seed, drop_last=False)
    dl = torch.utils.data.DataLoader(ds, a.micro, sampler=samp, num_workers=6, persistent_workers=True,
                                     collate_fn=lambda b: collate(b, proc.tokenizer.pad_token_id))
    steps_ep = math.ceil(len(dl) / acc); total = steps_ep * a.epochs
    opt = torch.optim.AdamW([{"params": g_llm, "lr": a.lr}, {"params": g_proj, "lr": a.lr_projector}], weight_decay=0.0)
    sch = get_cosine_schedule_with_warmup(opt, math.ceil(0.03 * total), total)
    params = g_llm + g_proj; step, t0 = 0, time.time()
    for ep in range(a.epochs):
        samp.set_epoch(ep); ddp.train(); run = []
        for i, (ids, att, lab, pix, npr) in enumerate(dl):
            last = (i + 1) % acc == 0 or i + 1 == len(dl)
            with (ddp.no_sync() if not last else torch.enable_grad()):
                out = ddp(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev), pixel_values=pix.to(dev, torch.bfloat16))
                (out.loss / acc).backward()
            run.append(out.loss.item())
            if last:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
                if rank == 0 and (step % 5 == 0 or a.smoke):
                    el = time.time() - t0
                    print(f"[{a.arm}] epoch {ep + 1} step {step}/{total} loss {np.mean(run[-acc:]):.4f} lr {sch.get_last_lr()[0]:.2e} "
                          f"{el / step:.1f} s/step, max mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
                if a.smoke and step >= a.smoke:
                    break
        if a.smoke:
            break
        if rank == 0:
            d = Path(a.out) / f"epoch{ep + 1}"; model.save_pretrained(d)
            torch.save({k: v.detach().float().cpu() for k, v in proj.state_dict().items()}, d / "projector.pt")
            print(f"[{a.arm}] epoch {ep + 1} done: mean loss {np.mean(run):.4f}; saved {d}", flush=True)
        dist.barrier()
    dist.destroy_process_group()


# ---- epoch selection ----
def split_line(text):
    """-> (status line or None, report). Only a well-formed leading status line is cut."""
    m = LINE.match(text or "")
    return (m.group(1), text[m.end():].strip()) if m else (None, (text or "").strip())


def load_gen(pattern):
    g = {}
    for f in sorted(glob.glob(str(CXR / pattern))):
        for l in open(f):
            if l.strip():
                r = json.loads(l); g.setdefault(r["patient"], r["report"])
    return g


def select(a):
    from raddelta.metric import RadDelta, macro_mcc, presence
    from raddelta.model import DEFAULT_CKPT
    m = RadDelta(DEFAULT_CKPT, a.gpu)
    S = pd.read_parquet(CXR / "sample_mimic_valf.parquet"); ref = dict(zip(S.patient, S.ref))
    runs = sorted({re.sub(r"_shard\d+\.jsonl$", "", Path(f).name) for f in glob.glob(str(CXR / f"gen_maira2_sf{a.arm}_epoch*_shard*.jsonl"))})
    Y = {p: presence(m, t) >= 0.5 for p, t in ref.items()}; best = None
    for r in runs:
        g = load_gen(r + "_shard*.jsonl"); pats = sorted(p for p in ref if p in g)
        H = np.array([presence(m, split_line(g[p])[1]) >= 0.5 for p in pats]); v = macro_mcc(np.array([Y[p] for p in pats]), H)
        print(f"[select {a.arm}] {r[4:]}: {len(pats)} val pairs, macro-MCC {v:.4f}", flush=True)
        if len(pats) == len(ref) and (best is None or v > best[1]):
            best = (r.split("_")[-1], v)
    print("BEST", a.arm, best[0] if best else "none", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--reports", required=True); d.add_argument("--out", default=None)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=["M1", "M3W", "M3"], required=True)
    t.add_argument("--init", help="warm-up epoch dir (M3)"); t.add_argument("--out"); t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--micro", type=int, default=5); t.add_argument("--batch", type=int, default=30)
    t.add_argument("--lr", type=float, default=1e-4); t.add_argument("--lr_projector", type=float, default=2e-5)
    t.add_argument("--smoke", type=int, default=0); t.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("select")
    s.add_argument("--arm", required=True, help="M1 / M3 (prefix as in the generation tags, e.g. rM3 for the paper run)")
    s.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args(argv)
    {"data": data, "train": train, "select": select}[a.cmd](a)


if __name__ == "__main__":
    main()
