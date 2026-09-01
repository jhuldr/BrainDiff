"""Table 3 reversal probe REBUILD (Move 2): restrict to CHANGE cases where a forward
direction exists, over the full test subset, and report a SIGNED-FLIP rate with a CI.

Change cases: GT label in {New lesion(1), Progressed(3), Improved(4), Resolved(6)}.
Direction of a report (via change_class): +1 {Progressed,New lesion}, -1 {Improved,Resolved}, 0 else.
Denominator: change-case pairs where the FORWARD report asserts a direction (d_fwd != 0).
Signed flip = the REVERSED report's direction is opposite-or-neutral vs forward
  (sign(d_rev) != sign(d_fwd)) -- i.e. reversing the images flips/degrades the asserted change.
A grounded model flips (high rate); a prior-report reciter is indifferent (low rate).
Reports flip_rate + 95% bootstrap CI for one checkpoint. GPU required.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import os, sys, argparse, json
sys.path.insert(0, _REPO)
import numpy as np, torch
from braindiff.eval.probe_delta_reliance import build_model
from braindiff.training.train_dual_priorreport import make_loaders, MAX_PROMPT_LENGTH, MAX_REPORT_TOKENS
from braindiff.models.prior_report_prompts import PriorReportPrompts
from braindiff.eval.temporal_score import change_class
from braindiff.data.dual import CHANGE_CLASSES

CHANGE_IDX = {1, 3, 4, 6}                       # New lesion, Progressed, Improved, Resolved
POS = {"Progressed", "New lesion"}; NEG = {"Improved", "Resolved"}
def direction(rep):
    c = change_class(rep); return 1 if c in POS else (-1 if c in NEG else 0)

@torch.no_grad()
def gen(model, batch, prompts, device, mn, swap):
    prompts.set_reports(list(batch["prior_report"]))
    tm,cm,pm,tr,cr,pr = "tokens_main","coords_main","present_main","tokens_ref","coords_ref","present_ref"
    if swap: tm,tr=tr,tm; cm,cr=cr,cm; pm,pr=pr,pm
    return model.generate_caption_batch(
        tokens_main=batch[tm].to(device), coords_main=batch[cm].to(device), present_main=batch[pm].to(device),
        tokens_ref=batch[tr].to(device), coords_ref=batch[cr].to(device), present_ref=batch[pr].to(device),
        prompt_table=prompts, max_new_tokens=mn, repetition_penalty=1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--include-delta", type=int, default=0)
    ap.add_argument("--diff-checkpoint", default=""); ap.add_argument("--base", default="nv_stage2_reportgen.pt")
    ap.add_argument("--csv", default="/home/data/BRAIN_DIFF_S4/splits_extended")
    ap.add_argument("--image-csv", default="/home/data/BRAIN_DIFF_S4/image_extended.csv")
    ap.add_argument("--n-batches", type=int, default=250); ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-caption-length", type=int, default=480); ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reports-out", default="",
                    help="dump {gt,fwd,rev} for every GT change-case, for offline re-scoring "
                         "with a different direction classifier (e.g. TF-IDF)")
    a = ap.parse_args(); dev = torch.device(f"cuda:{a.gpu}"); torch.cuda.set_device(dev)
    _, val, test = make_loaders(a.csv, a.image_csv, a.batch_size, 4, a.max_caption_length, seed=10, distributed=False)
    test = val if a.split == "val" else test
    model = build_model(a.base, a.ckpt, a.diff_checkpoint or None, dev, a)
    prompts = PriorReportPrompts(model.decoder.tokenizer, include_delta=bool(a.include_delta),
                                 max_prompt_length=MAX_PROMPT_LENGTH, max_report_tokens=MAX_REPORT_TOKENS)
    flips = []                                   # per change-case-with-fwd-direction: 1 if flipped/degraded
    n_change = n_fwddir = 0
    change_reports = []                          # every GT change-case: {gt,fwd,rev}
    for step, batch in enumerate(test):
        if step >= a.n_batches: break
        if step % a.num_shards != a.shard_index: continue   # this GPU's slice of batches
        gts = batch["change_label"].tolist()
        fwd = gen(model, batch, prompts, dev, a.max_caption_length, False)
        rev = gen(model, batch, prompts, dev, a.max_caption_length, True)
        for gt, f, r in zip(gts, fwd, rev):
            if gt not in CHANGE_IDX: continue
            n_change += 1
            change_reports.append({"gt": int(gt), "fwd": f, "rev": r})
            df = direction(f)
            if df == 0: continue                 # forward asserted no direction -> uninformative
            n_fwddir += 1
            dr = direction(r)
            flips.append(1 if np.sign(dr) != np.sign(df) else 0)
    flips = np.array(flips); n = len(flips); rng = np.random.default_rng(0)
    rate = float(flips.mean()) if n else float("nan")
    if n:
        s = np.array([flips[rng.choice(n,n,replace=True)].mean() for _ in range(10000)]); lo,hi = np.percentile(s,[2.5,97.5])
    else: lo=hi=float("nan")
    res = {"ckpt": a.ckpt, "split": a.split, "n_change_cases": n_change, "n_forward_direction": n_fwddir,
           "signed_flip_rate": rate, "ci": [float(lo), float(hi)], "flips": flips.tolist()}
    json.dump(res, open(a.out, "w")); print(json.dumps(res))
    if a.reports_out:
        json.dump(change_reports, open(a.reports_out, "w"))
        print(f"dumped {len(change_reports)} GT change-case reports -> {a.reports_out}")
    print("TABLE3_V2_DONE")

if __name__ == "__main__":
    main()
