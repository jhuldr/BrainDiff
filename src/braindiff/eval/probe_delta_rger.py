"""Does ablating the delta change REPORT QUALITY? Label-free: scores generated
reports against the REFERENCE reports (the generated_report target), not the noisy
7-way labels.

Generates reports under `real` vs `delta_roll` (wrong pair's delta) and scores each
with:
  rg_er     RadGraph-XL entity+relation F1 (the paper metric; 0.44 at S2)
  triple_f1 temporal_score change-direction F1 (Pearson 0.99 vs RadGraph)
  bleu4 / meteor

Read: the delta HELPS report quality if ablating it (delta_roll) LOWERS rg_er vs real.
Reuses probe_delta_reliance's build/ablation and generate_caption_batch (so the
ablation applies during generation). Greedy, repetition_penalty 1.0.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from braindiff.eval.probe_delta_reliance import build_model, patch_ablation
from braindiff.training.train_dual_priorreport import make_loaders, MAX_PROMPT_LENGTH, MAX_REPORT_TOKENS
from braindiff.models.prior_report_prompts import PriorReportPrompts
from braindiff.eval import nlg_score
from braindiff.eval.temporal_score import score as temporal_score
# NB: radgraph is NOT imported here -- it pins transformers<5 and cannot share this
# interpreter. rg_er is scored separately in the radgraph env from the dumped JSON.


def _refs(batch, loader):
    """Reference reports for this batch: prefer batch['caption'], else dataset.items."""
    if "caption" in batch:
        return list(batch["caption"])
    ds = loader.dataset
    return [ds.items[i]["caption"] for i in batch["sample_idx"].tolist()]


@torch.no_grad()
def generate(model, loader, prompts, device, mode, n_batches, max_new_tokens,
             shard_index=0, num_shards=1, no_report=False):
    model._probe_mode = mode
    hyps, refs = [], []
    for step, batch in enumerate(loader):
        if step >= n_batches:
            break
        if step % num_shards != shard_index:   # this GPU's slice of the fixed 60 batches
            continue
        # Framing B: withhold the prior report from the decoder prompt so the model must
        # read interval change from the images. Confound-free (same checkpoint) vs the
        # report-present condition -- isolates the delta's value when the text is absent.
        reports = [""] * len(batch["prior_report"]) if no_report else list(batch["prior_report"])
        prompts.set_reports(reports)
        reports = model.generate_caption_batch(
            tokens_main=batch["tokens_main"].to(device),
            coords_main=batch["coords_main"].to(device),
            present_main=batch["present_main"].to(device),
            tokens_ref=batch["tokens_ref"].to(device),
            coords_ref=batch["coords_ref"].to(device),
            present_ref=batch["present_ref"].to(device),
            prompt_table=prompts, max_new_tokens=max_new_tokens, repetition_penalty=1.0)
        hyps.extend(reports)
        refs.extend(_refs(batch, loader))
    return hyps, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/nv_stage4_priorreport_15.pt")
    ap.add_argument("--base", default="nv_stage2_reportgen.pt")
    ap.add_argument("--diff-checkpoint", default="nv_stage3_deltaunsup_extended.pt")
    ap.add_argument("--csv", default="/home/data/BRAIN_DIFF_S4/splits_extended")
    ap.add_argument("--image-csv", default="/home/data/BRAIN_DIFF_S4/image_extended.csv")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="val is what checkpoint selection used; test is the held-out "
                         "split and the one to report")
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-caption-length", type=int, default=480)
    ap.add_argument("--modes", default="real,delta_roll")
    ap.add_argument("--out-dir", default=_os.path.join(_REPO, "logs", "rger_dump"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--include-delta", type=int, default=1,
                    help="0 for the nodelta ablation (model built with no delta path)")
    ap.add_argument("--no-report", action="store_true",
                    help="Framing B: withhold the prior report from the decoder prompt "
                         "(same checkpoint, confound-free). Dumps go to rger_<mode>.noreport.json")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    os.makedirs(args.out_dir, exist_ok=True)

    _, val_loader, test_loader = make_loaders(args.csv, args.image_csv, args.batch_size, 4,
                                              args.max_caption_length, seed=10, distributed=False)
    loader = test_loader if args.split == "test" else val_loader
    if loader is None:
        raise SystemExit(f"no {args.split} split under {args.csv}")
    model = build_model(args.base, args.ckpt, args.diff_checkpoint, device, args)
    patch_ablation(model)
    prompts = PriorReportPrompts(model.decoder.tokenizer, include_delta=bool(args.include_delta),
                                 max_prompt_length=MAX_PROMPT_LENGTH,
                                 max_report_tokens=MAX_REPORT_TOKENS)

    sharded = args.num_shards > 1
    suffix = f".rank{args.shard_index}" if sharded else ""
    tag = f"[rank {args.shard_index}/{args.num_shards}] " if sharded else ""
    print(f"\nprobe_delta_rger  {tag}ckpt={args.ckpt}  split={args.split}  "
          f"n_batches={args.n_batches}\n", flush=True)
    rows = {}
    rtag = ".noreport" if args.no_report else ""
    for mode in args.modes.split(","):
        hyps, refs = generate(model, loader, prompts, device, mode,
                              args.n_batches, args.max_caption_length,
                              args.shard_index, args.num_shards, no_report=args.no_report)
        path = os.path.join(args.out_dir, f"rger_{mode}{rtag}{suffix}.json")
        with open(path, "w") as f:
            json.dump({"hyps": hyps, "refs": refs}, f)
        if sharded:
            # in-env metrics are computed once over the merged dumps (merge_rger_shards.py)
            print(f"  {tag}{mode:11s}  n={len(hyps):4d}  -> {path}", flush=True)
            continue
        # metrics computable IN THIS env (no radgraph):
        nlg = nlg_score.score(hyps, refs)
        ts = temporal_score(hyps, refs)
        rows[mode] = dict(triple_f1=ts["triple_f1"], bleu4=nlg["bleu4"],
                          meteor=nlg["meteor"], n=nlg["n"], path=path)
        m = rows[mode]
        print(f"  {mode:11s}  n={m['n']:4d}  triple_f1={m['triple_f1']:.4f}  "
              f"bleu4={m['bleu4']:.4f}  meteor={m['meteor']:.4f}   -> {path}", flush=True)

    if sharded:
        print(f"\n  [rank {args.shard_index}/{args.num_shards}] done; merge with "
              f"merge_rger_shards.py after all ranks finish.", flush=True)
        return

    if "real" in rows and len(rows) > 1:
        print("\n  --- delta effect on the IN-ENV metrics (real - ablated); + => delta HELPS ---")
        for mode in rows:
            if mode == "real":
                continue
            print(f"  real - {mode:9s}  d_triple_f1={rows['real']['triple_f1']-rows[mode]['triple_f1']:+.4f}  "
                  f"d_meteor={rows['real']['meteor']-rows[mode]['meteor']:+.4f}")

    print("\n  Now score rg_er in the RADGRAPH env (per mode):")
    for mode in rows:
        print(f"    python trainer/radgraph_score.py < {rows[mode]['path']}   # -> RG_RESULT for {mode}")
    print(flush=True)


if __name__ == "__main__":
    main()
