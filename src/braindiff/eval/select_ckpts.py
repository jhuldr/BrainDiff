"""Checkpoint selection sweep: generate `real` reports on a fixed val SUBSET for
every checkpoint of one run, hot-swapping the S4 weights over a single loaded base
(one 14B load per run, not one per checkpoint). Dumps per-checkpoint hyp/ref shards
that merge_select.py concatenates and scores. Selection only -- `real` mode, no
visual ablation.

All checkpoints see the IDENTICAL subset: the val loader is fixed-seed and we take
the first --n-batches batches, sharded by (step %% num_shards == shard_index).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

from braindiff.eval.probe_delta_reliance import build_model, patch_ablation
from braindiff.eval.probe_delta_rger import generate
from braindiff.training.train_dual_priorreport import make_loaders, MAX_PROMPT_LENGTH, MAX_REPORT_TOKENS
from braindiff.models.prior_report_prompts import PriorReportPrompts
from braindiff.training.checkpoint import load_stage_checkpoint


def tag_of(ckpt):
    return os.path.basename(ckpt).replace(".pt", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths (one run)")
    ap.add_argument("--base", default="nv_stage2_reportgen.pt")
    ap.add_argument("--diff-checkpoint", default="")
    ap.add_argument("--include-delta", type=int, default=1)
    ap.add_argument("--csv", default="/home/data/BRAIN_DIFF_S4/splits_extended")
    ap.add_argument("--image-csv", default="/home/data/BRAIN_DIFF_S4/image_extended.csv")
    ap.add_argument("--n-batches", type=int, default=43)   # ~256 pairs at bs 6
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-caption-length", type=int, default=480)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    os.makedirs(args.out_dir, exist_ok=True)
    ckpts = [c for c in args.ckpts.split(",") if c]

    _, val_loader, _ = make_loaders(args.csv, args.image_csv, args.batch_size, 4,
                                    args.max_caption_length, seed=10, distributed=False)
    # build once on the first checkpoint, then hot-swap the rest
    diff = args.diff_checkpoint or None
    model = build_model(args.base, ckpts[0], diff, device, args)
    patch_ablation(model)
    prompts = PriorReportPrompts(model.decoder.tokenizer, include_delta=bool(args.include_delta),
                                 max_prompt_length=MAX_PROMPT_LENGTH,
                                 max_report_tokens=MAX_REPORT_TOKENS)

    tag = f"[rank {args.shard_index}/{args.num_shards}]"
    for i, ck in enumerate(ckpts):
        if i > 0:  # first is already loaded by build_model
            load_stage_checkpoint(model, torch.load(ck, map_location=device),
                                  label=ck, is_main=True)
            model.eval()
        hyps, refs = generate(model, val_loader, prompts, device, "real",
                              args.n_batches, args.max_caption_length,
                              args.shard_index, args.num_shards)
        path = os.path.join(args.out_dir, f"sel_{tag_of(ck)}.rank{args.shard_index}.json")
        with open(path, "w") as f:
            json.dump({"hyps": hyps, "refs": refs}, f)
        print(f"{tag} {tag_of(ck):40s} n={len(hyps):4d} -> {path}", flush=True)

    print(f"{tag} done {len(ckpts)} checkpoints", flush=True)


if __name__ == "__main__":
    main()
