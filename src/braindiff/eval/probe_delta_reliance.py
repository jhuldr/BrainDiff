"""Does the S4 prior-report model actually use the diff (delta) signal?

S4's sole objective is LM CE with no localisation supervision, so the delta path could
erode -- the decoder could learn to ignore the 4 delta visual blocks entirely.

Method: mean val LM CE under content ablations of the assembled visual chunks, at fixed
prompt structure (block_present untouched, so the prompt is byte-identical and only block
content changes):

    real       chunks as produced
    delta_zero the 4 delta blocks zeroed -> removes the pair-specific diff signal
    delta_roll the 4 delta blocks rolled across the batch -> wrong pair's delta at the same
               norm, ruling out a "zeros are OOD" artifact
    scan_zero  the 8 scan blocks zeroed -> upper reference

The delta is used if delta_zero and delta_roll raise CE clearly above `real`. If
dCE(delta_zero) is within noise of 0 while dCE(scan_zero) is large, the delta has collapsed.

Runs single-GPU under no_grad (a grad-on forward OOMs; no_grad ~35 GiB).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from braindiff.models.captioner import DeltaDiffCaptioner_Qwen3
from braindiff.models.prior_report_prompts import PriorReportPrompts
from braindiff.data.dual import CHANGE_CLASSES
from braindiff.training.checkpoint import load_stage_checkpoint
from braindiff.training.train_dual_priorreport import make_loaders, MAX_PROMPT_LENGTH, MAX_REPORT_TOKENS


def build_model(base_ckpt, delta_state_ckpt, diff_checkpoint, device, args):
    include_delta = bool(getattr(args, "include_delta", 1))
    model = DeltaDiffCaptioner_Qwen3(
        single_timepoint=False, use_vision_lora=True,
        vision_lora_r=32, vision_lora_alpha=64, vision_lora_dropout=0.05,
        num_queries=64, include_delta=include_delta, lora_r=16, lora_alpha=32,
        lora_dropout=0.05, use_lora=True, pretrained_connector=False,
        max_caption_length=args.max_caption_length, num_change_classes=0,
        counterfactual_margin=0.5, device=device).to(device)
    # base S2: decoder base + connector.scan/proj + embeddings
    load_stage_checkpoint(model, torch.load(os.path.join("checkpoints", base_ckpt),
                          map_location=device), label=base_ckpt, is_main=True)
    # S3-sup delta path (in case the epoch-5 file's frozen connector.delta matches it anyway)
    if diff_checkpoint:
        load_stage_checkpoint(model, torch.load(os.path.join("checkpoints", diff_checkpoint),
                              map_location=device), label=diff_checkpoint, is_main=True,
                              strict_groups=("diff_encoder", "connector.delta"))
    # epoch-5 trained state ON TOP (diff_encoder, connector.delta, embeddings, decoder_lora)
    load_stage_checkpoint(model, torch.load(delta_state_ckpt, map_location=device),
                          label=delta_state_ckpt, is_main=True)
    model.eval()
    return model


def patch_ablation(model):
    """Wrap _assemble_visual so `model._probe_mode` ablates chunk CONTENT only.
    block_present is returned untouched, so the prompt structure is identical."""
    orig = model._assemble_visual
    m = model.num_modalities  # 4; block-major: ref[0:m], main[m:2m], delta[2m:3m]

    def wrapped(f_ref, f_main, present_ref, present_main):
        chunks, block_present = orig(f_ref, f_main, present_ref, present_main)
        mode = getattr(model, "_probe_mode", "real")
        d0, d1 = 2 * m, 3 * m           # delta columns
        if mode == "real":
            pass
        elif mode == "delta_zero":
            chunks[:, d0:d1] = 0
        elif mode == "delta_roll":
            if chunks.shape[0] > 1:
                chunks[:, d0:d1] = torch.roll(chunks[:, d0:d1], shifts=1, dims=0)
        elif mode == "delta_mean":
            # in-distribution removal of pair-specificity: every sample gets the
            # BATCH-MEAN delta block. Unlike delta_zero this is not OOD; if the
            # decoder used pair-specific delta content, this must raise CE like roll.
            if chunks.shape[0] > 1:
                mean_d = chunks[:, d0:d1].mean(dim=0, keepdim=True)
                # cross-sample std of the delta block, as a fraction of its mean abs:
                std = chunks[:, d0:d1].std(dim=0).mean().item()
                mag = chunks[:, d0:d1].abs().mean().item()
                model._probe_delta_cv = std / (mag + 1e-8)
                chunks[:, d0:d1] = mean_d.expand_as(chunks[:, d0:d1])
        elif mode == "scan_zero":
            chunks[:, 0:d0] = 0
        elif mode == "scan_mean":
            # in-distribution "no useful image": replace each PRESENT scan slot with
            # the batch mean of that (block,modality) group over the samples where it
            # is present. Keeps real embedding statistics (a generic average brain) in
            # the same placeholders, so the model is NOT pushed off-manifold the way
            # scan_zero is -- the scan analog of delta_mean. Absent slots stay zero;
            # block_present is untouched, so the prompt structure is identical.
            if chunks.shape[0] > 1:
                bp = block_present[:, 0:d0].to(chunks.dtype)[:, :, None, None]  # [B,d0,1,1]
                s = (chunks[:, 0:d0] * bp).sum(0, keepdim=True)
                cnt = bp.sum(0, keepdim=True).clamp(min=1.0)
                mean_s = (s / cnt).expand_as(chunks[:, 0:d0])
                keep = bp.bool().expand_as(chunks[:, 0:d0])
                chunks[:, 0:d0] = torch.where(keep, mean_s, chunks[:, 0:d0])
        elif mode == "vis_roll":
            # whole visual route ablated: roll EVERY block (scan+delta, or just
            # scan when include_delta=False) across the batch -> the wrong
            # patient's entire visual input, same distribution. block_present is
            # left as the real sample's, so the prompt structure is unchanged.
            if chunks.shape[0] > 1:
                chunks[:] = torch.roll(chunks, shifts=1, dims=0)
        else:
            raise ValueError(mode)
        return chunks, block_present

    model._assemble_visual = wrapped


@torch.no_grad()
def mean_ce(model, loader, prompts, device, mode, n_batches):
    model._probe_mode = mode
    tot, n = 0.0, 0
    for step, batch in enumerate(loader):
        if step >= n_batches:
            break
        prompts.set_reports(list(batch["prior_report"]))   # val: report always present
        lm_loss, *_ = model(
            tokens_main=batch["tokens_main"].to(device),
            coords_main=batch["coords_main"].to(device),
            present_main=batch["present_main"].to(device),
            tokens_ref=batch["tokens_ref"].to(device),
            coords_ref=batch["coords_ref"].to(device),
            present_ref=batch["present_ref"].to(device),
            prompt_table=prompts,
            labels=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            sentence_input_ids=None,
            sentence_attn=batch["sentence_attn"].to(device),
            sentence_mask=batch["sentence_mask"].to(device),
            content_mask=None,
            token_weights=batch["token_weights"].to(device),
            swap_perm=None, swap_valid=None,
        )
        tot += float(lm_loss)
        n += 1
    return tot / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/nv_stage4_priorreport_5.pt",
                    help="epoch-5 (or any) priorreport checkpoint to probe")
    ap.add_argument("--base", default="nv_stage2_reportgen.pt")
    ap.add_argument("--diff-checkpoint", default="nv_stage3_deltaunsup_extended.pt")
    ap.add_argument("--csv", default="/home/data/BRAIN_DIFF_S4/splits_extended")
    ap.add_argument("--image-csv", default="/home/data/BRAIN_DIFF_S4/image_extended.csv")
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--max-caption-length", type=int, default=480)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cap-gib", type=float, default=0,
                    help="cap this process's GPU memory (GiB) so it can share a GPU "
                         "with an active training run without OOM-ing it")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    # Cap footprint so a probe sharing a GPU with an active training run cannot OOM it.
    if getattr(args, "cap_gib", 0):
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.cap_gib / total), device.index)
        print(f"[guard] capped at {args.cap_gib:.0f} GiB of {total:.0f}", flush=True)

    _, val_loader, _ = make_loaders(
        args.csv, args.image_csv, args.batch_size, num_workers=4,
        max_caption_length=args.max_caption_length, seed=10, distributed=False)

    model = build_model(args.base, args.ckpt, args.diff_checkpoint, device, args)
    patch_ablation(model)
    prompts = PriorReportPrompts(model.decoder.tokenizer, include_delta=True,
                                 max_prompt_length=MAX_PROMPT_LENGTH,
                                 max_report_tokens=MAX_REPORT_TOKENS)

    print(f"\nprobe_delta_reliance  ckpt={args.ckpt}  n_batches={args.n_batches} "
          f"bs={args.batch_size}\n", flush=True)
    real, n = mean_ce(model, val_loader, prompts, device, "real", args.n_batches)
    real2, _ = mean_ce(model, val_loader, prompts, device, "real", args.n_batches)
    print(f"  DETERMINISM CHECK  real={real:.5f}  real2={real2:.5f}  |Δ|={abs(real2-real):.5f}"
          f"   (must be ~0, else the loader is non-deterministic and every dCE below is noise)",
          flush=True)
    dzero, _ = mean_ce(model, val_loader, prompts, device, "delta_zero", args.n_batches)
    droll, _ = mean_ce(model, val_loader, prompts, device, "delta_roll", args.n_batches)
    dmean, _ = mean_ce(model, val_loader, prompts, device, "delta_mean", args.n_batches)
    szero, _ = mean_ce(model, val_loader, prompts, device, "scan_zero", args.n_batches)
    cv = getattr(model, "_probe_delta_cv", float("nan"))

    print(f"  val batches used            {n}")
    print(f"  real                LM CE   {real:.4f}")
    print(f"  delta_zero  (OOD)   LM CE   {dzero:.4f}   dCE {dzero - real:+.4f}")
    print(f"  delta_roll  (clean) LM CE   {droll:.4f}   dCE {droll - real:+.4f}")
    print(f"  delta_mean  (clean) LM CE   {dmean:.4f}   dCE {dmean - real:+.4f}")
    print(f"  scan_zero (upper)   LM CE   {szero:.4f}   dCE {szero - real:+.4f}")
    print(f"  delta block cross-sample CV  {cv:.4f}   (near 0 => near-constant delta = collapse)")

    # The IN-DISTRIBUTION controls (roll, mean) are the criterion; delta_zero is
    # discarded as a zeros-are-OOD artifact. If the decoder used pair-specific delta
    # content, scrambling it (roll) or averaging it away (mean) must raise CE.
    d_clean = max(droll - real, dmean - real)
    print()
    print(f"  clean delta dCE (max roll/mean)  {d_clean:+.4f}   <- the decision number")
    print(f"  delta_zero dCE (OOD, ignored)    {dzero - real:+.4f}")
    if d_clean < 0.01:
        verdict = ("NOT USED / COLLAPSED — the decoder is invariant to pair-specific delta "
                   "content (in-distribution scramble moves CE ~0). delta_zero's large dCE is "
                   "the zeros-are-OOD artifact. Restart with connector.delta UNFROZEN.")
    elif d_clean < 0.03:
        verdict = "WEAK — delta barely used pair-specifically; borderline."
    else:
        verdict = "USED — scrambling the pair-specific delta raises CE. Frozen delta is fine."
    print(f"\n  VERDICT: {verdict}\n", flush=True)


if __name__ == "__main__":
    main()
