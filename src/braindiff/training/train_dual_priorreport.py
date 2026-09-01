"""S4 variant that conditions on the prior REPORT as well as the prior image.

train_dual.py and dataloaders/dual.py are not modified -- this is a parallel path, so the
existing S4 stage stays reproducible and comparable.

Prior report text alone predicts change-vs-stable at 0.693 AUC against 0.564 for the frozen
imaging stack (0.803 = the current-report ceiling), so the model was being asked to
re-derive from voxels what the prior radiologist had already written.

Same losses, freeze spec, checkpoint contract and DDP structure as train_dual.py. The
differences:

  * PriorReportDataset supplies `prior_report` per row, joined on study-UID pairs recovered
    from the item's paths -- positional joins are wrong because the parent dataset drops
    rows and generated_report is not unique.
  * PriorReportPrompts builds prompts per sample rather than from PromptTable's cache, since
    the report is not a function of (present_ref, present_main).
  * batch_size defaults to 8. The prompt grows from 205 tokens to a median 493, so the
    prefix goes ~961 -> ~1250; contrastive negatives shrink 48 -> 32 global as the cost.

Risk to watch in the epoch samples: the model can copy the prior report and call everything
stable. The metric that catches it is non-Stable change-recall, not val CE.
"""
import glob
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from braindiff.models.captioner import DeltaDiffCaptioner_Qwen3
from braindiff.models.paths import decoder_dir
from braindiff.models.prior_report_prompts import PriorReportPrompts
from braindiff.data.dual import CHANGE_CLASSES, CHANGE_CLASS_TO_IDX
from braindiff.data.dual_priorreport import get_prior_report_dataloader
from braindiff.training import notify
from braindiff.training.checkpoint import load_stage_checkpoint
from braindiff.training.losses import make_swap_perm
from braindiff.eval.temporal_score import score as temporal_score
from braindiff.training.ablate_partners import batch_partner_perm
from braindiff.training.freeze import apply_trainable
from braindiff.training.val_mask import build_content_masks
from braindiff.training.train_dual import (IMG_SIZE, ddp_setup, cleanup, resolve_split_csvs,
                                log_delta_stats)

# MEASURED over all 8,648 prior reports (Qwen3 tokenizer): p50 296, p90 546,
# p95 649, p99 852, max 31,006 -- one pathological report forces a cap to exist.
# 384 truncated 28.3% of reports; 1024 truncates 0.2%, and the full prompt then runs
# p50 492 / p99 1119 / max ~1240. Qwen3 has sliding_window null and 40,960 positions,
# so the binding constraint is memory, not the architecture.
MAX_PROMPT_LENGTH = 1536
# 768 truncates 2.0% of reports, against 28.3% at the previous 384. 1024 would reach
# 0.2% but the memory probe could not confirm headroom for it, and the real run's
# measured 68 GiB/GPU at bs 8 / cap 384 is the only trustworthy datapoint -- the
# worst-case probe pads every row to the batch maximum, which real batches never do.
MAX_REPORT_TOKENS = 768


def make_loaders(csv_path, image_csv, batch_size, num_workers, max_caption_length,
                 seed, distributed=False, content_weight=1.0):
    tokenizer = AutoTokenizer.from_pretrained(decoder_dir())
    rank = int(os.environ.get("RANK", 0))
    paths = resolve_split_csvs(csv_path, seed, rank, distributed)
    if rank == 0:
        print("[splits] " + ", ".join(
            f"{s}={os.path.basename(p) if p else 'NONE'}" for s, p in paths.items()),
            flush=True)
    out = {}
    for name, is_train in (("train", True), ("val", False), ("test", False)):
        if paths[name] is None:
            out[name] = None
            continue
        out[name] = get_prior_report_dataloader(
            csv_file=paths[name], image_csv=image_csv, img_size=IMG_SIZE,
            batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer,
            max_caption_length=max_caption_length, is_train=is_train,
            distributed=distributed, content_weight=content_weight)
    return out["train"], out["val"], out["test"]


def run_epoch(model, raw_model, loader, optimizer, device, is_train, desc, prompts,
              is_main=True, distributed=False, contrastive_weight=0.0,
              scheduler=None, grad_clip=1.0, content_masks=None, grad_accum=1,
              cf_weight=0.0, prior_report_dropout=0.0):
    model.train(is_train)
    total_loss = total_content = total_cf = 0.0
    n_steps = max(len(loader), 1)
    forward_model = model if is_train else raw_model
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    from tqdm import tqdm
    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_main)
    with ctx:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            # Boundary = the micro-step where the window closes and the optimiser
            # actually steps. Accumulation is what buys back the effective batch
            # after the prior report forced batch_size 8 -> 6.
            is_boundary = ((step + 1) % grad_accum == 0) or (step + 1 == n_steps)
            # Bind this batch's reports BEFORE the forward -- the captioner calls
            # prompts.batch() internally and has no way to pass them through.
            # Prior-report dropout (TRAIN ONLY). The report supplies ~74% of the
            # measured gain, so a model trained with it always present has no reason
            # to read the images. Withholding it on a fraction of steps removes that
            # option and forces image-based capability to develop; inference always
            # gets the report.
            reports = list(batch["prior_report"])
            if is_train and prior_report_dropout > 0:
                reports = ["" if random.random() < prior_report_dropout else r
                           for r in reports]
            prompts.set_reports(reports)
            content_mask = (content_masks[batch["sample_idx"]].to(device)
                            if content_masks is not None else None)
            # Contrastive all_gathers over the global batch; accumulation cannot
            # enlarge it, so computing it off-boundary is pure cost -- measured at
            # ~20% wall clock, not a correctness issue.
            want_contrastive = is_train and is_boundary
            # Counterfactual on the PRIOR IMAGE with the report held fixed: the
            # hinge requires NLL to be worse when the prior scan belongs to someone
            # else. A model ignoring the images cannot satisfy it. This is the
            # shuffled-image control turned into a training signal -- and unlike the
            # oracle-conditioned case it is not leaked, because the PRIOR report does
            # not state the current findings.
            swap_perm = has_partner = None
            if is_train and cf_weight > 0:
                swap_perm, has_partner = make_swap_perm(batch["change_label"].to(device))
            lm_loss, contrastive_loss, cf_loss, _, content_loss = forward_model(
                tokens_main=batch["tokens_main"].to(device),
                coords_main=batch["coords_main"].to(device),
                present_main=batch["present_main"].to(device),
                tokens_ref=batch["tokens_ref"].to(device),
                coords_ref=batch["coords_ref"].to(device),
                present_ref=batch["present_ref"].to(device),
                prompt_table=prompts,
                labels=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                sentence_input_ids=(batch["sentence_input_ids"].to(device)
                                    if want_contrastive else None),
                sentence_attn=batch["sentence_attn"].to(device),
                sentence_mask=batch["sentence_mask"].to(device),
                content_mask=content_mask,
                token_weights=batch["token_weights"].to(device),
                swap_perm=swap_perm, swap_valid=has_partner,
            )
            loss = lm_loss
            if is_train and cf_weight > 0 and cf_loss is not None:
                loss = loss + cf_weight * cf_loss
                total_cf += float(cf_loss)
            if is_train and contrastive_loss is not None:
                # x grad_accum so the term's expected contribution is unchanged by
                # how many micro-steps it is computed on.
                loss = loss + contrastive_weight * grad_accum * contrastive_loss

            if is_train:
                (loss / grad_accum).backward()
                if is_boundary:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += lm_loss.item()
            if content_loss is not None:
                total_content += content_loss.item()
            pbar.set_postfix(loss=f"{lm_loss.item():.4f}")

    mean_loss = total_loss / n_steps
    mean_content = total_content / n_steps
    mean_cf = total_cf / n_steps
    if distributed:
        t = torch.tensor([mean_loss, mean_content], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean_loss, mean_content = t[0].item(), t[1].item()
    return mean_loss, mean_content, mean_cf


def sample_captions(raw_model, loader, device, prompts, max_new_tokens, n=2):
    raw_model.eval()
    batch = next(iter(loader))
    prompts.set_reports(batch["prior_report"][:n])
    tokenizer = raw_model.decoder.tokenizer
    with torch.no_grad():
        caps = raw_model.generate_caption_batch(
            tokens_main=batch["tokens_main"][:n].to(device),
            coords_main=batch["coords_main"][:n].to(device),
            present_main=batch["present_main"][:n].to(device),
            tokens_ref=batch["tokens_ref"][:n].to(device),
            coords_ref=batch["coords_ref"][:n].to(device),
            present_ref=batch["present_ref"][:n].to(device),
            prompt_table=prompts, max_new_tokens=max_new_tokens,
            repetition_penalty=1.0)
    pairs = []
    for i, cap in enumerate(caps):
        gt_ids = batch["input_ids"][i]
        gt = tokenizer.decode(gt_ids[gt_ids != -100], skip_special_tokens=True)
        print(f"    [{i}] GT:  {gt}", flush=True)
        print(f"    [{i}] GEN: {cap}", flush=True)
        pairs.append((gt, cap))
    return pairs



@torch.no_grad()
def select_score(raw_model, loader, device, prompts, max_new_tokens, patients,
                 n_pairs=256, gap_weight=1.0, distributed=False):
    """Selection metric: generated-report quality plus how much the images matter.

Val CE picks epoch 3, while epoch 10 is tied on rg_er (-0.007 [-0.015, +0.000]), better on
METEOR and non-Stable change-recall, and uses the images more (image gap +0.046 vs +0.035,
difference +0.011 [+0.001, +0.021]). CE cannot distinguish paraphrasing the prior report
from reading the scans.

So generate a fixed subset twice, once with the true images and once with another patient's,
and score both with temporal_score (CPU, Pearson 0.998 against RadGraph within a stage,
5.4 ms/report; RadGraph cannot run in-loop as it pins transformers <5).

    criterion = quality(true) + gap_weight * (quality(true) - quality(shuffled))

Higher is better; gap_weight 0 reproduces plain quality selection. NOT validated against
downstream rg_er -- used for ranking checkpoints, not for a final claim.
    """
    raw_model.eval()
    world = dist.get_world_size() if distributed else 1
    per_rank = max(1, n_pairs // world)

    hyps_t, hyps_s, refs = [], [], []
    for batch in loader:
        if len(refs) >= per_rank:
            break
        ii = batch["sample_idx"].tolist()
        common = dict(max_new_tokens=max_new_tokens, num_beams=1,
                      repetition_penalty=1.0, no_repeat_ngram_size=0)
        prompts.set_reports(batch["prior_report"])
        hyps_t.extend(raw_model.generate_caption_batch(
            tokens_main=batch["tokens_main"].to(device),
            coords_main=batch["coords_main"].to(device),
            present_main=batch["present_main"].to(device),
            tokens_ref=batch["tokens_ref"].to(device),
            coords_ref=batch["coords_ref"].to(device),
            present_ref=batch["present_ref"].to(device),
            prompt_table=prompts, **common))
        perm, _ = batch_partner_perm([patients[i] for i in ii])
        prompts.set_reports(batch["prior_report"])
        hyps_s.extend(raw_model.generate_caption_batch(
            tokens_main=batch["tokens_main"][perm].to(device),
            coords_main=batch["coords_main"][perm].to(device),
            present_main=batch["present_main"][perm].to(device),
            tokens_ref=batch["tokens_ref"][perm].to(device),
            coords_ref=batch["coords_ref"][perm].to(device),
            present_ref=batch["present_ref"][perm].to(device),
            prompt_table=prompts, **common))
        refs.extend(batch["caption"])

    hyps_t, hyps_s, refs = hyps_t[:per_rank], hyps_s[:per_rank], refs[:per_rank]
    q_true = temporal_score(hyps_t, refs)["triple_f1"]
    q_shuf = temporal_score(hyps_s, refs)["triple_f1"]
    v = torch.tensor([q_true, q_shuf], device=device)
    if distributed:
        dist.all_reduce(v, op=dist.ReduceOp.AVG)
    q_true, q_shuf = v[0].item(), v[1].item()
    return q_true + gap_weight * (q_true - q_shuf), q_true, q_true - q_shuf


def main(csv, image_csv, epochs=100, batch_size=8, lr=1e-4, num_workers=8,
         max_caption_length=480, grad_accum=1, lora_r=16, lora_alpha=32,
         lora_dropout=0.05, content_weight=1.0,
         trainable=("connector.delta", "diff_encoder", "embeddings",
                    "contrastive", "decoder_lora"),
         patience=0, cf_weight_train=0.0, prior_report_dropout=0.0,
         select_every=1, select_pairs=256, gap_weight=1.0,
         save_dir="checkpoints",
         save_name="nv_stage4_priorreport.pt", seed=10,
         use_lora=True, use_vision_lora=True, vision_lora_r=32, vision_lora_alpha=64,
         vision_lora_dropout=0.05, num_queries=64, include_delta=True,
         contrastive_weight=0.0, cf_weight=0.0, change_weight=0,
         counterfactual_margin=0.5, checkpoint=None, diff_checkpoint=None,
         checkpoint_every=5, **_ignored):
    rank, local_rank, world_size = ddp_setup()
    distributed = world_size > 1
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    train_loader, val_loader, test_loader = make_loaders(
        csv, image_csv, batch_size, num_workers, max_caption_length, seed,
        distributed=distributed, content_weight=content_weight)
    if is_main:
        print(f"Split sizes — train: {len(train_loader.dataset)}  "
              f"val: {len(val_loader.dataset)}", flush=True)

    for _r in range(world_size if distributed else 1):
        if not distributed or rank == _r:
            model = DeltaDiffCaptioner_Qwen3(
                single_timepoint=False, use_vision_lora=use_vision_lora,
                vision_lora_r=vision_lora_r, vision_lora_alpha=vision_lora_alpha,
                vision_lora_dropout=vision_lora_dropout, num_queries=num_queries,
                include_delta=include_delta, lora_r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, use_lora=use_lora,
                pretrained_connector=(checkpoint is None),
                max_caption_length=max_caption_length,
                num_change_classes=len(CHANGE_CLASSES) if change_weight > 0 else 0,
                counterfactual_margin=counterfactual_margin, device=device).to(device)
        if distributed:
            dist.barrier()

    if checkpoint:
        p = os.path.join(save_dir, checkpoint)
        load_stage_checkpoint(model, torch.load(p, map_location=device), label=p,
                              is_main=is_main)
    if diff_checkpoint:
        p = os.path.join(save_dir, diff_checkpoint)
        load_stage_checkpoint(model, torch.load(p, map_location=device), label=p,
                              is_main=is_main,
                              strict_groups=("diff_encoder", "connector.delta"))

    apply_trainable(model, trainable, is_main=is_main)
    prompts = PriorReportPrompts(model.decoder.tokenizer, include_delta=include_delta,
                                 max_prompt_length=MAX_PROMPT_LENGTH,
                                 max_report_tokens=MAX_REPORT_TOKENS)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if distributed else model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.ndim > 1], "weight_decay": 0.01},
         {"params": [p for p in params if p.ndim <= 1], "weight_decay": 0.0}], lr=lr)
    total_steps = epochs * max(1, len(train_loader) // grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps),
                                                total_steps)

    tokenizer = AutoTokenizer.from_pretrained(decoder_dir())
    content_masks = build_content_masks(
        [it["caption"] for it in val_loader.dataset.items], tokenizer, max_caption_length)

    if is_main:
        n_par = sum(p.numel() for p in params)
        notify.send(f"*{save_name[:-3]}* started (PRIOR REPORT conditioning)\n"
                    f"{epochs} epochs x {len(train_loader)} steps   "
                    f"bs {batch_size}/rank x {world_size} = {batch_size * world_size}\n"
                    f"lr {lr:g}   {n_par/1e6:.1f}M trainable\n"
                    f"prior report capped at {MAX_REPORT_TOKENS} tokens "
                    f"(Impression preserved)\n"
                    f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val"
                    + (f"\nwarm start: {checkpoint}" if checkpoint else "")
                    + (f" + delta {diff_checkpoint}" if diff_checkpoint else ""))

    # Patient ids for the selection subset's shuffled arm. Same caption-join used by
    # the eval harness; items carry no patient_uid.
    import pandas as pd, glob as _glob
    _v = pd.read_csv(_glob.glob(os.path.join(csv, "*val.csv"))[0])
    _norm = lambda x: " ".join(str(x).split())
    _lut = {}
    for _r, _p in zip(_v["generated_report"], _v["patient_uid"]):
        _lut.setdefault(_norm(_r), _p)
    val_patients = [_lut.get(_norm(it["caption"]), f"__u{i}")
                    for i, it in enumerate(val_loader.dataset.items)]

    best_sel, stale = float("-inf"), 0
    best_val_ce = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if distributed:
            train_loader.batch_sampler.set_epoch(epoch)
        train_loss, _, train_cf = run_epoch(model, raw_model, train_loader, optimizer, device,
                                  True, f"Epoch {epoch}/{epochs} [train]", prompts,
                                  is_main=is_main, distributed=distributed,
                                  contrastive_weight=contrastive_weight,
                                  scheduler=scheduler, grad_accum=grad_accum,
                                  cf_weight=cf_weight,
                                  prior_report_dropout=prior_report_dropout)
        val_loss, val_content, _ = run_epoch(raw_model, raw_model, val_loader, None, device,
                                          False, f"Epoch {epoch}/{epochs} [val]", prompts,
                                          is_main=is_main, distributed=distributed,
                                          content_masks=content_masks)

        # Selection runs on EVERY rank (it all-reduces), so it must sit outside the
        # is_main block or the collective deadlocks.
        # No in-loop selection: every checkpoint_every epoch is kept and ranked
        # afterwards by eval_checkpoints_pr.py on generated reports, which is the
        # only signal that can see image reliance. val CE is still tracked, but only
        # to mark a reference checkpoint -- it is explicitly NOT the criterion.
        sel_value = q_true = gap = None
        is_best = val_loss < best_val_ce

        if is_main:
            samples = sample_captions(raw_model, val_loader, device, prompts,
                                      round(max_caption_length * 1.1))
            marker = ""
            if is_best or epoch % checkpoint_every == 0 or epoch == epochs:
                state = {k: v for k, v in raw_model.state_dict().items()
                         if not k.startswith("decoder.") or "lora_" in k}
                if epoch % checkpoint_every == 0 or epoch == epochs:
                    torch.save(state, os.path.join(save_dir, f"{save_name[:-3]}_{epoch}.pt"))
                    marker = "  *"
                if is_best:
                    torch.save(state, os.path.join(save_dir, save_name))
                    marker += "  BEST"
            trunc = (f"{prompts.n_truncated}/{prompts.n_seen} prompts truncated"
                     if prompts.n_seen else "")
            sel_txt = ("" if sel_value is None else
                       f"  sel={sel_value:.4f} (q={q_true:.4f} gap={gap:+.4f})")
            print(f"Epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  content={val_content:.4f}  cf={train_cf:.4f}{sel_txt}{marker}"
                  f"   {trunc}", flush=True)
            # Slack EVERY epoch, and outside any try/except that guards saving --
            # a webhook 500 must never abort a rank (train_dual.py documents why).
            notify.epoch_report(
                stage=save_name[:-3], epoch=epoch, epochs=epochs,
                train_loss=train_loss, val_loss=val_loss,
                best_val=None, marker=marker,
                minutes=(time.time() - t0) / 60.0, samples=samples,
                stale=(0 if is_best else stale + 1), patience=patience,
                extra=[(f"SELECTION {sel_value:.4f}  = quality {q_true:.4f} "
                        f"+ {gap_weight:g} x image-gap {gap:+.4f}   (higher better)"
                        if sel_value is not None else "no selection this epoch"),
                       f"val CE {val_loss:.4f} / content {val_content:.4f}  (diagnostic only)",
                       f"cf {train_cf:.4f}   report-dropout {prior_report_dropout:g}",
                       trunc],
                stale_unit="selection rounds")

        if val_loss < best_val_ce:
            best_val_ce, stale = val_loss, 0
        else:
            stale += 1
            if patience and stale >= patience:
                if is_main:
                    print(f"Early stop: {stale} rounds without improvement "
                          f"(best selection {best_sel:.4f}).", flush=True)
                break

    if is_main:
        notify.send(f"*{save_name[:-3]}* finished — {epochs} epochs, checkpoints every "
                    f"{checkpoint_every}. Rank them with eval_checkpoints_pr.py; "
                    f"val CE is NOT the criterion.")
    cleanup(world_size)
