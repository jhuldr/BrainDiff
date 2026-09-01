import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from braindiff.models.paths import decoder_dir

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from braindiff.models.captioner import DeltaDiffCaptioner_Qwen3
from braindiff.models.prompts import PromptTable
from braindiff.training.freeze import apply_trainable
from braindiff.training.losses import make_swap_perm
from braindiff.training.checkpoint import load_stage_checkpoint
from braindiff.training import notify
from braindiff.training.splits import s1_val_pool, s2_val_studies
from braindiff.data.single import *



# NeuroVFM grid: 193x229x193 @1mm -> 1x1x4mm -> 12x14x12 = 2016 tokens.
IMG_SIZE = (48, 224, 192)
# Decoder resolved through the HF cache (models/paths.py): downloaded once, never again.

# Block repeated 20-grams when generating. Long enough that the clinical phrases
# reports legitimately repeat ("are of normal signal intensity", ~7 tokens) are
# untouched; short enough that a repeated sentence cannot be emitted.
NO_REPEAT_NGRAM = 20

def ddp_setup():
    """Init the process group from torchrun env vars. Returns (rank, local_rank, world_size)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup(world_size):
    if world_size > 1:
        dist.destroy_process_group()


def split_csv(csv_path, seed):
    """Split into train/val by STUDY MEMBERSHIP, not row position.

    The old 95/5 positional split was drawn independently of S1's, which put
    1,722 of 1,905 val studies (90.4%) inside S1's training set -- S1's `pcls`
    task is the whole S2 frame concatenated (build_unified_dataframe.py:39), so
    the two stages share all 38,086 studies. Val is now drawn from the studies
    S1 held out, which are the only ones the model has never trained on.

    `seed` is accepted for signature compatibility and is unused: membership
    comes from the pinned manifest, so the split no longer moves when the CSV is
    rewritten (which already happened once, silently).
    """
    df = pd.read_csv(csv_path, low_memory=False)
    val_uids = s2_val_studies()
    is_val = df["study_uid"].isin(val_uids)
    train, val = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)

    # Guard rail at the point of use: val must be exactly the requested size and
    # must live entirely inside S1's held-out pool. If either fails the leak is
    # back, and a silent 90% contamination is not something to discover later.
    pool = s1_val_pool()
    assert len(val) == len(val_uids), f"val {len(val)} != {len(val_uids)} requested studies"
    assert set(val["study_uid"]) <= pool, "S2 val contains studies outside S1's held-out pool"
    assert not set(train["study_uid"]) & set(val["study_uid"]), "S2 train/val overlap"
    return train, val


def make_loaders(csv_path, batch_size, num_workers, max_caption_length, seed, distributed: bool = False,
                 content_weight: float = 1.0):
    tokenizer = AutoTokenizer.from_pretrained(decoder_dir())

    train_df, val_df = split_csv(csv_path, seed)

    tmp_dir = "/tmp/braindiff_splits_single"
    os.makedirs(tmp_dir, exist_ok=True)
    splits = {"train": (train_df, True), "val": (val_df, False)}
    tmp_paths = {name: os.path.join(tmp_dir, f"{name}.csv") for name in splits}

    # split_csv is deterministic (fixed seed), so every rank computes the same
    # DataFrames — but writing the shared tmp path from every rank is a race:
    # pandas.to_csv isn't atomic, so a rank reading the file can see another
    # rank's write mid-flight and silently load a truncated (shorter) CSV. That
    # gives ranks different dataset/batch counts, which desyncs every collective
    # downstream. Only rank 0 writes; everyone else waits at the barrier before
    # reading.
    is_main = not distributed or dist.get_rank() == 0
    if is_main:
        for name, (df, _) in splits.items():
            df.to_csv(tmp_paths[name], index=False)
    if distributed:
        dist.barrier()

    loaders = {}
    for name, (_, is_train) in splits.items():
        loaders[name] = get_diff_caption_dataloader(
            csv_file=tmp_paths[name],
            img_size=IMG_SIZE,
            batch_size=batch_size,
            num_workers=num_workers,
            tokenizer=tokenizer,
            max_caption_length=max_caption_length,
            is_train=is_train,
            distributed=distributed,
            # BOTH splits are content-weighted: val CE is now the selection
            # criterion, so it has to be the same quantity training minimises --
            # an unweighted val CE would pick the checkpoint that best recites
            # the normal template, which is the exact failure content_weight
            # exists to fix. Safe on val because `reorder_report_sections` is
            # train-only, so val token positions (and therefore the weights) are
            # fixed for the whole run.
            # Cost: val loss is no longer comparable to earlier runs, or across
            # runs with different content_weight. Compare rg_er instead.
            content_weight=content_weight,
        )

    return loaders["train"], loaders["val"]

def log_feature_stats(model, batch, device):
    toks_main = batch["tokens_main"][:4].to(device)
    crds_main = batch["coords_main"][:4].to(device)
    pres_main = batch["present_main"][:4].to(device)
    with torch.no_grad():
        feats = model.encode_multimodal(toks_main, crds_main, pres_main)
        feat_norm = feats.norm(dim=-1).mean().item()
    print(f"  [feat] main={feat_norm:.3f}")


def print_sample_captions(model, val_loader, device, max_output_length, n=2,
                          prompt_table=None):
    model.eval()
    batch = next(iter(val_loader))
    toks_main = batch["tokens_main"][:n].to(device)
    crds_main = batch["coords_main"][:n].to(device)
    pres_main = batch["present_main"][:n].to(device)
    tokenizer = model.decoder.tokenizer

    print("  [samples]")
    with torch.no_grad():
        caps = model.generate_caption_batch(
            tokens_main=toks_main, coords_main=crds_main, present_main=pres_main,
            prompt_table=prompt_table, max_new_tokens=max_output_length,
            repetition_penalty=1.0, no_repeat_ngram_size=NO_REPEAT_NGRAM)
    # Returned as well as printed so the epoch notifier can ship the same pairs
    # without generating a second time -- generation is the expensive part here.
    pairs = []
    for i, caption in enumerate(caps):
        gt_ids = batch["input_ids"][i]
        gt = tokenizer.decode(gt_ids[gt_ids != -100], skip_special_tokens=True)
        print(f"    [{i}] GT:  {gt}")
        print(f"    [{i}] GEN: {caption}")
        pairs.append((gt, caption))
    return pairs


def run_epoch(model, raw_model, loader, optimizer, device, is_train, desc,
              is_main=True, distributed=False, contrastive_weight=0.0,
              cf_weight=0.0, scheduler=None, grad_clip=1.0, prompt_table=None,
              grad_accum=1):
    model.train(is_train)
    total_loss = 0.0
    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_main)

    # Use raw_model for validation to avoid DDP sync
    forward_model = model if is_train else raw_model

    n_steps = len(loader)
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            toks_main = batch["tokens_main"].to(device)
            crds_main = batch["coords_main"].to(device)
            pres_main = batch["present_main"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            sent_ids = batch["sentence_input_ids"].to(device)
            sent_attn = batch["sentence_attn"].to(device)
            sent_mask = batch["sentence_mask"].to(device)
            tok_w = batch["token_weights"].to(device)

            # Single-timepoint counterfactual: the swapped block is the only scan,
            # so this is the grounding version of the dual stage's prior swap --
            # and it runs on 38k reports rather than 8.6k pairs.
            swap_perm = has_partner = None
            if cf_weight > 0:
                swap_perm, has_partner = make_swap_perm(batch["group"].to(device))

            # Accumulation boundary: the micro-step where gradients are
            # all-reduced and the optimiser actually steps.
            #
            # The sentence-contrastive term runs only on that step. MEASURED
            # 2026-08-08: its cross-rank all_gather inside a no_sync window does
            # NOT hang (test_ddp_smoke --con-every, 4 ranks, clean exit) -- the
            # reason is cost. Accumulation never enlarges the contrastive batch
            # (it sees bs-per-rank either way), so running it every micro-step buys
            # nothing and costs 20% wall-clock (3604 vs 2994 ms/micro-step).
            # Scaled by grad_accum below so the expected contribution is unchanged.
            is_boundary = ((step + 1) % grad_accum == 0) or (step + 1 == n_steps)
            want_contrastive = contrastive_weight > 0 and (is_boundary or not is_train)

            lm_loss, contrastive_loss, cf_loss, _, _ = forward_model(
                tokens_main=toks_main,
                coords_main=crds_main,
                present_main=pres_main,
                prompt_table=prompt_table,
                labels=input_ids,
                attention_mask=attn_mask,
                sentence_input_ids=sent_ids if want_contrastive else None,
                sentence_attn=sent_attn,
                sentence_mask=sent_mask,
                swap_perm=swap_perm,
                swap_valid=has_partner,
                # Both splits: val CE is the selection criterion and must be the
                # same quantity training minimises. The val loader is built with
                # the same content_weight, so on val this is already whatever
                # make_loaders produced (all-ones when content_weight == 1.0,
                # which is an exact no-op).
                token_weights=tok_w,
            )
            loss = lm_loss
            if contrastive_loss is not None:
                # x grad_accum because it only runs on the boundary step, so its
                # expected contribution over an accumulation window matches a run
                # that computed it every micro-step.
                loss = loss + contrastive_weight * grad_accum * contrastive_loss
            if cf_weight > 0 and cf_loss is not None:
                loss = loss + cf_weight * cf_loss

            if is_train:
                scaled = loss / grad_accum
                if distributed and not is_boundary:
                    # Skip the gradient all-reduce on non-boundary micro-steps.
                    with model.no_sync():
                        scaled.backward()
                else:
                    scaled.backward()
                if is_boundary:
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], grad_clip)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            con = contrastive_loss.item() if contrastive_loss is not None else 0.0
            cf = cf_loss.item() if cf_loss is not None else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", lm=f"{lm_loss.item():.4f}",
                             con=f"{con:.4f}", cf=f"{cf:.4f}")

    mean_loss = total_loss / len(loader)

    # Average the per-rank mean across ranks for an accurate figure.
    if distributed:
        t = torch.tensor(mean_loss, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean_loss = t.item()

    return mean_loss


def main(
    csv: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 4,
    max_caption_length: int = 1024,   # MEASURED: covers 99.89% of S2 reports
                                      # (Qwen3 tokens; the old 610 truncated 3.94%)
    grad_accum: int = 1,
    # Declarative freezing. Authoritative over any use_*/trainable ctor flag.
    trainable: tuple = ("connector.scan", "connector.proj", "embeddings",
                        "contrastive", "decoder_lora"),
    # Selection: val loss, every epoch, lower is better. Triple-F1 is gone from
    # this loop entirely -- generation was its whole cost and it no longer picks
    # anything; evaluate_checkpoints.py + radgraph_score.py rank the periodic
    # checkpoints offline. `patience` counts EPOCHS without improvement.
    patience: int = 15,
    save_dir: str = "checkpoints",
    save_name: str = "best_model_single.pt",
    seed: int = 10,
    use_lora: bool = True,
    use_vision_lora: bool = True,
    vision_lora_r: int = 64,
    vision_lora_alpha: int = 128,
    vision_lora_dropout: float = 0.05,
    lora_r: int = 16,
    lora_alpha: int = 32,
    # LoRA is BUILT whenever use_vision_lora is set (it decides the
    # checkpoint key names); this controls whether it TRAINS.
    vision_lora_trainable: bool = True,
    # Perceiver latents per block -- the visual token count the decoder sees.
    # A change here IS a shape change on connector.*.queries, so a later stage
    # that disagrees fails loudly in checkpoint.py rather than silently.
    num_queries: int = 64,
    include_delta: bool = False,
    checkpoint: str = None,
    contrastive_weight: float = 0.0,
    cf_weight: float = 0.0,
    counterfactual_margin: float = 0.5,
    # >1 downweights normality/absence sentences in the captioning CE, rescaled
    # so the loss magnitude (and therefore the LR schedule) is unchanged.
    # 1.0 is an exact no-op.
    content_weight: float = 1.0,
    checkpoint_every: int = 5,
):
    rank, local_rank, world_size = ddp_setup()
    distributed = world_size > 1
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Device: {device}  world_size: {world_size}")

    # All splits are sharded across ranks; val/test loss is all-reduced.
    train_loader, val_loader = make_loaders(
        csv, batch_size, num_workers, max_caption_length, seed,
        distributed=distributed, content_weight=content_weight,
    )
    if is_main and content_weight != 1.0:
        print(f"Content-weighted CE on TRAIN AND VAL: normality sentences x1.0, "
              f"rest x{content_weight} (renormalised to mean 1). Val loss is this "
              f"CE alone — no contrastive, no cf — and is the selection criterion, "
              f"so it is NOT comparable to runs at a different content_weight.",
              flush=True)
    if is_main:
        print(f"Split sizes — train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")


    rank = dist.get_rank() if dist.is_initialized() else 0

    print(
        f"[rank {rank}] val dataset len={len(val_loader.dataset)}, "
        f"val sampler len={len(val_loader.sampler)}, "
        f"val loader batches={len(val_loader)}",
        flush=True
    )

    # Stagger the 30 GB LLM load across ranks: 4 simultaneous reads of a 30 GB
    # shard set saturates the filesystem long before it saturates host RAM.
    for r in range(world_size if distributed else 1):
        if not distributed or rank == r:
            model = DeltaDiffCaptioner_Qwen3(
                single_timepoint=True,
                use_vision_lora=use_vision_lora,
                vision_lora_r=vision_lora_r,
                vision_lora_alpha=vision_lora_alpha,
                vision_lora_dropout=vision_lora_dropout,
                num_queries=num_queries,
                include_delta=include_delta,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                pretrained_connector=(checkpoint is None),
                max_caption_length=max_caption_length,
                counterfactual_margin=counterfactual_margin,
                device=device,
            ).to(device)
        if distributed:
            dist.barrier()

    # Warm-start from the previous curriculum stage. Partial overlap is expected:
    # freeze config and LoRA/delta modules differ across stages, so load leniently.
    if checkpoint:
        ckpt_path = os.path.join(save_dir, checkpoint)
        load_stage_checkpoint(model, torch.load(ckpt_path, map_location=device),
                              label=ckpt_path, is_main=is_main)

    # Declarative freezing -- the stage's `trainable` list is authoritative. Runs
    # AFTER the checkpoint load so nothing a load touched stays accidentally live.
    apply_trainable(model, trainable, is_main=is_main)

    prompt_table = PromptTable(model.decoder.tokenizer, single_timepoint=True,
                               include_delta=False)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if distributed else model

    # Decay only on ndim>1 params -- weight decay on norms/biases/embeddings
    # just shrinks them without regularizing anything.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in trainable if p.ndim > 1], "weight_decay": 0.01},
         {"params": [p for p in trainable if p.ndim <= 1], "weight_decay": 0.0}],
        lr=lr,
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    first_batch = next(iter(train_loader))

    # Sent here rather than at the top of main(): everything expensive that can
    # still fail -- decoder build, checkpoint load, freeze audit, both loaders --
    # is already done, so a "started" message means the run really is training and
    # not about to die on a bad checkpoint path 90 seconds in.
    if is_main:
        n_par = sum(p.numel() for p in trainable)
        notify.send(
            f"*{save_name[:-3]}* started\n"
            f"{epochs} epochs x {len(train_loader)} steps   "
            f"bs {batch_size}/rank x {world_size} = {batch_size * world_size}\n"
            f"lr {lr:g}   {n_par/1e6:.1f}M trainable   cf {cf_weight:g}   "
            f"con {contrastive_weight:g}\n"
            f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val rows"
            + (f"\nwarm start: {checkpoint}" if checkpoint else "\nno warm start"))

    # Selection is two-stage: val loss drives early stopping and keeps a
    # shortlist, radgraph_score.py over the every-5-epoch checkpoints makes the
    # final call. Lower is better, so this initializes to +inf rather than -1.
    best_val, best_ckpt, stale = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        if is_main:
            log_feature_stats(raw_model, first_batch, device)
        train_loss = run_epoch(
            model, raw_model, train_loader, optimizer, device,
            is_train=True, desc=f"Epoch {epoch}/{epochs} [train]",
            is_main=is_main, distributed=distributed, contrastive_weight=contrastive_weight,
            cf_weight=cf_weight, scheduler=scheduler,
            prompt_table=prompt_table, grad_accum=grad_accum,
        )

        # Validation runs on every rank (val_loader is sharded via
        # DistributedSampler); the loss is all-reduced so every rank agrees.
        # Sampling/checkpointing stay rank-0-only to avoid duplicate writes.
        
        # Selection criterion: the content-weighted LM CE, and nothing else.
        # Neither `contrastive_weight` nor `cf_weight` is passed, so both default
        # to 0 and the auxiliary terms are absent from val. The contrastive term
        # sits near chance (con ~ 10.0, which is why its weight was cut from 0.25
        # to 0.05), so as a printed diagnostic it was harmless but as a selection
        # criterion it is ~0.5 of noise on a quantity whose meaningful
        # epoch-to-epoch spread is far smaller. The cf hinge is a margin on a
        # random partner draw -- also noise here. What is left is the term that
        # actually measures report quality.
        val_loss = run_epoch(
            raw_model, raw_model, val_loader, None, device,
            is_train=False, desc=f"Epoch {epoch}/{epochs} [val]",
            is_main=is_main, distributed=distributed,
            # REQUIRED, and not a loss-weight choice like the two omissions above:
            # the prefix cannot be built without prompts, so forward() asserts
            # immediately. Omitting it made every val epoch raise "need prompt_ids
            # or a PromptTable" -- invisible until a run first survived a whole
            # training epoch.
            prompt_table=prompt_table,
        )

        # No in-loop generation: rg_er-based ranking is a post-hoc pass
        # (evaluate_checkpoints.py + radgraph_score.py --hyp_ref_dir) over the
        # periodic checkpoints, run after training, so it's off this loop's
        # critical path. Selection here is val loss, computed above.
        ok = 1
        marker, samples = "", []
        if is_main:
            try:
                t0 = time.time()
                print(f"[rank0] starting print_sample_captions @ {t0:.1f}", flush=True)
                samples = print_sample_captions(
                    raw_model, val_loader, device, round(max_caption_length * 1.1),
                    prompt_table=prompt_table)
                t1 = time.time()
                print(f"[rank0] print_sample_captions done in {t1 - t0:.1f}s", flush=True)

                # Val loss exists every epoch, so a new best can land on an epoch
                # that is not a periodic checkpoint -- the state_dict is built if
                # either wants it, and shared when both do.
                is_periodic = (epoch % checkpoint_every == 0) or (epoch == epochs)
                is_best = val_loss < best_val
                if is_periodic or is_best:
                    t2 = time.time()
                    print(f"[rank0] starting state_dict + save @ {t2:.1f}", flush=True)
                    # Persist the full visual pipeline (+ LoRA adapters) regardless of what is
                    # frozen this stage, so weights trained early and frozen later still carry
                    # forward. The frozen base LLM is excluded — it is reloaded each stage.
                    state = {k: v for k, v in raw_model.state_dict().items()
                             if not k.startswith("decoder.") or "lora_" in k}
                    if is_periodic:
                        torch.save(state, Path(save_dir) / f"{save_name[:-3]}_{epoch}.pt")
                        marker = "  *"
                    # best_ckpt now actually gets written, so the final test pass
                    # has something to load.
                    if is_best:
                        torch.save(state, Path(save_dir) / save_name)
                        marker += "  BEST"
                    t3 = time.time()
                    print(f"[rank0] checkpoint save done in {t3 - t2:.1f}s", flush=True)
                print(f"Epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}{marker}", flush=True)

            except Exception as e:
                print(f"[rank0] FAILED in post-validation block: {e}", flush=True)
                traceback.print_exc()
                ok = 0

            # OUTSIDE the try above, on purpose. That block's except sets ok=0, which
            # is all-reduced below and aborts every rank -- correct for a failed save,
            # catastrophic for a webhook 500 at hour 30. notify.send() also swallows
            # its own errors, so this is belt and braces on a 36 h run.
            # `best_val` is still the PREVIOUS best here (it updates after the barrier),
            # so report min(best_val, val_loss) to avoid claiming a stale best on the
            # epoch that just improved.
            notify.epoch_report(
                stage=save_name[:-3], epoch=epoch, epochs=epochs,
                train_loss=train_loss, val_loss=val_loss,
                best_val=min(best_val, val_loss), marker=marker,
                minutes=(time.time() - t_epoch) / 60.0, samples=samples,
                # `stale` is not updated until after the barrier below, so read it
                # through the SAME predicate the early-stop block uses -- otherwise
                # the message reports last epoch's count and reads as one epoch of
                # slack more than there is.
                stale=(0 if val_loss < best_val else stale + 1), patience=patience)

        if distributed:
            ok_t = torch.tensor(ok, device=device)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            if ok_t.item() == 0:
                raise RuntimeError("Rank 0 failed in post-validation block — aborting all ranks. See rank 0 log above.")
            dist.barrier()

        # Early stopping on val loss. run_epoch all-reduces it, so every rank
        # holds the same number and the stop decision is identical without a
        # broadcast. Must mirror the `is_best` predicate above exactly -- the
        # save happens on rank 0, this update happens everywhere, and both read
        # `best_val` before it moves.
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            best_ckpt = os.path.join(save_dir, save_name)
        else:
            stale += 1
            if patience and stale >= patience:
                if is_main:
                    print(f"Early stop: {stale} epochs without improvement "
                          f"on val loss (best {best_val:.4f}).", flush=True)
                break

    if distributed:
        dist.barrier()

    # No test set needed for Step2 of curriculum, why waste resoruces.    
    """
    raw_model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=False)
    la = run_epoch(
        raw_model, raw_model, val_loader, None, device,
        is_train=False, desc="Test", distributed=distributed,
    )

    if is_main:
        print(f"\nTest loss: {test_loss:.4f}")
    """

    cleanup(world_size)


if __name__ == "__main__":
    # Launch with: torchrun --standalone --nproc_per_node=N -m braindiff.training.train_single ...
    # batch_size is per-GPU.
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="process_mrrate/data/brain_tumor_data.csv")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_caption_length", type=int, default=610)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--save_name", default="best_model_single.pt")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_vision_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vision_lora_r", type=int, default=64)
    p.add_argument("--vision_lora_alpha", type=int, default=128)
    p.add_argument("--vision_lora_dropout", type=float, default=0.05)
    p.add_argument("--num_queries", type=int, default=64)
    p.add_argument("--include_delta", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--contrastive_weight", type=float, default=0.0)
    p.add_argument("--content_weight", type=float, default=1.0)
    p.add_argument("--checkpoint_every", type=int, default=5)
    args = p.parse_args()
    main(
        args.csv, args.epochs, args.batch_size, args.lr,
        args.num_workers, args.max_caption_length, args.save_dir, args.save_name, args.seed,
        use_lora=args.use_lora,
        use_vision_lora=args.use_vision_lora,
        vision_lora_r=args.vision_lora_r,
        vision_lora_alpha=args.vision_lora_alpha,
        vision_lora_dropout=args.vision_lora_dropout,
        num_queries=args.num_queries,
        include_delta=args.include_delta,
        checkpoint=args.checkpoint,
        contrastive_weight=args.contrastive_weight,
        content_weight=args.content_weight,
        checkpoint_every=args.checkpoint_every,
    )
