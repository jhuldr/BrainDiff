import argparse
import glob
import os
import re
import sys
import time
import traceback
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from braindiff.models.paths import decoder_dir
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from braindiff.models.captioner import DeltaDiffCaptioner_Qwen3
from braindiff.models.prompts import PromptTable
from braindiff.training.freeze import apply_trainable
from braindiff.training.losses import make_swap_perm
from braindiff.training.val_mask import build_content_masks
from braindiff.training.checkpoint import load_stage_checkpoint
from braindiff.training import notify
from braindiff.data.dual import *



# NeuroVFM grid: 193x229x193 @1mm -> 1x1x4mm -> 12x14x12 = 2016 tokens.
IMG_SIZE = (48, 224, 192)
# Decoder resolved through the HF cache (models/paths.py): downloaded once, never again.


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
    """Shuffle and split CSV ~80/10/10 into train/val/test, grouped by patient_uid so the same
patient's studies never span splits. No rows are dropped.

Patients differ in row count, so splitting by patient count drifts the row proportions.
Instead, walk the shuffled patients once and greedily drop each into whichever split is
furthest below its target row share -- groups stay intact, proportions stay near 80/10/10.
    """
    df = pd.read_csv(csv_path)
    counts = df["patient_uid"].value_counts()
    patients = pd.Series(counts.index).sample(frac=1, random_state=seed).reset_index(drop=True)

    total = len(df)
    targets = {"train": 0.8 * total, "val": 0.1 * total, "test": 0.1 * total}
    current = {"train": 0, "val": 0, "test": 0}
    assignment = {"train": set(), "val": set(), "test": set()}

    for pid in patients:
        split = max(targets, key=lambda s: targets[s] - current[s])
        assignment[split].add(pid)
        current[split] += counts[pid]

    return (
        df[df["patient_uid"].isin(assignment["train"])].reset_index(drop=True),
        df[df["patient_uid"].isin(assignment["val"])].reset_index(drop=True),
        df[df["patient_uid"].isin(assignment["test"])].reset_index(drop=True),
    )


def resolve_split_csvs(csv_path, seed, rank, distributed):
    """Return {split: csv path or None}. Accepts a DIRECTORY or a single CSV.

Directory (preferred): pre-built splits from process_mrrate/stage4_split.py, which pins
studies to train and guarantees no patient spans two splits. Files are matched by suffix
(`*train.csv`, `*val.csv`, `*test.csv`), so any prefix works.

Single CSV (legacy): falls back to split_csv(). Kept so old configs run, but it re-splits
on every launch and cannot honour pinned studies.

Test is optional, and absent means either no `*test.csv` or one with zero rows --
stage4_split.py writes all three files even with --test 0, so an existence check alone
would hand back an empty loader and the test pass would divide by zero.
    """
    if not os.path.isdir(csv_path):
        tmp_dir = "/tmp/braindiff_splits_dual"
        if rank == 0:
            train_df, val_df, test_df = split_csv(csv_path, seed)
            os.makedirs(tmp_dir, exist_ok=True)
            train_df.to_csv(os.path.join(tmp_dir, "train.csv"), index=False)
            val_df.to_csv(os.path.join(tmp_dir, "val.csv"), index=False)
            test_df.to_csv(os.path.join(tmp_dir, "test.csv"), index=False)
        if distributed:
            dist.barrier()
        return {s: os.path.join(tmp_dir, f"{s}.csv") for s in ("train", "val", "test")}

    found = {}
    for split in ("train", "val", "test"):
        hits = sorted(glob.glob(os.path.join(csv_path, f"*{split}.csv")))
        if len(hits) > 1:
            raise SystemExit(f"{csv_path} has {len(hits)} files matching *{split}.csv "
                             f"({[os.path.basename(h) for h in hits]}); expected one.")
        found[split] = hits[0] if hits else None

    for required in ("train", "val"):
        if found[required] is None:
            raise SystemExit(f"{csv_path} has no *{required}.csv. A split directory must "
                             f"provide train and val; only test is optional.")
    if found["test"] is not None and len(pd.read_csv(found["test"])) == 0:
        found["test"] = None
    return found


def make_loaders(csv_path, image_csv, batch_size, num_workers, max_caption_length, seed,
                 distributed: bool = False, content_weight: float = 1.0):
    """Returns (train, val, test). `test` is None when the split has no test set."""
    tokenizer = AutoTokenizer.from_pretrained(decoder_dir())
    rank = int(os.environ.get("RANK", 0))

    paths = resolve_split_csvs(csv_path, seed, rank, distributed)
    if rank == 0:
        src = "directory" if os.path.isdir(csv_path) else "single CSV, split in-process"
        print(f"[splits] {src}: " + ", ".join(
            f"{s}={os.path.basename(p) if p else 'NONE'}" for s, p in paths.items()), flush=True)

    loaders = {}
    for name, is_train in (("train", True), ("val", False), ("test", False)):
        if paths[name] is None:
            loaders[name] = None
            continue
        loaders[name] = get_diff_caption_dataloader(
            csv_file=paths[name],
            image_csv = image_csv,
            # BOTH splits are weighted, as at S2: val CE is the selection
            # criterion, so it has to be the same quantity training minimises.
            # An unweighted val CE would select the checkpoint that best recites
            # the normal template -- the exact failure content_weight exists to
            # fix. Safe on val because reorder_report_sections is train-only, so
            # val token positions are fixed for the whole run.
            content_weight=content_weight,
            img_size=IMG_SIZE,
            batch_size=batch_size,
            num_workers=num_workers,
            tokenizer=tokenizer,
            max_caption_length=max_caption_length,
            is_train=is_train,
            distributed=distributed
        )

    return loaders["train"], loaders["val"], loaders["test"]


def log_delta_stats(model, batch, device):
    toks_ref = batch["tokens_ref"][:4].to(device)
    crds_ref = batch["coords_ref"][:4].to(device)
    toks_main = batch["tokens_main"][:4].to(device)
    crds_main = batch["coords_main"][:4].to(device)
    pres_ref = batch["present_ref"][:4].to(device)
    pres_main = batch["present_main"][:4].to(device)
    with torch.no_grad():
        # encode_multimodal keeps the modality axis now; the DiffEncoder is
        # defined on the flat [B, 4*2016, 768] grid, so flatten before calling it.
        f_ref = model.encode_multimodal(toks_ref, crds_ref, pres_ref)
        f_main = model.encode_multimodal(toks_main, crds_main, pres_main)
        b, m, n, d = f_ref.shape
        f_ref, f_main = f_ref.reshape(b, m * n, d), f_main.reshape(b, m * n, d)
        ref_norm   = f_ref.norm(dim=-1).mean().item()
        main_norm  = f_main.norm(dim=-1).mean().item()
        if model.include_delta:
            delta, *_ = model.diff_encoder(f_ref, f_main)
            delta_norm = delta.norm(dim=-1).mean().item()
            ratio      = delta_norm / max(ref_norm, 1e-8)
            print(f"  [delta] ref={ref_norm:.3f}  main={main_norm:.3f}  delta={delta_norm:.3f}  ratio={ratio:.3f}")
        else:
            print(f"  [delta] ref={ref_norm:.3f}  main={main_norm:.3f}")


def print_sample_captions(model, val_loader, device, max_output_length, n=2,
                          prompt_table=None):
    """Mirror of train_single.print_sample_captions, for the dual (ref+main) input.

`prompt_table` is required in practice: neither dual.py nor single.py emits `prompt_ids`,
so generate_caption_batch's assert is the only guard between a missing table and a silent
wrong-prompt generation.
    """
    model.eval()
    batch = next(iter(val_loader))
    toks_ref = batch["tokens_ref"][:n].to(device)
    crds_ref = batch["coords_ref"][:n].to(device)
    pres_ref = batch["present_ref"][:n].to(device)
    toks_main = batch["tokens_main"][:n].to(device)
    crds_main = batch["coords_main"][:n].to(device)
    pres_main = batch["present_main"][:n].to(device)
    tokenizer = model.decoder.tokenizer

    print("  [samples]")
    with torch.no_grad():
        caps = model.generate_caption_batch(
            tokens_main=toks_main, coords_main=crds_main, present_main=pres_main,
            tokens_ref=toks_ref, coords_ref=crds_ref, present_ref=pres_ref,
            prompt_table=prompt_table, max_new_tokens=max_output_length,
            # 1.0 is not a default to drift from -- at S2 a penalty of 1.5 cost
            # more than half of rg_er by suppressing the clinical vocabulary a
            # report must repeat. No n-gram blocking: S4 has never generated with
            # it, and these samples should read like what the offline eval scores.
            repetition_penalty=1.0)
    # Returned as well as printed so the epoch notifier ships the same pairs
    # without generating a second time -- generation is the expensive part.
    pairs = []
    for i, caption in enumerate(caps):
        gt_ids = batch["input_ids"][i]
        gt = tokenizer.decode(gt_ids[gt_ids != -100], skip_special_tokens=True)
        print(f"    [{i}] GT:  {gt}")
        print(f"    [{i}] GEN: {caption}")
        pairs.append((gt, caption))
    return pairs


def module_grad_norm(module):
    """L2 norm over all parameter grads in a module (0.0 if none have grads)."""
    grads = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack(grads)).item()


def log_grad_norms(raw_model, path):
    """Append projection/encoder/LM grad norms to a txt file. Call after backward()."""
    proj = module_grad_norm(raw_model.connector)
    enc = module_grad_norm(raw_model.vision_encoder)
    lm = module_grad_norm(raw_model.decoder)
    with open(path, "a") as f:
        f.write(f"projection={proj:.6e}  encoder={enc:.6e}  LM={lm:.6e}\n")




def run_epoch(model, raw_model, loader, optimizer, device, is_train, desc,
              is_main=True, distributed=False, contrastive_weight=0.0,
              cf_weight=0.0, change_weight=0.0, change_alpha=None, scheduler=None,
              grad_clip=1.0, content_masks=None, prompt_table=None, grad_accum=1):
    model.train(is_train)
    total_loss = 0.0
    n_steps = len(loader)
    total_content = 0.0
    content_batches = 0
    # Change-head accuracy is the cheap per-epoch read on whether the vision path
    # works at all -- compare against the 0.596 prior-text-only F1 floor.
    change_correct = change_total = 0
    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_main)

    # Use raw_model for validation to avoid DDP sync
    forward_model = model if is_train else raw_model

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            # Accumulation boundary: the micro-step where gradients are
            # all-reduced and the optimiser actually steps. See train_single.py
            # for why the contrastive term is gated to it (cost, not correctness).
            is_boundary = ((step + 1) % grad_accum == 0) or (step + 1 == n_steps)
            toks_ref = batch["tokens_ref"].to(device)
            crds_ref = batch["coords_ref"].to(device)
            toks_main = batch["tokens_main"].to(device)
            crds_main = batch["coords_main"].to(device)
            pres_ref = batch["present_ref"].to(device)
            pres_main = batch["present_main"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            sent_ids = batch["sentence_input_ids"].to(device)
            sent_attn = batch["sentence_attn"].to(device)
            sent_mask = batch["sentence_mask"].to(device)
            change_labels = batch["change_label"].to(device)
            # Precomputed per-sample mask over clinical tokens; indexed by dataset
            # position so it survives sharding and the fixed val ordering.
            content_mask = (content_masks[batch["sample_idx"]].to(device)
                            if content_masks is not None else None)

            # Only build the counterfactual pass when it is actually weighted --
            # it costs a second decoder forward.
            swap_perm = has_partner = None
            if cf_weight > 0:
                swap_perm, has_partner = make_swap_perm(change_labels)

            lm_loss, contrastive_loss, cf_loss, change_logits, content_loss = forward_model(
                tokens_main=toks_main,
                coords_main=crds_main,
                present_main=pres_main,
                tokens_ref=toks_ref,
                coords_ref=crds_ref,
                present_ref=pres_ref,
                prompt_table=prompt_table,
                labels=input_ids,
                attention_mask=attn_mask,
                sentence_input_ids=sent_ids if (is_boundary or not is_train) else None,
                sentence_attn=sent_attn,
                sentence_mask=sent_mask,
                swap_perm=swap_perm,
                swap_valid=has_partner,
                content_mask=content_mask,
                # Per-token CE weights from the dataloader. All-ones when
                # content_weight == 1.0, so this is an exact no-op by default.
                token_weights=batch["token_weights"].to(device),
            )
            loss = lm_loss
            if contrastive_loss is not None:
                loss = loss + contrastive_weight * grad_accum * contrastive_loss
            if cf_weight > 0 and cf_loss is not None:
                loss = loss + cf_weight * cf_loss
            if change_logits is not None:
                change_correct += (change_logits.argmax(-1) == change_labels).sum().item()
                change_total += change_labels.numel()

            if is_train:
                scaled = loss / grad_accum
                if distributed and not is_boundary:
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
            if content_loss is not None:
                total_content += content_loss.item()
                content_batches += 1
            con = contrastive_loss.item() if contrastive_loss is not None else 0.0
            cf = cf_loss.item() if cf_loss is not None else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", lm=f"{lm_loss.item():.4f}",
                             con=f"{con:.4f}", cf=f"{cf:.4f}")

    mean_loss = total_loss / len(loader)
    mean_content = total_content / content_batches if content_batches else float("nan")
    if is_main and change_total:
        print(f"  [{desc}] change-head acc={change_correct / change_total:.4f}", flush=True)

    # Average the per-rank means across ranks for an accurate figure.
    if distributed:
        t = torch.tensor([mean_loss, mean_content], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean_loss, mean_content = t[0].item(), t[1].item()

    return mean_loss, mean_content


def main(
    csv: str,
    image_csv: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 4,
    max_caption_length: int = 512,   # MEASURED: covers 100% of S4 reports
                                     # (max 456 Qwen3 tokens; old 300 cut 1.95%)
    grad_accum: int = 1,
    # Was referenced at the model construction below but never declared -- S4 died
    # with NameError before its first step. Default mirrors train_single.py, since
    # S4 CONTINUES S2's adapter and a mismatch drops it on the shape filter.
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    # >1 downweights normality/absence sentences in the captioning CE, relative
    # to clinical content. S4 needs this more than S2 did: 52% of pairs are
    # Stable, so reciting the normal template is the loss-minimising strategy.
    content_weight: float = 1.0,
    trainable: tuple = ("connector.delta", "diff_encoder", "embeddings",
                        "contrastive", "decoder_lora"),
    # Selection: val loss, every epoch, as at S2. patience counts epochs.
    patience: int = 15,
    save_dir: str = "checkpoints",
    save_name:str = "best_model_dual.pt",
    seed: int = 10,
    use_lora: bool = True,
    use_vision_lora: bool = True,
    vision_lora_r: int = 64,
    vision_lora_alpha: int = 128,
    vision_lora_dropout: float = 0.05,
    # LoRA is BUILT whenever use_vision_lora is set (it decides the
    # checkpoint key names); this controls whether it TRAINS.
    vision_lora_trainable: bool = True,
    # Perceiver latents per block -- the visual token count the decoder sees.
    # Three blocks at S4 (ref/main/delta), so the prefix is 3*num_queries + 2
    # boi/eoi tokens, spent against MedGemma's 1024 sliding window.
    num_queries: int = 64,
    include_delta: bool = False,
    checkpoint: str = None,
    diff_checkpoint: str = None,
    contrastive_weight: float = 0.0,
    cf_weight: float = 0.0,
    counterfactual_margin: float = 0.5,
    change_weight: float = 0.0,
    checkpoint_every: int = 5
):
    rank, local_rank, world_size = ddp_setup()
    distributed = world_size > 1
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Device: {device}  world_size: {world_size}")

    # All splits are sharded across ranks; val/test loss is all-reduced.
    train_loader, val_loader, test_loader = make_loaders(
        csv, image_csv, batch_size, num_workers, max_caption_length, seed,
        distributed=distributed, content_weight=content_weight,
    )
    if is_main and content_weight != 1.0:
        print(f"Content weighting ON: boilerplate x1.0, clinical content "
              f"x{content_weight} (renormalised to mean 1). Val loss is this "
              f"same weighted CE, so it is NOT comparable to runs at a "
              f"different content_weight -- rank on rg_er instead.", flush=True)
    if is_main:
        print(f"Split sizes — train: {len(train_loader.dataset)}  "
              f"val: {len(val_loader.dataset)}  "
              f"test: {len(test_loader.dataset) if test_loader is not None else 'none'}")

    # Stagger the 30 GB LLM load across ranks: 4 simultaneous reads of a
    # 30 GB shard set saturates the filesystem long before host RAM.
    for _r in range(world_size if distributed else 1):
        if not distributed or rank == _r:
            model = DeltaDiffCaptioner_Qwen3(
            single_timepoint=False,
            use_vision_lora=use_vision_lora,
            vision_lora_r=vision_lora_r,
            vision_lora_alpha=vision_lora_alpha,
            vision_lora_dropout=vision_lora_dropout,
            num_queries=num_queries,
            include_delta=include_delta,
            # Read from the stage config. BrainDiff hardcoded r=32/alpha=64 here,
            # silently overriding the YAML -- and S4 now INHERITS S2's adapter, so a
            # rank/alpha mismatch would drop it on checkpoint.py's shape filter.
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_lora=use_lora,
            pretrained_connector=(checkpoint is None),
            max_caption_length=max_caption_length,
            num_change_classes=len(CHANGE_CLASSES) if change_weight > 0 else 0,
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

    # Load the S3-pretrained delta path. S3 now trains TWO modules -- the
    # DiffEncoder and Perceiver_delta -- and writes them under the captioner's own
    # key names (diff_encoder.*, connector.delta.*), so they load straight onto the
    # model with no remapping.
    if diff_checkpoint:
        diff_path = os.path.join(save_dir, diff_checkpoint)
        diff_state = torch.load(diff_path, map_location=device)
        # A pre-port S3 file holds bare DiffEncoder keys and a V-JEPA-width module.
        # Say so rather than silently keeping 0 tensors.
        if diff_state and not any(k.startswith(("diff_encoder.", "connector.delta."))
                                  for k in diff_state):
            raise ValueError(
                f"{diff_path} has bare keys ({list(diff_state)[:3]}...), i.e. it "
                f"predates the NeuroVFM S3 port. Re-run stage nv_stage3_deltatune.pt."
            )
        load_stage_checkpoint(model, diff_state, label=diff_path, is_main=is_main,
                              strict_groups=("diff_encoder", "connector.delta"))


    # Declarative freezing -- the stage's `trainable` list is authoritative.
    # Runs AFTER the checkpoint load so nothing a load touched stays live.
    apply_trainable(model, trainable, is_main=is_main)

    prompt_table = PromptTable(model.decoder.tokenizer,
                               single_timepoint=False,
                               include_delta=include_delta)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if distributed else model

    # Decay only on ndim>1 params -- weight decay on norms/biases/embeddings just
    # shrinks them toward zero without regularizing anything. Mirrors
    # trainer/DifferenceModel/train.py:build_optimizer, which is the one stage
    # that already had a real optimizer config.
    trainable = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.ndim > 1]
    no_decay = [p for p in trainable if p.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )
    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    # Inverse-frequency weights for the change head, from the training split's own
    # class counts -- 52% of S4 is Stable, so an unweighted head predicts it always.
    change_alpha = None
    if change_weight > 0:
        labels = torch.tensor([CHANGE_CLASS_TO_IDX[it["classification"]]
                               for it in train_loader.dataset.items])
        counts = torch.bincount(labels, minlength=len(CHANGE_CLASSES)).float().clamp(min=1)
        change_alpha = (counts.sum() / (len(CHANGE_CLASSES) * counts)).to(device)
        if is_main:
            print(f"Change-head class weights: {change_alpha.tolist()}")

    # Clinical-token mask for the val metric. Validation captions are not
    # augmented, so token positions are fixed and this is built once.
    tokenizer = AutoTokenizer.from_pretrained(decoder_dir())
    content_masks = build_content_masks(
        [it["caption"] for it in val_loader.dataset.items],
        tokenizer, max_caption_length,
    )
    if is_main:
        frac = content_masks.float().mean().item()
        print(f"Content-token mask: {frac:.3f} of caption positions", flush=True)

    # Selection is two-stage, exactly as at S2: val loss drives early stopping and
    # keeps a shortlist, radgraph_score.py over the every-5-epoch checkpoints makes
    # the final call. Lower is better, so this starts at +inf rather than -1.
    best_val = float("inf")
    best_ckpt = None
    stale = 0
    
    #grad_log_path = os.path.join(save_dir, "grad_norms.txt") if is_main else None
    grad_log_path = None  # Disable grad norm logging for now

    first_batch = next(iter(train_loader))

    # Same placement rationale as S2/S3: both checkpoint loads, the freeze audit
    # and all three loaders are already done, so a "started" message means this
    # run really is training and not about to die on a bad checkpoint path.
    if is_main:
        n_par = sum(p.numel() for p in trainable)
        notify.send(
            f"*{save_name[:-3]}* started\n"
            f"{epochs} epochs x {len(train_loader)} steps   "
            f"bs {batch_size}/rank x {world_size} = {batch_size * world_size}\n"
            f"lr {lr:g}   {n_par/1e6:.1f}M trainable   cf {cf_weight:g}   "
            f"con {contrastive_weight:g}   change {change_weight:g}\n"
            f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val pairs\n"
            f"select: val loss every epoch, patience {patience}"
            + (f"\nwarm start: {checkpoint}" if checkpoint else "\nno warm start")
            + (f" + delta {diff_checkpoint}" if diff_checkpoint else ""))

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        if distributed:
            train_loader.batch_sampler.set_epoch(epoch)
        if is_main:
            log_delta_stats(raw_model, first_batch, device)
        
        train_loss, _ = run_epoch(
            model, raw_model, train_loader, optimizer, device,
            is_train=True, desc=f"Epoch {epoch}/{epochs} [train]",
            is_main=is_main, distributed=distributed, contrastive_weight=contrastive_weight,
            cf_weight=cf_weight, change_weight=change_weight,
            prompt_table=prompt_table, grad_accum=grad_accum, change_alpha=change_alpha,
            scheduler=scheduler,
        )

        # Selection criterion: the LM CE, and nothing else -- same choice as S2.
        # Neither `contrastive_weight` nor `change_weight` is passed, so both
        # default to 0 and the auxiliary terms are absent from val. The contrastive
        # term sits near chance, so as a printed diagnostic it is harmless but as a
        # selection criterion it is ~0.5 of noise on a quantity whose meaningful
        # epoch-to-epoch spread is far smaller. The counterfactual is omitted for
        # the same reason plus cost: it doubles the decoder forward for a number
        # that only diagnoses training, and it is already reported from train.
        val_loss, val_content_loss = run_epoch(
            raw_model, raw_model, val_loader, None, device,
            is_train=False, desc=f"Epoch {epoch}/{epochs} [val]",
            is_main=is_main, distributed=distributed,
            content_masks=content_masks,
            # Required -- forward() cannot build the prefix without prompts and
            # asserts. Same omission that made every S2 val epoch raise.
            prompt_table=prompt_table,
        )
        if is_main:
            print(f"  [val] content-token CE={val_content_loss:.4f}  (diagnostic, not selected on)", flush=True)

        # No in-loop generation for SELECTION: rg_er-based ranking is a post-hoc
        # pass (evaluate_checkpoints.py + radgraph_score.py) over the periodic
        # checkpoints. The sampling below generates 2 reports purely to read.
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
                # that is not a periodic checkpoint -- build the state_dict if
                # either wants it, and share it when both do.
                is_periodic = (epoch % checkpoint_every == 0) or (epoch == epochs)
                is_best = val_loss < best_val
                if is_periodic or is_best:
                    t2 = time.time()
                    print(f"[rank0] starting state_dict + save @ {t2:.1f}", flush=True)
                    state = {k: v for k, v in raw_model.state_dict().items()
                             if not k.startswith("decoder.") or "lora_" in k}
                    if is_periodic:
                        torch.save(state, Path(save_dir) / f"{save_name[:-3]}_{epoch}.pt")
                        marker = "  *"
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

            # OUTSIDE the try above, on purpose -- that block's except sets ok=0,
            # which is all-reduced below and aborts every rank. Correct for a
            # failed save, catastrophic for a webhook 500 at hour 20.
            #
            # `best_val` is still the PREVIOUS best here (it updates after the
            # barrier), so report min(best_val, val_loss) to avoid claiming a stale
            # best on the epoch that just improved, and read `stale` through the
            # SAME predicate the early-stop block uses -- otherwise the message
            # reports last epoch's count and reads as one epoch of slack more than
            # there is.
            notify.epoch_report(
                stage=save_name[:-3], epoch=epoch, epochs=epochs,
                train_loss=train_loss, val_loss=val_loss,
                best_val=min(best_val, val_loss), marker=marker,
                minutes=(time.time() - t_epoch) / 60.0, samples=samples,
                stale=(0 if val_loss < best_val else stale + 1), patience=patience,
                extra=[f"content-token CE {val_content_loss:.4f}   (diagnostic)"])

        if distributed:
            ok_t = torch.tensor(ok, device=device)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            if ok_t.item() == 0:
                raise RuntimeError("Rank 0 failed in post-validation block — aborting all ranks. See rank 0 log above.")
            dist.barrier()

        # Early stopping on val loss. run_epoch all-reduces it, so every rank holds
        # the same number and the stop decision is identical without a broadcast.
        # Must mirror the `is_best` predicate above exactly -- the save happens on
        # rank 0, this update happens everywhere, and both read `best_val` before
        # it moves.
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

    # Evaluate best checkpoint on test set; runs on every rank (test_loader is
    # sharded), loss all-reduced, only rank 0 prints.
    if distributed:
        dist.barrier()

    # No test split is a legitimate configuration -- a splits directory built with
    # --test 0, or one that never wrote a test file. Checked BEFORE hunting for a
    # checkpoint so the log does not announce a test pass it is about to skip.
    if test_loader is None:
        if is_main:
            print("\nNo test split in this configuration — skipping the test pass.", flush=True)
        best_ckpt = None
    # best_ckpt is save_name itself, which IS written now that val loss selects
    # every epoch -- but it is still absent if no epoch ever improved (a crash
    # before epoch 1 completes). Fall back to the newest periodic file, and skip
    # the pass rather than dying after a full run. Final checkpoint choice is made
    # offline by radgraph_score.py regardless.
    elif not os.path.exists(best_ckpt):
        stem = os.path.splitext(os.path.basename(save_name))[0]
        periodic = sorted(
            glob.glob(os.path.join(save_dir, f"{stem}_*.pt")),
            key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)),
        )
        best_ckpt = periodic[-1] if periodic else None
        if is_main:
            print(f"\nNo '{save_name}' on disk; "
                  + (f"testing newest periodic checkpoint {os.path.basename(best_ckpt)}"
                     if best_ckpt else "no periodic checkpoints found, skipping test pass"))

    if best_ckpt is not None:
        raw_model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=False)
        test_loss, test_content = run_epoch(
            raw_model, raw_model, test_loader, None, device,
            is_train=False, desc="Test", distributed=distributed,
            prompt_table=prompt_table,   # required -- see the val call above
        )
        if is_main:
            print(f"\nTest loss: {test_loss:.4f}   content-token CE: {test_content:.4f}")

    cleanup(world_size)


if __name__ == "__main__":
    # Launch with: torchrun --standalone --nproc_per_node=N -m braindiff.training.train_dual ...
    # batch_size is per-GPU.
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="process_mrrate/data/longitudinal_data_revised.csv")
    p.add_argument("--image_csv", default="/home/data/MR-RATE-longitudinal")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_caption_length", type=int, default=300)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_vision_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vision_lora_r", type=int, default=64)
    p.add_argument("--vision_lora_alpha", type=int, default=128)
    p.add_argument("--vision_lora_dropout", type=float, default=0.05)
    p.add_argument("--num_queries", type=int, default=64)
    args = p.parse_args()
    main(
        args.csv, args.image_csv, args.epochs, args.batch_size, args.lr,
        args.num_workers, args.max_caption_length, args.save_dir, "dual_training.pt", args.seed,
        use_lora=args.use_lora,
        use_vision_lora=args.use_vision_lora,
        vision_lora_r=args.vision_lora_r,
        vision_lora_alpha=args.vision_lora_alpha,
        vision_lora_dropout=args.vision_lora_dropout,
        num_queries=args.num_queries,
    )
