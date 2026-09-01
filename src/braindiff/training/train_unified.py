import argparse
import os
import sys
import traceback
import time
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer
from braindiff.models.paths import decoder_dir

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from braindiff.models.captioner import DeltaDiffCaptioner_Qwen3
from braindiff.models.prompts import PromptTable
from braindiff.training.freeze import apply_trainable
from braindiff.data.unified import *


# NeuroVFM grid: 193x229x193 @1mm -> 1x1x4mm -> 12x14x12 = 2016 tokens.
IMG_SIZE = (48, 224, 192)
# Decoder resolved through the HF cache (models/paths.py): downloaded once, never again.

#ignore contrastive loss for now, no need to add additional complexity to the training loop
CONTRASTIVE_WEIGHT = 0


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


def split_csv(csv_path, seed, min_class_count=MIN_CLASS_COUNT):
    """Shuffle and split 90/10 into train/val, stratified on the sampler's
    (group, sub) key.

    There is no test split: S1 is a curriculum stage, not a reported result --
    nothing is model-selected on a held-out S1 set, and the numbers that get
    reported come from S2/S4 evaluation. The old 10% test partition was scored
    once at the end and otherwise unused, so those rows are worth more as
    training signal.

    Splitting within each bucket rather than over the whole frame keeps both
    objectives -- and every surviving pathology class -- present in all three
    partitions. A plain row shuffle can drop a rare class entirely into val,
    which would score the model on a label it never trained on.

    Rare classes are filtered *before* the split so all three partitions agree
    on the surviving label set by construction."""
    df = pd.read_csv(csv_path)
    df = filter_rare_classes(df, min_class_count)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Must key on the SAME column the sampler does (`pathology_bucket`, the rarest
    # label present). Keying the split on the multi-label caption instead would
    # stratify over 1,835 near-singleton strings while the sampler used ~29
    # buckets, so a class could be well represented in the sampler and absent
    # from val.
    keys = [strat_key(r.task, r.with_lesion, getattr(r, "pathology_bucket", None))
            for r in df.itertuples()]
    df = df.assign(_strat=pd.Series(keys, index=df.index).map(str))

    train, val = [], []
    for _, bucket in df.groupby("_strat", sort=True):
        n = len(bucket)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        # Keep the val slice at the SAME offsets the old 80/10/10 split used, so
        # val loss stays comparable across runs. Only the former test tail moves,
        # and it joins train.
        val.append(bucket.iloc[n_train:n_train + n_val])
        train.append(pd.concat([bucket.iloc[:n_train], bucket.iloc[n_train + n_val:]]))

    return tuple(
        pd.concat(part).drop(columns="_strat").sample(frac=1, random_state=seed).reset_index(drop=True)
        for part in (train, val)
    )


def make_loaders(csv_path, batch_size, num_workers, max_caption_length, seed,
                  max_prompt_length: int = 384,
                  min_class_count: int = MIN_CLASS_COUNT,
                  distributed: bool = False, is_main: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(decoder_dir())

    # Own tmp dir: train_single.py writes /tmp/braindiff_splits_single with a
    # different schema and split ratio.
    tmp_dir = "/tmp/braindiff_splits_unified"
    splits = {"train": True, "val": False}

    # Only rank 0 computes the split and writes the CSVs; other ranks wait so
    # they never read a file mid-write.
    if is_main:
        os.makedirs(tmp_dir, exist_ok=True)
        train_df, val_df = split_csv(csv_path, seed, min_class_count)
        for name, df in zip(splits, (train_df, val_df)):
            df.to_csv(os.path.join(tmp_dir, f"{name}.csv"), index=False)
    if distributed:
        dist.barrier()

    loaders = {}
    for name, is_train in splits.items():
        tmp_csv = os.path.join(tmp_dir, f"{name}.csv")
        loaders[name] = get_diff_caption_dataloader(
            csv_file=tmp_csv,
            img_size=IMG_SIZE,
            batch_size=batch_size,
            num_workers=num_workers,
            tokenizer=tokenizer,
            max_caption_length=max_caption_length,
            max_prompt_length=max_prompt_length,
            is_train=is_train,
            distributed=distributed,
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


def print_sample_captions(model, val_loader, device, max_output_length, max_batches=8):
    """Generate one sample per task, so every objective is visible each time.

    Printing the first n rows of one batch was not enough: the balanced sampler
    splits each batch 50/50 bounding-vs-pathology and the bounding half is spread
    over aref/gcap/caref, so a single batch routinely showed the same two tasks
    and a silently broken one could go unnoticed for a whole run. Scan up to
    `max_batches` for whatever is still missing.
    """
    model.eval()
    picked = {}
    for bi, batch in enumerate(val_loader):
        if len(picked) == len(TASKS) or bi >= max_batches:
            break
        for i, task in enumerate(batch["task"]):
            if task in picked:
                continue
            picked[task] = tuple(batch[k][i] for k in
                                 ("tokens_main", "coords_main", "present_main",
                                  "prompt_ids", "prompt_attn")) + (batch["caption"][i],)

    print("  [samples]")
    with torch.no_grad():
        for task in TASKS:
            if task not in picked:
                continue
            toks, crds, pres, prompt, prompt_attn, gt = picked[task]
            caption = model.generate_caption_batch(
                tokens_main=toks.unsqueeze(0).to(device),
                coords_main=crds.unsqueeze(0).to(device),
                present_main=pres.unsqueeze(0).to(device),
                prompt_ids=prompt.unsqueeze(0).to(device),
                prompt_attn=prompt_attn.unsqueeze(0).to(device),
                repetition_penalty=1.0,
                max_new_tokens=max_output_length,
            )[0]
            # `caption` is the raw GT string for both objectives, so print it
            # directly instead of round-tripping through the tokenizer.
            print(f"    ({task}) GT:  {gt}")
            print(f"    ({task}) GEN: {caption}")

    missing = [t for t in TASKS if t not in picked]
    if missing:
        print(f"    [warn] no val sample for {', '.join(missing)} in {max_batches} batches")


def run_epoch(model, raw_model, loader, optimizer, device, is_train, desc, is_main=True, distributed=False):
    model.train(is_train)
    total_loss = 0.0
    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_main)

    # Use raw_model for validation to avoid DDP sync
    forward_model = model if is_train else raw_model

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in pbar:
            toks_main = batch["tokens_main"].to(device)
            crds_main = batch["coords_main"].to(device)
            pres_main = batch["present_main"].to(device)
            input_ids = batch["input_ids"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_attn = batch["prompt_attn"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            # forward returns 5 values; stage 1 is single-timepoint with the
            # decoder frozen, so only the LM term is used here.
            lm_loss, _contrastive, _cf, _change_logits, _content = forward_model(
                tokens_main=toks_main,
                coords_main=crds_main,
                present_main=pres_main,
                labels=input_ids,
                attention_mask=attn_mask,
                # S1 supplies a PER-ROW prompt (the `prompt` column carries box
                # coords / region names), so it passes ids directly rather than
                # using PromptTable, which keys on modality presence alone.
                prompt_ids=prompt_ids,
                prompt_attn=prompt_attn,
            )
            loss = lm_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = total_loss / len(loader)

    # Only all_reduce during training and validation
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
    max_caption_length: int = 96,    # MEASURED: max S1 caption is 66 Qwen3 tokens
                                     # across all four tasks (aref 25, gcap 20,
                                     # caref 44, pcls 66)
    # Left-padded prompt width. MEASURED worst case is 339 tokens (pcls, which gets
    # PATHOLOGY_PROMPT injected); 384 = +45 margin.
    # Not free headroom -- it is dead sequence on every sample (see dataloaders/unified.py).
    # The loader RAISES rather than truncates if a prompt exceeds it, so too small is a
    # startup crash, not silent loss of the conditioning signal.
    max_prompt_length: int = 384,
    grad_accum: int = 1,
    # Declarative freezing. Authoritative over any use_*/trainable ctor flag.
    trainable: tuple = ("encoder_lora", "connector.scan", "connector.proj",
                        "embeddings"),
    save_dir: str = "checkpoints",
    save_name: str = "best_model_unified.pt",
    seed: int = 10,
    use_lora: bool = True,
    use_vision_lora: bool = True,
    vision_lora_r: int = 64,
    vision_lora_alpha: int = 128,
    vision_lora_dropout: float = 0.05,
    # LoRA is BUILT whenever use_vision_lora is set (it decides the
    # checkpoint key names); this controls whether it TRAINS. S1 has no
    # predecessor, so True is the only value that makes sense here -- but
    # curriculum.py puts it in `common` for every stage, and its absence
    # here made `--stage nv_stage1_unified.pt` a TypeError at launch.
    vision_lora_trainable: bool = True,
    # Perceiver latents per block -- the visual token count the decoder sees.
    # A change here IS a shape change on connector.*.queries, so a later stage
    # that disagrees fails loudly in checkpoint.py rather than silently.
    num_queries: int = 64,
    include_delta: bool = False,
    checkpoint: str = None,
    min_class_count: int = MIN_CLASS_COUNT,
    patience: int = 10,
):
    rank, local_rank, world_size = ddp_setup()
    distributed = world_size > 1
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Device: {device}  world_size: {world_size}")

    # Both loaders are sharded across ranks.
    train_loader, val_loader = make_loaders(
        csv, batch_size, num_workers, max_caption_length, seed,
        max_prompt_length=max_prompt_length,
        min_class_count=min_class_count, distributed=distributed, is_main=is_main,
    )
    if is_main:
        print(f"Split sizes — train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    # Stagger the 30 GB LLM load across ranks: 4 simultaneous reads of a
    # 30 GB shard set saturates the filesystem long before host RAM.
    for _r in range(world_size if distributed else 1):
        if not distributed or rank == _r:
            model = DeltaDiffCaptioner_Qwen3(
            single_timepoint=True,
            use_vision_lora=use_vision_lora,
            vision_lora_r=vision_lora_r,
            vision_lora_alpha=vision_lora_alpha,
            vision_lora_dropout=vision_lora_dropout,
            num_queries=num_queries,
            include_delta=include_delta,
            use_lora=use_lora,
            pretrained_connector=(checkpoint is None),
            max_caption_length=max_caption_length,
            device=device,
            ).to(device)
        if distributed:
            dist.barrier()

    # Warm-start from the previous curriculum stage. Partial overlap is expected:
    # freeze config and LoRA/delta modules differ across stages, so load leniently.
    if checkpoint:
        ckpt_path = os.path.join(save_dir, checkpoint)
        state = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if is_main:
            print(f"Loaded {ckpt_path}: {len(state)} tensors  (missing={len(missing)}  unexpected={len(unexpected)})")


    # Declarative freezing -- the stage's `trainable` list is authoritative.
    # Runs AFTER the checkpoint load so nothing a load touched stays live.
    apply_trainable(model, trainable, is_main=is_main)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if distributed else model

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    best_val_loss = float("inf")
    best_ckpt = os.path.join(save_dir, save_name)
    epochs_no_improve = 0

    first_batch = next(iter(train_loader))

    for epoch in range(1, epochs + 1):
        if distributed:
            train_loader.batch_sampler.set_epoch(epoch)
        if is_main:
            log_feature_stats(raw_model, first_batch, device)

        train_loss = run_epoch(
            model, raw_model, train_loader, optimizer, device,
            is_train=True, desc=f"Epoch {epoch}/{epochs} [train]",
            is_main=is_main, distributed=distributed,
        )

        # Validation runs on every rank (val_loader is sharded via
        # DistributedSampler); the loss is all-reduced so every rank agrees --
        # the improved/patience bookkeeping below is computed from that
        # already-synced value on every rank, so no extra communication is
        # needed to keep the early-stop decision consistent across ranks.
        # Sampling/checkpointing stay rank-0-only to avoid duplicate writes.
        val_loss = run_epoch(
            raw_model, raw_model, val_loader, None, device,
            is_train=False, desc=f"Epoch {epoch}/{epochs} [val]",
            is_main=is_main, distributed=distributed,
        )

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ok = 1
        if is_main:
            try:
                t0 = time.time()
                print(f"[rank0] starting print_sample_captions @ {t0:.1f}", flush=True)
                print_sample_captions(raw_model, val_loader, device, round(max_caption_length * 1.1))
                t1 = time.time()
                print(f"[rank0] print_sample_captions done in {t1 - t0:.1f}s", flush=True)

                marker = ""
                if improved:
                    t2 = time.time()
                    print(f"[rank0] starting state_dict + save @ {t2:.1f}", flush=True)
                    # Persist the full visual pipeline (+ LoRA adapters) regardless of what is
                    # frozen this stage, so weights trained early and frozen later still carry
                    # forward. The frozen base LLM is excluded — it is reloaded each stage.
                    state = {k: v for k, v in raw_model.state_dict().items()
                             if not k.startswith("decoder.") or "lora_" in k}
                    torch.save(state, best_ckpt)
                    t3 = time.time()
                    print(f"[rank0] checkpoint save done in {t3 - t2:.1f}s", flush=True)
                    marker = "  *"

                print(f"Epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}{marker}  "
                      f"(no_improve={epochs_no_improve}/{patience})", flush=True)

            except Exception as e:
                print(f"[rank0] FAILED in post-validation block: {e}", flush=True)
                traceback.print_exc()
                ok = 0

        if distributed:
            ok_t = torch.tensor(ok, device=device)
            dist.all_reduce(ok_t, op=dist.ReduceOp.MIN)
            if ok_t.item() == 0:
                raise RuntimeError("Rank 0 failed in post-validation block — aborting all ranks. See rank 0 log above.")
            dist.barrier()

        if epochs_no_improve >= patience:
            if is_main:
                print(f"Early stopping: no val improvement in {patience} epochs "
                      f"(best val={best_val_loss:.4f})", flush=True)
            break

    if distributed:
        dist.barrier()

    # No test pass: the split is 90/10 train/val and those rows now train the
    # model. Selection is on val loss, and S1 is never reported on directly.
    if is_main:
        print(f"\nBest checkpoint: {best_ckpt}")

    cleanup(world_size)


if __name__ == "__main__":
    # Launch with: torchrun --standalone --nproc_per_node=N -m braindiff.training.train_unified ...
    # batch_size is per-GPU.
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="process_mrrate/data/brain_tumor_data.csv")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_caption_length", type=int, default=128)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--save_name", default="best_model_unified.pt")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_vision_lora", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vision_lora_r", type=int, default=64)
    p.add_argument("--vision_lora_alpha", type=int, default=128)
    p.add_argument("--vision_lora_dropout", type=float, default=0.05)
    p.add_argument("--num_queries", type=int, default=64)
    p.add_argument("--include_delta", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--min_class_count", type=int, default=MIN_CLASS_COUNT,
                   help="Drop pathology classes with fewer real examples than this")
    p.add_argument("--patience", type=int, default=15,
                   help="Stop early after this many epochs with no val-loss improvement")
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
        min_class_count=args.min_class_count,
        patience=args.patience,
    )
