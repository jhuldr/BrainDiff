import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from braindiff.models.diff_encoder import DiffPretrainModel
from braindiff.data.diff_pairs import get_diff_pair_dataloader, split_pairs
from braindiff.training import notify


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


def build_optimizer(model, lr, weight_decay):
    """AdamW with weight decay only on non-LayerNorm/non-bias trainable params."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):   # LayerNorm/embedding vectors + biases
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


KEYS = ["loss", "recon", "norm", "gate", "antisym", "disc", "compress",
        "disc_acc", "compress_acc", "delta_norm", "gate_mean"]


def run_epoch(model, loader, optimizer, scheduler, device, desc,
              lambdas, is_main=True, distributed=False, is_train=True):
    model.train(is_train)
    l1, l2, l3, l4, l5, l6 = lambdas
    totals = {k: 0.0 for k in KEYS}
    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_main)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in pbar:
            tp = lambda name: (batch[f"tokens_{name}"].to(device),
                               batch[f"coords_{name}"].to(device),
                               batch[f"present_{name}"].to(device))
            is_dup = batch["is_dup"].to(device)

            l_recon, l_norm, l_gate, l_antisym, l_disc, l_compress, stats = model(
                tp("ref"), tp("main"), is_dup)
            loss = (l1 * l_recon + l2 * l_norm + l3 * l_gate
                    + l4 * l_antisym + l5 * l_disc + l6 * l_compress)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            totals["loss"] += loss.item()
            for k, v in (("recon", l_recon), ("norm", l_norm), ("gate", l_gate),
                         ("antisym", l_antisym), ("disc", l_disc),
                         ("compress", l_compress)):
                totals[k] += v.item()
            for k, v in stats.items():
                totals[k] += v.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", disc=f"{l_disc.item():.4f}",
                             cmp=f"{l_compress.item():.4f}")

    n = max(len(loader), 1)
    mean = {k: v / n for k, v in totals.items()}
    # Average the per-rank means across ranks for an accurate figure.
    if distributed:
        t = torch.tensor([mean[k] for k in KEYS], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean = {k: t[i].item() for i, k in enumerate(KEYS)}
    return mean


def val_objective(m, lambdas):
    """Selection metric for S3.

    L_norm and L_gate are omitted BY CONSTRUCTION: dup_fraction is 0 outside
    training, so with no duplicate rows both terms are identically 0 and averaging
    them in would silently dilute the metric with a constant. A duplicate pair is a
    training-time construct.

    Honesty label: unlike temporal_score (Pearson 0.998 vs RadGraph), this has NOT
    been validated as a proxy for downstream S4 report quality -- S3 has no reports,
    so there is nothing to validate it against. It is the only signal available at
    this stage, and it is used for early stopping, not for a final claim.
    """
    l1, _, _, l4, l5, l6 = lambdas
    return (l1 * m["recon"] + l4 * m["antisym"] + l5 * m["disc"]
            + l6 * m["compress"])


def main(
    csv: str,
    image_csv: str,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    num_workers: int = 4,
    bottleneck_dim: int = 128,
    dup_fraction: float = 0.15,
    lambda_recon: float = 1.0,
    lambda_norm: float = 0.01,
    lambda_gate: float = 0.01,
    lambda_antisym: float = 0.01,
    # Retrieval term that stops delta == 0 being a free win. Weighted on par with
    # reconstruction: it is the only pressure keeping the delta informative.
    lambda_disc: float = 1.0,
    # The only term that reaches Perceiver_delta. On par with disc: pretraining
    # that resampler is the reason this stage exists.
    lambda_compress: float = 1.0,
    disc_temperature: float = 0.1,
    disc_negatives: int = 2,
    local_attn_layers: int = 0,
    attn_dim: int = 512,
    # Must match the stage that wrote pretrained_ckpt: LoRA is built
    # (frozen) purely so the encoder keys line up.
    vision_lora_r: int = 16,
    vision_lora_alpha: int = 32,
    num_queries: int = 64,
    warmup_ratio: float = 0.1,
    pretrained_ckpt: str = None,
    val_fraction: float = 0.1,
    seed: int = 10,
    patience: int = 5,
    save_dir: str = "checkpoints",
    save_name: str = "diff_encoder_s3.pt",
    cache_dir: str = "/home/data/BRAIN_DIFF_S3/tmp_nvfm",
):
    # Build the dataloader (and warm its persistent cache) before the process
    # group exists. Warming can take a long time on a cold cache; doing it
    # after ddp_setup() means the other ranks are blocked on NCCL's
    # communicator rendezvous, which times out far sooner than a slow warm-up
    # finishes. get_diff_pair_dataloader synchronizes ranks via a filesystem
    # marker instead of a collective, so this is safe to call pre-DDP.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    # Patient-grouped: every S3 pair is same-patient, so a random split would put
    # the same patient's consecutive studies on both sides.
    train_df, val_df = split_pairs(csv, val_fraction=val_fraction, seed=seed,
                                   image_csv=image_csv)
    train_loader = get_diff_pair_dataloader(
        train_df, image_csv, batch_size, num_workers,
        is_train=True, dup_fraction=dup_fraction, distributed=distributed,
        cache_dir=cache_dir, split_name="train",
    )
    val_loader = get_diff_pair_dataloader(
        val_df, image_csv, batch_size, num_workers,
        is_train=False, dup_fraction=0.0, distributed=distributed,
        cache_dir=cache_dir, split_name="val",
    )

    rank, local_rank, world_size = ddp_setup()
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Device: {device}  world_size: {world_size}")
        print(f"Pairs — train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    model = DiffPretrainModel(
        bottleneck_dim=bottleneck_dim, local_attn_layers=local_attn_layers,
        disc_temperature=disc_temperature, disc_negatives=disc_negatives,
        attn_dim=attn_dim, num_queries=num_queries, device=device,
        vision_lora_r=vision_lora_r, vision_lora_alpha=vision_lora_alpha,
    ).to(device)

    # Warm-start the vision stack from S2, then freeze everything except
    # Perceiver_delta and the DiffEncoder.
    if pretrained_ckpt:
        state = torch.load(pretrained_ckpt, map_location="cpu")
        model.load_pretrained_vision(state, label=pretrained_ckpt, is_main=is_main)
    else:
        model.freeze_vision()
        if is_main:
            print("No --pretrained_ckpt given: connector is randomly initialised.")

    if is_main:
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        print(f"Trainable modules: "
              f"{sorted({n.split('.')[0] + ('.' + n.split('.')[1] if n.startswith('captioner') else '') for n in trainable})}")

    if distributed:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if distributed else model

    optimizer = build_optimizer(model, lr, weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps
    )

    lambdas = (lambda_recon, lambda_norm, lambda_gate, lambda_antisym,
               lambda_disc, lambda_compress)

    # Same placement rationale as S2: everything that can still fail cheaply --
    # cache warm, split, model build, vision warm-start -- is already done, so a
    # "started" message means this run really is training.
    if is_main:
        n_par = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        notify.send(
            f"*{save_name[:-3]}* started\n"
            f"{epochs} epochs x {len(train_loader)} steps   "
            f"bs {batch_size}/rank x {world_size} = {batch_size * world_size}\n"
            f"lr {lr:g}   {n_par/1e6:.1f}M trainable   "
            f"recon {lambda_recon:g}  disc {lambda_disc:g}  compress {lambda_compress:g}\n"
            f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val pairs"
            + (f"\nwarm start: {pretrained_ckpt}" if pretrained_ckpt else "\nno warm start"))

    best_val = float("inf")
    stale = 0
    best_ckpt = os.path.join(save_dir, save_name)

    # Both modules carry into S4: DiffEncoder computes the dense delta and
    # Perceiver_delta compresses it. Saving only the former (as before) would have
    # thrown away the thing this stage now trains. Reconstructor is scaffolding.
    snapshot = raw_model.delta_state_dict

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        m = run_epoch(
            model, train_loader, optimizer, scheduler, device,
            desc=f"Epoch {epoch}/{epochs}", lambdas=lambdas,
            is_main=is_main, distributed=distributed, is_train=True,
        )
        v = run_epoch(
            model, val_loader, None, None, device,
            desc=f"Val {epoch}/{epochs}", lambdas=lambdas,
            is_main=is_main, distributed=distributed, is_train=False,
        )
        val_loss = val_objective(v, lambdas)

        if is_main:
            marker = ""
            if val_loss < best_val:
                torch.save(snapshot(), best_ckpt)
                marker = "  * BEST"
            print(f"Epoch {epoch:03d}  train={m['loss']:.4f}  val={val_loss:.4f}"
                  f"  (recon={v['recon']:.4f} disc={v['disc']:.4f} "
                  f"compress={v['compress']:.4f} antisym={v['antisym']:.4f}){marker}")
            # Collapse watch. Last S3 run: ||delta|| fell 32.4 -> 1.32 and the gate
            # to 0.0069, i.e. the module learned "nothing changed" before it ever
            # saw a report. disc/compress accuracy have hard chance floors.
            print(f"           [val] disc_acc={v['disc_acc']:.3f} "
                  f"(chance {1/(disc_negatives+1):.3f})  "
                  f"compress_acc={v['compress_acc']:.3f}  "
                  f"||delta||={v['delta_norm']:.3f}  gate={v['gate_mean']:.4f}",
                  flush=True)

            # The collapse watch is the whole point of reading this stage remotely:
            # a run whose val loss looks fine while ||delta|| decays toward 0 has
            # learned "nothing changed", and that is only visible in these numbers.
            # No `samples` -- S3 has no reports.
            #
            # `best_val` still holds the PREVIOUS best here (it updates after the
            # is_main block), so report min(...) rather than claiming a stale best
            # on the epoch that just improved, and read `stale` through the same
            # predicate the early-stop block uses.
            notify.epoch_report(
                stage=save_name[:-3], epoch=epoch, epochs=epochs,
                train_loss=m["loss"], val_loss=val_loss,
                best_val=min(best_val, val_loss), marker=marker,
                minutes=(time.time() - t_epoch) / 60.0,
                stale=(0 if val_loss < best_val else stale + 1), patience=patience,
                extra=[f"recon {v['recon']:.4f}   disc {v['disc']:.4f}   "
                       f"compress {v['compress']:.4f}   antisym {v['antisym']:.4f}",
                       f"disc_acc {v['disc_acc']:.3f} (chance {1/(disc_negatives+1):.3f})"
                       f"   compress_acc {v['compress_acc']:.3f}",
                       f"||delta|| {v['delta_norm']:.3f}   gate {v['gate_mean']:.4f}"])

        # Computed on every rank from all-reduced numbers, so the stop decision is
        # identical everywhere without a broadcast.
        if val_loss < best_val:
            best_val, stale = val_loss, 0
        else:
            stale += 1
            if patience and stale >= patience:
                if is_main:
                    print(f"Early stop: {stale} epochs without val improvement "
                          f"(best {best_val:.4f}).", flush=True)
                break
        if distributed:
            dist.barrier()

    # Final snapshot (rank 0 only).
    if is_main:
        torch.save(snapshot(), os.path.join(save_dir, f"final_{save_name}"))

    cleanup(world_size)


if __name__ == "__main__":
    # Launch with: torchrun --standalone --nproc_per_node=N -m braindiff.training.train_delta ...
    # batch_size is per-GPU.
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/home/data/BRAIN_DIFF_S3/main.csv")
    p.add_argument("--image_csv", default="/home/data/BRAIN_DIFF_S3/image.csv")
    p.add_argument("--epochs", type=int, default=50)
    # Measured: DiffEncoder fwd+bwd at 4 modalities is 8.06 GiB @ bs8, 16.01 @ bs16,
    # 31.91 @ bs32 (bf16, one H200). S3 is dataloader-bound regardless, so a larger
    # batch buys optimizer steps, not wall-clock.
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--bottleneck_dim", type=int, default=128)
    p.add_argument("--dup_fraction", type=float, default=0.15)
    p.add_argument("--lambda_recon", type=float, default=1.0)
    p.add_argument("--lambda_norm", type=float, default=0.01)
    p.add_argument("--lambda_gate", type=float, default=0.01)
    p.add_argument("--lambda_antisym", type=float, default=0.01)
    p.add_argument("--lambda_disc", type=float, default=1.0)
    p.add_argument("--lambda_compress", type=float, default=1.0)
    p.add_argument("--disc_temperature", type=float, default=0.1)
    p.add_argument("--disc_negatives", type=int, default=2)
    p.add_argument("--local_attn_layers", type=int, default=0)
    p.add_argument("--attn_dim", type=int, default=512)
    p.add_argument("--vision_lora_r", type=int, default=16)
    p.add_argument("--vision_lora_alpha", type=int, default=32)
    p.add_argument("--num_queries", type=int, default=64)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    # S2 vision weights to warm-start. Everything but Perceiver_delta is frozen.
    p.add_argument("--pretrained_ckpt", default=None)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--save_name", default="diff_encoder_s3.pt")
    p.add_argument("--cache_dir", default="/home/data/BRAIN_DIFF_S3/tmp_nvfm")
    args = p.parse_args()
    # Keyword-only: main()'s signature grows (lambda_disc was inserted mid-list and
    # silently shifted every later positional), and a shifted arg here fails deep
    # inside training rather than at the call.
    main(**vars(args))
