"""S3-sup: train the delta encoder to localise change, not just fingerprint pairs.

Derived from train_delta.py. The data path, cache warming, DDP setup and checkpoint contract
are unchanged; what differs is the objective, which adds the three terms documented in
models/change_map_pretrain.py.

Selection is not the old val objective -- a weighted sum of terms the previous run maximised
while its gate stayed at chance (measured CHANGE/SHUFFLED 0.99x with disc_acc 0.994).
Selection here is the gate-tracking loss plus the global change head; the old terms are
still logged.

    torchrun --nproc_per_node=4 -m braindiff.training.curriculum --track neurovfm \
        --stage nv_stage3_deltasup.pt
"""
import os
import sys
import time

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from braindiff.models.change_map_pretrain import ChangeMapPretrainModel
from braindiff.data.diff_pairs import _uid_to_path, subject_of
from braindiff.data.change_map_pairs import get_change_map_pair_dataloader
from braindiff.training import notify
from braindiff.training.checkpoint import load_stage_checkpoint
from braindiff.training.train_delta import build_optimizer, ddp_setup, cleanup

# Three loss terms. `gate_track`/`gate_dup` are the two halves of `localize`,
# logged separately so the nuisance control stays readable; they are stats, not
# separately-weighted losses. `recon_baseline` is kept as the diagnostic that
# justified dropping L_recon.
KEYS = ["loss", "disc", "localize", "global",
        "disc_acc", "delta_norm", "gate_mean",
        "recon_baseline", "global_acc", "gate_track", "gate_dup",
        "gate_stable", "n_global"]

N_CHANGE_CLASSES = 7


def run_epoch(model, loader, optimizer, scheduler, device, desc, lambdas,
              change_alpha=None, is_main=True, distributed=False, is_train=True):
    model.train(is_train)
    (l_disc_w, l_loc_w, l_glob_w) = lambdas
    totals = {k: 0.0 for k in KEYS}
    pbar = tqdm(loader, desc=desc, leave=False, disable=not is_main)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in pbar:
            tp = lambda name: (batch[f"tokens_{name}"].to(device),
                               batch[f"coords_{name}"].to(device),
                               batch[f"present_{name}"].to(device))
            l_disc, l_localize, l_global, stats = model(
                tp("ref"), tp("main"), batch["is_dup"].to(device),
                change_label=batch["change_label"].to(device),
                has_global=batch["has_global"].to(device),
                change_alpha=change_alpha)

            loss = (l_disc_w * l_disc + l_loc_w * l_localize
                    + l_glob_w * l_global)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            totals["loss"] += loss.item()
            for k, v in (("disc", l_disc), ("localize", l_localize),
                         ("global", l_global)):
                totals[k] += v.item()
            for k, v in stats.items():
                totals[k] += float(v)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             loc=f"{l_localize.item():.4f}",
                             glb=f"{l_global.item():.4f}")

    n = max(len(loader), 1)
    mean = {k: v / n for k, v in totals.items()}
    if distributed:
        t = torch.tensor([mean[k] for k in KEYS], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        mean = {k: t[i].item() for i, k in enumerate(KEYS)}
    return mean


def split_change_map_pairs(csv, val_fraction, seed, image_csv):
    """Patient-grouped split across BOTH corpora.

`split_pairs` derives the group from the subject id in the S3 volume path. S4's study UIDs
match no such pattern, so it falls back to one patient per UID -- and since one S4 patient
contributes several pairs under different UIDs, that patient lands on both sides (measured:
1340 pairs straddling). Grouped explicitly here and asserted, not trusted.
    """
    df = pd.read_csv(csv)
    uid_to_path = _uid_to_path(image_csv)
    groups = []
    for _, r in df.iterrows():
        if r.get("source") == "s4" and str(r.get("patient_uid", "")):
            groups.append(f"s4:{r['patient_uid']}")      # authoritative, from the split file
        else:
            groups.append(f"s3:{subject_of(r['UID_1'], uid_to_path)}")
    df = df.assign(_group=groups)

    uniq = pd.Series(sorted(df["_group"].unique()))
    uniq = uniq.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    counts = df["_group"].value_counts()
    target = val_fraction * len(df)
    val_groups, running = set(), 0
    for g in uniq:
        if running >= target:
            break
        val_groups.add(g)
        running += counts[g]

    train = df[~df["_group"].isin(val_groups)].drop(columns="_group").reset_index(drop=True)
    val = df[df["_group"].isin(val_groups)].drop(columns="_group").reset_index(drop=True)

    # Assert, do not assume: recompute membership from the emitted frames.
    tg = {g for g, keep in zip(groups, ~df["_group"].isin(val_groups)) if keep}
    vg = {g for g, keep in zip(groups, df["_group"].isin(val_groups)) if keep}
    if tg & vg:
        raise SystemExit(f"{len(tg & vg)} groups straddle the split")
    print(f"[split] train {len(train)} pairs / {len(tg)} groups   "
          f"val {len(val)} pairs / {len(vg)} groups   (0 straddling)", flush=True)
    return train, val


def val_objective(m, lambdas):
    """Select on what the stage is supposed to learn, not on what it already wins.

The previous S3 selected on recon/antisym/disc/compress and converged while its gate sat at
chance. Those terms are logged but excluded; the criterion is the gate-tracking loss plus
the global change head, both lower-is-better.

NOT validated against downstream S4 report quality -- nothing at S3 can be, since the stage
has no reports. The held-out check is diag_gate_localization.py's CHANGE/SHUFFLED ratio,
run after training.

`disc` is included as the only remaining token-local constraint: excluding it would select
almost entirely on `global` (global moved 1.1256 -> 1.0224 across the last run while
gate_track moved 0.1842 -> 0.1681 non-monotonically), and a checkpoint chosen on a 7-way
head at 0.20-0.28 accuracy is not chosen on localisation.

`localize`'s dup half is structurally 0 here (dup_fraction=0 on val), so on val this
reduces to the tracking half.
    """
    l_disc_w, l_loc_w, l_glob_w = lambdas
    return (l_disc_w * m["disc"] + l_loc_w * m["localize"]
            + l_glob_w * m["global"])


def main(
    csv: str,
    image_csv: str,
    epochs: int = 50,
    batch_size: int = 18,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    num_workers: int = 6,
    bottleneck_dim: int = 128,
    dup_fraction: float = 0.15,
    lambda_recon: float = 1.0,
    lambda_norm: float = 0.01,
    lambda_gate: float = 0.01,
    lambda_antisym: float = 0.01,
    lambda_disc: float = 1.0,
    lambda_compress: float = 1.0,
    lambda_gate_track: float = 1.0,
    lambda_gate_dup: float = 0.5,
    lambda_gate_stable: float = 0.25,
    lambda_global: float = 1.0,
    spliced_disc: bool = True,
    splice_frac: float = 0.25,
    disc_temperature: float = 0.1,
    disc_negatives: int = 2,
    local_attn_layers: int = 0,
    attn_dim: int = 512,
    vision_lora_r: int = 16,
    vision_lora_alpha: int = 32,
    num_queries: int = 64,
    warmup_ratio: float = 0.1,
    pretrained_ckpt: str = None,
    delta_ckpt: str = None,
    val_fraction: float = 0.1,
    seed: int = 10,
    patience: int = 8,
    save_dir: str = "checkpoints",
    save_name: str = "nv_stage3_deltasup.pt",
    cache_dir: str = "/home/data/BRAIN_DIFF_S3/tmp_nvfm",
):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    # split_pairs groups on the real subject id parsed from the volume path. That
    # matters here as much as it did for the original S3: uuid grouping leaked 90%
    # of that split.
    train_df, val_df = split_change_map_pairs(csv, val_fraction, seed, image_csv)
    train_loader = get_change_map_pair_dataloader(
        train_df, image_csv, batch_size, num_workers, is_train=True,
        dup_fraction=dup_fraction, distributed=distributed,
        cache_dir=cache_dir, split_name="train_sup")
    val_loader = get_change_map_pair_dataloader(
        val_df, image_csv, batch_size, num_workers, is_train=False,
        dup_fraction=0.0, distributed=distributed,
        cache_dir=cache_dir, split_name="val_sup")

    rank, local_rank, world_size = ddp_setup()
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Device: {device}  world_size: {world_size}")
        print(f"Pairs — train: {len(train_loader.dataset)}  val: {len(val_loader.dataset)}")

    model = ChangeMapPretrainModel(
        bottleneck_dim=bottleneck_dim, local_attn_layers=local_attn_layers,
        disc_temperature=disc_temperature, disc_negatives=disc_negatives,
        attn_dim=attn_dim, num_queries=num_queries, device=device,
        vision_lora_r=vision_lora_r, vision_lora_alpha=vision_lora_alpha,
        spliced_disc=spliced_disc, splice_frac=splice_frac,
        dup_weight=lambda_gate_dup, stable_weight=lambda_gate_stable,
        num_change_classes=N_CHANGE_CLASSES,
    ).to(device)

    if pretrained_ckpt:
        state = torch.load(pretrained_ckpt, map_location="cpu")
        model.load_pretrained_vision(state, label=pretrained_ckpt, is_main=is_main)
    else:
        model.freeze_vision()
        if is_main:
            print("No --pretrained_ckpt given: connector is randomly initialised.")

    # `change_map` (the ChangeMapEncoder) is a NEW module with no warm-start available
    # (the old delta_ckpt carries connector.delta.* keys the model no longer has). It
    # starts fresh and is trained here from scratch. `delta_ckpt` is accepted but
    # ignored -- there is nothing compatible to load.
    if delta_ckpt and is_main:
        print(f"note: delta_ckpt {delta_ckpt} ignored -- change_map starts fresh "
              f"(no compatible warm-start).", flush=True)

    # Objective is LABEL-FREE: change_alpha is unused, but the trainer still passes it.
    change_alpha = None

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if distributed else model

    optimizer = build_optimizer(model, lr, weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(warmup_ratio * total_steps), total_steps)

    # Three terms. lambda_recon/norm/gate/antisym/compress are still accepted in the
    # signature (curriculum.py forwards them and test_dispatch binds the real
    # signature) but are no longer used -- see models/change_map_pretrain.py for why each
    # was cut. lambda_gate_dup is not here: it is the dup half's weight INSIDE
    # `localize`, passed to the model as `dup_weight`, so it stays a YAML knob.
    lambdas = (lambda_disc, lambda_gate_track, lambda_global)

    if is_main:
        n_par = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        notify.send(
            f"*{save_name[:-3]}* started\n"
            f"{epochs} epochs x {len(train_loader)} steps   "
            f"bs {batch_size}/rank x {world_size} = {batch_size * world_size}\n"
            f"lr {lr:g}   {n_par/1e6:.1f}M trainable\n"
            f"gate_track {lambda_gate_track:g}  gate_dup {lambda_gate_dup:g}  "
            f"global {lambda_global:g}  spliced_disc {spliced_disc}\n"
            f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val pairs"
            + (f"\nwarm start: {pretrained_ckpt}" if pretrained_ckpt else "\nno warm start"))

    best_val = float("inf")
    stale = 0
    best_ckpt = os.path.join(save_dir, save_name)
    snapshot = raw_model.delta_state_dict

    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        m = run_epoch(model, train_loader, optimizer, scheduler, device,
                      desc=f"Epoch {epoch}/{epochs}", lambdas=lambdas,
                      change_alpha=change_alpha, is_main=is_main,
                      distributed=distributed, is_train=True)
        v = run_epoch(model, val_loader, None, None, device,
                      desc=f"Val {epoch}/{epochs}", lambdas=lambdas,
                      change_alpha=change_alpha, is_main=is_main,
                      distributed=distributed, is_train=False)
        val_loss = val_objective(v, lambdas)

        if is_main:
            marker = ""
            if val_loss < best_val:
                torch.save(snapshot(), best_ckpt)
                marker = "  * BEST"
            print(f"Epoch {epoch:03d}  train={m['loss']:.4f}  val={val_loss:.4f}"
                  f"  (track={v['gate_track']:.4f} dup={v['gate_dup']:.4f} stab={v['gate_stable']:.4f} "
                  f"global={v['global']:.4f} acc={v['global_acc']:.3f}){marker}", flush=True)
            # recon vs its trivial baseline, to test whether L_recon is
            # vacuous, i.e. whether B_hat = A already achieves the same MSE. If
            # these two sit on top of each other, the delta contributes nothing
            # to reconstruction and lambda_recon is buying nothing.
            print(f"  [watch] baseline(B=A)={v['recon_baseline']:.6f}"
                  f"   ||delta||={v['delta_norm']:.3f}  gate={v['gate_mean']:.4f}"
                  f"  disc={v['disc']:.4f}  disc_acc={v['disc_acc']:.3f}",
                  flush=True)

            notify.epoch_report(
                stage=save_name[:-3], epoch=epoch, epochs=epochs,
                train_loss=m["loss"], val_loss=val_loss,
                best_val=min(best_val, val_loss), marker=marker,
                minutes=(time.time() - t_epoch) / 60.0, samples=(),
                stale=(0 if val_loss < best_val else stale + 1), patience=patience,
                extra=[f"gate_track {v['gate_track']:.4f}   gate_dup {v['gate_dup']:.4f}   "
                       f"gate_stable {v['gate_stable']:.4f}",
                       f"global CE {v['global']:.4f}   acc {v['global_acc']:.3f} "
                       f"(chance 0.14, majority 0.53)",
                       f"B=A baseline {v['recon_baseline']:.6f}",
                       f"||delta|| {v['delta_norm']:.3f}   gate {v['gate_mean']:.4f}   "
                       f"disc_acc {v['disc_acc']:.3f}"])

        if val_loss < best_val:
            best_val, stale = val_loss, 0
        else:
            stale += 1
            if patience and stale >= patience:
                if is_main:
                    print(f"Early stop: {stale} epochs without improvement "
                          f"(best {best_val:.4f}).", flush=True)
                break

    if is_main:
        notify.send(f"*{save_name[:-3]}* finished — best val {best_val:.4f}\n"
                    f"next: diag_gate_localization.py on {best_ckpt} "
                    f"(pass = CHANGE/SHUFFLED clearly > 1.0)")
    cleanup(world_size)
