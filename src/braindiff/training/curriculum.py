import argparse
import os
import sys

from importlib import resources as _resources
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from braindiff.training.train_single import main as train_single_main
from braindiff.training.train_unified import main as train_unified_main, MIN_CLASS_COUNT
from braindiff.training.train_dual import main as train_dual_main
from braindiff.training.train_dual_priorreport import main as train_dual_priorreport_main
from braindiff.training.train_delta import main as train_delta_main
from braindiff.training.train_change_map import main as train_change_map_main
from braindiff.training.freeze import validate as validate_trainable

SAVE_DIR = "checkpoints"  # matches the trainers' default save_dir
DEFAULT_NUM_WORKERS = 16  # per-rank; the trainers' own default is 4
DEFAULT_NUM_QUERIES = 64  # visual tokens per SERIES; must agree across stages

# HF_HUB_OFFLINE: every weight is already in ~/.cache/huggingface (models/paths.py).
#   This drops the Hub revision checks at startup that otherwise hang if the Hub
#   is unreachable -- 8 of them with 4 ranks x 2 repos.
# expandable_segments: the 30 GB decoder plus a long run leaves no slack for
#   allocator fragmentation.
#
#   HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   torchrun --standalone --nproc_per_node=4 \
#     -m braindiff.training.curriculum --track neurovfm --stage nv_stage1_unified.pt
#
# Run `python -m braindiff.eval.test_dispatch` first (~30 s, no GPU).

# Prefix every launch with:
#   HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# HF_HUB_OFFLINE: weights are already cached; this drops 8 Hub revision checks at
#   startup (4 ranks x 2 models) that otherwise hang if the Hub is unreachable.
# expandable_segments: stage 1 at batch_size 8 reserves ~98% of a 143 GB H200, so
#   there is no slack for allocator fragmentation over a long run.

def find_stage(cfg, track, save_name):
    """Return (stage dict) for the stage whose save_name matches."""
    section = cfg[track]
    for stage in section["stages"]:
        if stage["save_name"] == save_name:
            return stage
    names = [s["save_name"] for s in section["stages"]]
    raise SystemExit(f"stage '{save_name}' not found in track '{track}'. options: {names}")


def run_stage(stage):
    # YAML has no real null, so stage 1's "checkpoint: None" arrives as the string "None".
    checkpoint = stage.get("checkpoint")
    if checkpoint in (None, "None"):
        checkpoint = None

    # Dispatch delta BEFORE building `common`: the delta trainer shares almost
    # none of those keys, and the S3 stage has no `use_lora` at all -- so
    # building common first raises KeyError before we ever reach the branch.
    stage_type = stage["type"]
    if "trainable" in stage:
        validate_trainable(stage["trainable"])
    if stage_type == "delta":
        return run_delta_stage(stage, checkpoint)
    # S3 change-map pretraining: same param surface as `delta` plus the change-map
    # lambdas. Kept as a separate type so the original `delta` stage is untouched and
    # stays reproducible. "delta_supervised" is the pre-2026-08-30 name for this stage
    # and is still accepted so older YAML and run logs resolve; the objective is
    # label-free either way (models/change_map_pretrain.py).
    if stage_type in ("change_map", "delta_supervised"):
        return run_delta_stage(stage, checkpoint, change_map=True)

    common = dict(
        csv=stage["csv"],
        epochs=stage["epochs"],
        batch_size=stage["batch_size"],
        lr=float(stage["lr"]),
        use_lora=stage["use_lora"],
        # Declarative freezing: the stage's list is authoritative over any
        # constructor flag. Validated here so a typo fails at dispatch.
        trainable=tuple(stage["trainable"]),
        grad_accum=stage.get("grad_accum", 1),
        save_name=stage["save_name"],
        use_vision_lora=stage.get("use_vision_lora", True),
        # Whether the encoder's LoRA TRAINS. It is BUILT whenever use_vision_lora
        # is set, because that decides the checkpoint key names -- a later stage
        # built without it matches 0 of an earlier stage's 184 encoder tensors.
        vision_lora_trainable=stage.get("vision_lora_trainable", True),
        vision_lora_r=stage.get("vision_lora_r", 64),
        vision_lora_alpha=stage.get("vision_lora_alpha", 128),
        vision_lora_dropout=stage.get("vision_lora_dropout", 0.05),
        include_delta=stage["include_delta"],
        checkpoint=checkpoint,
        # Data loading dominates every stage (165/654/1672 ms per sample for
        # S1/S2/S4), so this is the setting that decides epoch time. It was in
        # the YAML but never forwarded -- the trainers ran at their own default
        # of 4, not the 16 the header's throughput numbers were measured at.
        num_workers=int(stage.get("num_workers", DEFAULT_NUM_WORKERS)),
        # Perceiver latents per block. Unlike vision_lora_alpha, this is a real
        # tensor shape on connector.*.queries, so a stage that disagrees with its
        # predecessor fails in checkpoint.py rather than loading silently.
        num_queries=int(stage.get("num_queries", DEFAULT_NUM_QUERIES)),
    )

    # Sentence-level contrastive weight (report-gen stages only); defaults to off.
    contrastive_weight = float(stage.get("contrastive_weight", 0.0))
    max_caption_length = int(stage.get("max_caption_length", 600))

    # Counterfactual grounding: swap the prior (dual) or the only scan (single)
    # between samples and require the caption to get less likely. Off by default
    # so existing tracks are unchanged.
    # >1 downweights normality/absence sentences in the S2 captioning CE.
    # 1.0 is an exact no-op, so omitting it reproduces the previous behaviour.
    content_weight = float(stage.get("content_weight", 1.0))
    cf_weight = float(stage.get("cf_weight", 0.0))
    counterfactual_margin = float(stage.get("counterfactual_margin", 0.5))
    change_weight = float(stage.get("change_weight", 0.0))

    # Early-stopping patience. Only the delta branch forwarded this before, so a
    # `patience:` on any other stage was silently ignored -- the same class of
    # dead YAML key as num_queries/num_workers. All four trainers accept it;
    # omitting it still falls back to each trainer's own default.
    if "patience" in stage:
        common["patience"] = int(stage["patience"])

    # Decoder LoRA geometry. Never forwarded before, so the YAML's "MUST equal
    # S2's" comment was enforced only by the trainers' defaults happening to agree
    # with it -- editing the YAML alone changed nothing. Not in `common`: the
    # unified trainer builds no decoder LoRA and takes none of these.
    lora_r = int(stage.get("lora_r", 16))
    lora_alpha = int(stage.get("lora_alpha", 32))
    lora_dropout = float(stage.get("lora_dropout", 0.05))

    if stage_type == "single":
        train_single_main(**common, contrastive_weight=contrastive_weight,
                          lora_r=lora_r, lora_alpha=lora_alpha,
                          max_caption_length=max_caption_length,
                          cf_weight=cf_weight, counterfactual_margin=counterfactual_margin,
                          content_weight=content_weight)
    elif stage_type == "unified":
        # Bounding-box grounding and pathology classification in one stage; the
        # dataloader interleaves both objectives within every batch.
        min_class_count = int(stage.get("min_class_count", MIN_CLASS_COUNT))
        train_unified_main(**common, max_caption_length=max_caption_length,
                           max_prompt_length=int(stage.get("max_prompt_length", 384)),
                           min_class_count=min_class_count)
    elif stage_type in ("dual", "dual_priorreport"):
        # DiffEncoder pretrained in the delta stage warm-starts the delta block.
        # `dual_priorreport` shares this entire kwarg surface and swaps only the
        # trainer, which additionally conditions on report1 (see
        # trainer/train_dual_priorreport.py). S4 itself is untouched.
        dual_main = (train_dual_priorreport_main if stage_type == "dual_priorreport"
                     else train_dual_main)
        # Keys only the prior-report trainer understands. Passed explicitly because
        # its signature ends in **_ignored, which would otherwise swallow a
        # misspelled or unrouted knob and train with the default in silence.
        extra = {}
        if stage_type == "dual_priorreport":
            for k in ("prior_report_dropout", "checkpoint_every"):
                if k in stage:
                    extra[k] = stage[k]
        dual_main(**common, image_csv=stage["image_csv"], diff_checkpoint=stage.get("diff_checkpoint"),
                        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                        content_weight=content_weight,
                        contrastive_weight=contrastive_weight, max_caption_length=max_caption_length,
                        cf_weight=cf_weight, counterfactual_margin=counterfactual_margin,
                        change_weight=change_weight, **extra)
    else:
        raise SystemExit(f"unknown stage type: {stage_type}")


def run_delta_stage(stage, checkpoint, change_map=False):
    # The delta trainer has a different param set: no use_lora/vision LoRA/
    # include_delta, and warm-starts a frozen fusion encoder from the prior stage's
    # checkpoint via pretrained_ckpt (a path).
    kwargs = dict(
        csv=stage["csv"],
        image_csv = stage["image_csv"],
        epochs=stage["epochs"],
        batch_size=stage["batch_size"],
        lr=float(stage["lr"]),
        save_name=stage["save_name"],
        pretrained_ckpt=os.path.join(SAVE_DIR, checkpoint) if checkpoint else None,
        num_workers=int(stage.get("num_workers", DEFAULT_NUM_WORKERS)),
        # S3 trains connector.delta, so this must match the S4 stage that
        # inherits it -- hence the shared default rather than the loop below.
        num_queries=int(stage.get("num_queries", DEFAULT_NUM_QUERIES)),
    )
    # Optional delta-specific hyperparameters; fall back to the trainer defaults.
    for k in ("bottleneck_dim", "dup_fraction", "warmup_ratio",
              "lambda_recon", "lambda_norm", "lambda_gate", "lambda_antisym",
              "lambda_disc", "lambda_compress", "disc_temperature",
              "disc_negatives", "local_attn_layers", "attn_dim",
              "val_fraction", "patience", "cache_dir",
              "vision_lora_r", "vision_lora_alpha"):
        if k in stage:
            kwargs[k] = stage[k]
    if change_map:
        for k in ("lambda_gate_track", "lambda_gate_dup", "lambda_gate_stable",
                  "lambda_global",
                  "spliced_disc", "splice_frac", "weight_decay"):
            if k in stage:
                kwargs[k] = stage[k]
        # A bare filename in the YAML, like `checkpoint:` -- resolve it against
        # SAVE_DIR the same way, so the two are written the same way in the stage.
        if stage.get("delta_ckpt") not in (None, "None"):
            kwargs["delta_ckpt"] = os.path.join(SAVE_DIR, stage["delta_ckpt"])
        return train_change_map_main(**kwargs)
    train_delta_main(**kwargs)


if __name__ == "__main__":
    # Launch with: torchrun --standalone --nproc_per_node=N -m braindiff.training.MultiModal.curriculum ...
    p = argparse.ArgumentParser()
    p.add_argument("--config",
                   default=str(_resources.files("braindiff") / "configs" / "curriculum.yaml"))
    p.add_argument("--track", default="no_delta")
    p.add_argument("--stage", required=True, help="save_name of the stage to run")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stage = find_stage(cfg, args.track, args.stage)
    run_stage(stage)
