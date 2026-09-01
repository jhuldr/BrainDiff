"""Stage-to-stage checkpoint loading, with a report loud enough to catch a miss.

This exists because a silent partial load already happened. S3/S4 were configured with
`use_vision_lora: false`, which does not merely skip training the encoder's LoRA -- it skips
BUILDING it, and `get_peft_model` renames every encoder key. Those stages matched 0 of 184
encoder tensors from the S2 checkpoint, fell back to the bare NeuroVFM weights, and reported
nothing: with strict=False a total miss and a perfect load look identical.

Second trigger for the same failure: `vision_lora_r` defaults to 64 in the trainers while
the curriculum sets 16, and a rank mismatch silently drops all 48 LoRA tensors on a shape
filter.

So: group the keys, print what landed, and refuse to continue when a group the checkpoint
clearly contains lands empty.
"""

GROUPS = (
    ("encoder", lambda k: k.startswith("vision_encoder")),
    ("  of which LoRA", lambda k: k.startswith("vision_encoder") and "lora_" in k),
    ("  of which base", lambda k: k.startswith("vision_encoder") and "lora_" not in k),
    ("connector.scan", lambda k: k.startswith("connector.scan")),
    ("connector.delta", lambda k: k.startswith("connector.delta")),
    ("connector.proj", lambda k: k.startswith("connector.proj")),
    ("diff_encoder", lambda k: k.startswith("diff_encoder")),
    ("change_map", lambda k: k.startswith("change_map")),
    ("decoder LoRA", lambda k: k.startswith("decoder") and "lora_" in k),
)

# Groups whose total absence means the stage is not warm-started the way the
# curriculum says it is. `connector.delta` is excluded: S2 is single-timepoint and
# legitimately has no delta resampler to hand on.
REQUIRED = ("encoder", "connector.scan")


def load_stage_checkpoint(model, state, label="checkpoint", is_main=True,
                          strict_groups=REQUIRED):
    """Shape-filtered load onto `model`, with a per-group report.

    strict=False suppresses missing and unexpected keys but NOT size mismatches,
    so the filter is what keeps an older checkpoint from raising -- and the report
    is what keeps it from passing unnoticed.

    Returns the dict of tensors actually loaded.
    """
    own = model.state_dict()
    kept, mismatched, absent = {}, [], []
    for k, v in state.items():
        if k not in own:
            absent.append(k)
        elif own[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(own[k].shape)))
        else:
            kept[k] = v
    model.load_state_dict(kept, strict=False)

    if is_main:
        print(f"Loaded {label}: {len(kept)}/{len(state)} tensors")
        for name, pred in GROUPS:
            in_ckpt = sum(1 for k in state if pred(k))
            if not in_ckpt:
                continue
            got = sum(1 for k in kept if pred(k))
            flag = "" if got == in_ckpt else "   <-- PARTIAL" if got else "   <-- NONE"
            print(f"    {name:18s} {got:4d}/{in_ckpt}{flag}")
        if mismatched:
            print(f"    shape mismatch on {len(mismatched)}, e.g. "
                  f"{mismatched[0][0]}: ckpt {mismatched[0][1]} vs model {mismatched[0][2]}")
        if absent:
            print(f"    not in this model: {len(absent)} (e.g. {absent[0]})")
        # A released checkpoint carries no frozen backbone: those weights are NeuroVFM's and
        # are fetched from the Hub by models/paths.py before the checkpoint is applied. Say so,
        # so the short encoder line is not read as a partial load.
        enc_lora = sum(1 for k in state if k.startswith("vision_encoder") and "lora_" in k)
        enc_base = sum(1 for k in state if k.startswith("vision_encoder") and "lora_" not in k)
        if enc_lora and not enc_base:
            print("    encoder base absent from this checkpoint -- backbone comes from the "
                  "Hub (mlinslab/neurovfm-encoder); this is expected for a release build")

    for name in strict_groups:
        pred = dict((n, p) for n, p in GROUPS)[name]
        in_ckpt = sum(1 for k in state if pred(k))
        if in_ckpt and not sum(1 for k in kept if pred(k)):
            raise RuntimeError(
                f"{label}: the checkpoint holds {in_ckpt} '{name}' tensors and NONE "
                f"of them loaded. Almost always a build-config mismatch -- check "
                f"use_vision_lora (LoRA renames every encoder key) and "
                f"vision_lora_r (a rank mismatch drops LoRA on the shape filter). "
                f"Refusing to train on an unintentionally re-initialised {name}."
            )
    return kept
