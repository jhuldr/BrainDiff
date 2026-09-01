"""Declarative trainable groups: one YAML list per stage, applied here.

Every cell of the curriculum's trainability table becomes a one-line ablation:

    trainable: [encoder_lora, connector.scan, connector.proj, decoder_lora]

Three properties that matter:

  1. Freeze all, then unfreeze -- never unfreeze selectively. A parameter in no group is
     frozen by construction, so adding a module later cannot silently start training it.
  2. A named group matching zero parameters raises. A stage that believes it trains encoder
     LoRA but built none would otherwise run to completion and write a checkpoint.
     Deliberately-absent groups are expressed by omission, not by naming and matching none.
  3. Group names are shared with trainer/checkpoint.py, so the freeze audit and the load
     report can be diffed by eye.
"""
from typing import Dict, Sequence

GROUPS = {
    "encoder_lora":    lambda k: k.startswith("vision_encoder") and "lora_" in k,
    "encoder_base":    lambda k: k.startswith("vision_encoder") and "lora_" not in k,
    "connector.scan":  lambda k: k.startswith("connector.scan"),
    "connector.proj":  lambda k: k.startswith("connector.proj"),
    "connector.delta": lambda k: k.startswith("connector.delta"),
    "diff_encoder":    lambda k: k.startswith("diff_encoder"),
    "change_map":      lambda k: k.startswith("change_map"),
    "embeddings":      lambda k: k.startswith(("temporal_embedding", "modality_type_embedding")),
    "contrastive":     lambda k: k.startswith(("visual_proj", "text_proj", "logit_temperature")),
    "change_head":     lambda k: k.startswith("change_head"),
    "decoder_lora":    lambda k: k.startswith("decoder") and "lora_" in k,
    "decoder_base":    lambda k: k.startswith("decoder") and "lora_" not in k,
}


def validate(spec: Sequence[str]) -> None:
    """Raise on an unknown group name. Called by test_dispatch at bind time so a
    typo fails in 30 s instead of at hour three of a multi-hour launch."""
    unknown = [g for g in spec if g not in GROUPS]
    if unknown:
        raise ValueError(
            f"unknown trainable group(s) {unknown}; valid names: {sorted(GROUPS)}")


def apply_trainable(model, spec: Sequence[str], is_main: bool = True) -> Dict[str, int]:
    """Freeze everything, unfreeze the named groups, print an audit, return counts."""
    validate(spec)
    spec = list(spec)

    for p in model.parameters():
        p.requires_grad = False

    counts = {g: 0 for g in spec}
    params = {g: 0 for g in spec}
    for name, p in model.named_parameters():
        for g in spec:
            if GROUPS[g](name):
                p.requires_grad = True
                counts[g] += 1
                params[g] += p.numel()
                break

    empty = [g for g in spec if counts[g] == 0]
    if empty:
        raise ValueError(
            f"trainable group(s) {empty} matched 0 parameters. Either the module was "
            f"not built (check use_vision_lora / use_lora / include_delta) or the "
            f"group name no longer matches the model's parameter names. Omit the "
            f"group rather than naming one that matches nothing.")

    if is_main:
        total = sum(p.numel() for p in model.parameters())
        train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[freeze] trainable groups: {', '.join(spec)}")
        for g in spec:
            print(f"[freeze]   {g:<18} {counts[g]:>4} tensors  {params[g]:>13,} params")
        print(f"[freeze]   {'TOTAL':<18} {'':>4}          {train:>13,} / {total:,} "
              f"({100*train/max(total,1):.2f}%)")
    return params


def audit(model, spec: Sequence[str]) -> None:
    """Post-backward check: every named group must have a gradient, every unnamed group must have
none. Driven by the same spec the trainer used, so the test cannot drift from the config.
Catches the silent `enable_input_require_grads` failure (zero grad, no error).

Partial counts within a trainable group are normal, not failures:
  * decoder_lora reports 160/320 at step 0 -- LoRA initialises lora_B to zeros, so
    dL/d(lora_A) is zero until B has taken one step.
  * embeddings reports 2/4 in single-timepoint mode: temporal_embedding_ref and _delta are
    built for the checkpoint lineage but no single-timepoint forward touches them.
The check is `any`, not `all`, for this reason.
    """
    validate(spec)
    spec = set(spec)
    seen = {}
    for name, p in model.named_parameters():
        for g, pred in GROUPS.items():
            if pred(name):
                has = p.grad is not None and p.grad.abs().sum().item() > 0
                seen.setdefault(g, []).append(has)
                break
    problems = []
    for g, flags in sorted(seen.items()):
        if g in spec and not any(flags):
            problems.append(f"{g}: TRAINABLE but every gradient is zero/None")
        if g not in spec and any(flags):
            problems.append(f"{g}: FROZEN but has a non-zero gradient")
    for g, flags in sorted(seen.items()):
        mark = "train" if g in spec else "frozen"
        print(f"[audit] {g:<18} {mark}  {sum(flags)}/{len(flags)} tensors with grad")
    if problems:
        raise AssertionError("gradient audit failed:\n  " + "\n  ".join(problems))
    print("[audit] PASS")
