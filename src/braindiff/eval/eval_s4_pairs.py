"""Generate the first N val pairs for one S4 checkpoint and dump {hyps, refs} JSON.

Mirrors train_dual.main's model/loader construction exactly (same stage config,
same split dir, same seed), then loads the checkpoint the same way the in-trainer
test pass does (strict=False -- the frozen decoder weights are not in the file).
The val loader is shuffle=False, so "first N" is the same deterministic subset
select.py scored in-loop and is comparable across checkpoints.

Greedy, repetition_penalty=1.0, no n-gram blocking -- the settings S4 trained and
sampled under. Score the dumps afterwards with score_s4_pairs.py (BLEU-4/METEOR)
and radgraph_score.py (rg_er, isolated env).

  python -m braindiff.eval.eval_s4_pairs --ckpt nv_stage4_reportgen.pt --n_pairs 128 \
      --out checkpoints/s4_eval/best.json
"""
import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml

from braindiff.models.captioner import DeltaDiffCaptioner_Qwen3
from braindiff.models.prompts import PromptTable
from braindiff.data.report_text import strip_pathology_prefix
from braindiff.training.curriculum import find_stage
from braindiff.training.train_dual import IMG_SIZE, make_loaders

CONFIG = os.path.join(os.path.dirname(__file__), "curriculum.yaml")


def main(ckpt, n_pairs, out, save_dir, batch_size, num_workers, seed):
    stage = find_stage(yaml.safe_load(open(CONFIG)), "neurovfm", "nv_stage4_reportgen.pt")
    device = torch.device("cuda")
    max_caption_length = stage["max_caption_length"]

    _, val_loader, _ = make_loaders(
        stage["csv"], stage["image_csv"], batch_size, num_workers,
        max_caption_length, seed, distributed=False,
        content_weight=stage["content_weight"],
    )

    model = DeltaDiffCaptioner_Qwen3(
        single_timepoint=False,
        use_vision_lora=stage["use_vision_lora"],
        vision_lora_r=stage["vision_lora_r"],
        vision_lora_alpha=stage["vision_lora_alpha"],
        vision_lora_dropout=stage["vision_lora_dropout"],
        num_queries=stage["num_queries"],
        include_delta=stage["include_delta"],
        lora_r=stage["lora_r"],
        lora_alpha=stage["lora_alpha"],
        lora_dropout=stage["lora_dropout"],
        use_lora=stage["use_lora"],
        pretrained_connector=False,   # warm-started stage: the file supplies it
        max_caption_length=max_caption_length,
        num_change_classes=0,         # change_weight is 0 in the S4 config
        counterfactual_margin=stage["counterfactual_margin"],
        device=device,
    ).to(device)

    state = torch.load(os.path.join(save_dir, ckpt), map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The frozen decoder base is absent from the file by design (only lora_ keys
    # are saved), so `missing` is large and expected -- `unexpected` is not.
    live = [k for k in state if k in dict(model.named_parameters())
            or k in dict(model.named_buffers())]
    print(f"[ckpt] {ckpt}: {len(state)} tensors in file, {len(live)} matched, "
          f"{len(unexpected)} unexpected", flush=True)
    if unexpected:
        raise SystemExit(f"unexpected keys: {unexpected[:5]}")

    prompt_table = PromptTable(model.decoder.tokenizer, single_timepoint=False,
                               include_delta=stage["include_delta"])
    model.eval()

    hyps, refs = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=ckpt):
            if len(hyps) >= n_pairs:
                break
            caps = model.generate_caption_batch(
                tokens_main=batch["tokens_main"].to(device),
                coords_main=batch["coords_main"].to(device),
                present_main=batch["present_main"].to(device),
                tokens_ref=batch["tokens_ref"].to(device),
                coords_ref=batch["coords_ref"].to(device),
                present_ref=batch["present_ref"].to(device),
                prompt_table=prompt_table,
                max_new_tokens=round(max_caption_length * 1.1),
                num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
            )
            hyps.extend(caps)
            refs.extend(batch["caption"])

    hyps = [strip_pathology_prefix(h) for h in hyps][:n_pairs]
    refs = [strip_pathology_prefix(r) for r in refs][:n_pairs]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"checkpoint": ckpt, "n_pairs": len(hyps), "hyps": hyps, "refs": refs}, f, indent=1)
    print(f"Wrote {out}  ({len(hyps)} pairs)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n_pairs", type=int, default=128)
    p.add_argument("--out", required=True)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=10)
    a = p.parse_args()
    main(a.ckpt, a.n_pairs, a.out, a.save_dir, a.batch_size, a.num_workers, a.seed)
