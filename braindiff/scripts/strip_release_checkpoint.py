"""Package the release checkpoint: Stage 2 trainable tensors merged with the Stage 1-D encoder blocks 10-12 + norm, so it
loads without the Stage 1-D checkpoint. Frozen NeuroVFM / Qwen3 weights are not included; tensors stay fp32, bit-identical.

    python scripts/strip_release_checkpoint.py --stage2 checkpoints/stage2/epoch20.pt \
        --stage1 checkpoints/stage1/epoch5.pt --out checkpoints/braindiff.pt
"""
import argparse
import hashlib

import torch

TOP = tuple(f"encoder.model.blocks.{i}." for i in (9, 10, 11)) + ("encoder.model.norm.",)
CFG = ("lora_r", "lora_alpha", "max_prompt", "max_report_tokens", "train_connector", "status_first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", required=True); ap.add_argument("--stage1", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s2 = torch.load(a.stage2, map_location="cpu", weights_only=False)
    s1 = torch.load(a.stage1, map_location="cpu", weights_only=False)
    top = {k: v for k, v in s1["model"].items() if k.startswith(TOP)}
    sd = dict(s2["model"])
    assert not set(sd) & set(top), "Stage 2 checkpoint already holds encoder blocks"
    assert len(top) == 35, f"expected 35 encoder block / norm tensors in the Stage 1-D checkpoint, got {len(top)}"
    assert any("lora_B" in k for k in sd) and any(k.startswith("connector.scan.") for k in sd) \
        and "modality_type_embedding.weight" in sd, "Stage 2 checkpoint lacks LoRA / connector / modality embedding"
    assert not any(k.startswith("encoder.") and k not in top for k in sd), "frozen NeuroVFM weights in the checkpoint"
    sd.update(top)
    cfg = {k: s2["cfg"][k] for k in CFG}
    print(f"stage2: {len(s2['model'])} tensors (dropped keys: {sorted(set(s2) - {'model', 'epoch', 'step', 'cfg'})}); "
          f"stage1 encoder blocks: {len(top)}; release: {len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} params")
    if a.dry_run:
        print("dry run; nothing written")
        return
    torch.save({"model": sd, "epoch": s2["epoch"], "step": s2["step"], "cfg": cfg}, a.out)
    print(f"wrote {a.out}  md5 {hashlib.md5(open(a.out, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
