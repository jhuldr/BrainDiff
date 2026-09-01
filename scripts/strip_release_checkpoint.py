"""Produce the redistributable checkpoint: our trained weights, none of NeuroVFM's.

The trained checkpoint carries NeuroVFM's frozen ViT-B verbatim -- 136 non-LoRA
`vision_encoder.*` tensors, 85.8 M parameters. Those weights are NeuroVFM's, not ours, and the
repository they come from is gated, so they must not be redistributed with this project. They
are also redundant: `models/paths.py` already resolves the encoder from the Hub, and the model
builds its backbone from there before any checkpoint is applied.

This script drops exactly those tensors and keeps everything the project trained: encoder LoRA,
the connector, the change map when present, the decoder LoRA, and the embeddings.

    python scripts/strip_release_checkpoint.py \
        --in  checkpoints/nv_stage4_priorreport_nodelta_10.pt \
        --out checkpoints/braindiff_production.pt

The result loads against a backbone built from the Hub: `trainer/checkpoint.py` reports per
group, so an encoder line reading `48/48` (LoRA only) rather than `184/184` is the expected
signature of a stripped file, not a partial load.
"""
import argparse
import hashlib
import torch

# NeuroVFM's own weights: frozen backbone tensors, never trained here.
def is_third_party(key: str) -> bool:
    return key.startswith("vision_encoder") and "lora_" not in key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sd = torch.load(a.src, map_location="cpu")
    drop = [k for k in sd if is_third_party(k)]
    keep = {k: v for k, v in sd.items() if k not in set(drop)}

    n_drop = sum(sd[k].numel() for k in drop)
    n_keep = sum(v.numel() for v in keep.values())
    print(f"in  : {len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} params")
    print(f"drop: {len(drop)} tensors, {n_drop:,} params   (NeuroVFM ViT-B, fetched from the Hub)")
    print(f"keep: {len(keep)} tensors, {n_keep:,} params")

    assert not any(is_third_party(k) for k in keep), "third-party weights survived the filter"
    assert any("lora_" in k and k.startswith("vision_encoder") for k in keep), \
        "encoder LoRA was dropped -- the filter is too broad"
    assert any(k.startswith("connector.scan") for k in keep), "connector.scan missing"

    if a.dry_run:
        print("dry run; nothing written")
        return
    torch.save(keep, a.dst)
    md5 = hashlib.md5(open(a.dst, "rb").read()).hexdigest()
    print(f"wrote {a.dst}  md5 {md5}")


if __name__ == "__main__":
    main()
