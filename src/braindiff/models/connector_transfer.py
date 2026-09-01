"""Remap NeuroVFM's released vision_connector.pt onto our two-resampler connector.

Upstream ships one `perceiver` and one `mlp`. We need `scan` + `delta` + a shared
`proj`. Because models/connector.py subclasses upstream's PerceiverResampler and uses
upstream's timm Mlp, every parameter name below is inherited -- this is a pure prefix
rename with no name-matching risk.

`delta` initialises as a COPY of `scan`. S3 is what makes them diverge; starting it
from the pretrained scan weights beats starting it from noise, because a residual field
is still a field of 768-d NeuroVFM features.
"""
import torch


def remap_pretrained_connector(state: dict) -> dict:
    """vision_connector.pt state dict -> keys our VisionConnector expects.

    Raises on any key that is neither `perceiver.*` nor `mlp.*` -- an unexpected key
    means the released format changed and a silent partial load would follow.
    """
    state = state.get("state_dict", state)
    out, n_p, n_m = {}, 0, 0
    for k, v in state.items():
        if k.startswith("perceiver."):
            suf = k[len("perceiver."):]
            out[f"scan.{suf}"] = v.clone()
            out[f"delta.{suf}"] = v.clone()
            n_p += 1
        elif k.startswith("mlp."):
            out[f"proj.{k[len('mlp.'):]}"] = v.clone()
            n_m += 1
        else:
            raise KeyError(f"unexpected key in vision_connector.pt: {k!r}")
    assert len(out) == 2 * n_p + n_m, "remap lost or duplicated keys"
    return out


def load_pretrained_connector(connector, path: str = None, is_main: bool = True) -> dict:
    """Load the released connector into `connector`, strictly.

    strict=True on purpose: a silent partial load here degrades every downstream stage,
    and trainer/checkpoint.py exists because exactly that once happened.
    When connector.delta is None (build_delta=False), the delta.* keys are dropped
    first -- that is the only tolerated omission.
    """
    from braindiff.models.paths import connector_path
    path = path or connector_path()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    mapped = remap_pretrained_connector(raw)
    if connector.delta is None:
        mapped = {k: v for k, v in mapped.items() if not k.startswith("delta.")}
    incompatible = connector.load_state_dict(mapped, strict=True)
    if is_main:
        n = sum(v.numel() for v in mapped.values())
        print(f"[connector] loaded {len(mapped)} tensors / {n:,} params from {path}")
    return mapped
