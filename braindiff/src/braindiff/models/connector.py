"""Per-scan connector: one series' 2016 encoder tokens -> 64 queries -> Qwen3 width, initialised from NeuroVFM's
vision_connector.pt.

Per-series is load-bearing: each perceiver call sees one modality's tokens, so no padded keys; an absent modality is
dropped, matching the distribution vision_connector.pt was trained on. Subclasses upstream so the checkpoint loads with
a prefix rename; the one override removes upstream's hard bf16 cast.
"""
import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from neurovfm.models.perceiver import PerceiverResampler as _NVFMResampler

VISUAL_DIM = 768        # NeuroVFM ViT-B width
DECODER_DIM = 5120      # Qwen3-14B hidden_size
NUM_QUERIES = 64


class PerceiverResampler(_NVFMResampler):
    """Upstream module, dtype-neutral. [B, N, 768] -> [B, num_queries, 768]."""

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        q = self.queries.unsqueeze(0).expand(visual.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, visual)
        return self.final_norm(q)


class VisionConnector(nn.Module):
    """[B, N, 768] -> [B, num_queries, 5120]."""

    def __init__(self, visual_dim=VISUAL_DIM, decoder_dim=DECODER_DIM, num_queries=NUM_QUERIES, num_layers=6,
                 num_heads=8, dropout=0.0):
        super().__init__()
        self.scan = PerceiverResampler(dim=visual_dim, num_queries=num_queries, num_layers=num_layers,
                                       num_heads=num_heads, dropout=dropout)
        self.num_queries = num_queries
        # keys proj.fc1.* / proj.fc2.*; no leading LayerNorm (the resampler's final_norm already normalises).
        self.proj = Mlp(in_features=visual_dim, hidden_features=decoder_dim, out_features=decoder_dim,
                        act_layer=nn.GELU, drop=0.0)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        return self.proj(self.scan(visual))


def remap_pretrained_connector(state: dict) -> dict:
    """vision_connector.pt state dict -> VisionConnector keys. Raises on a key that is neither perceiver.* nor mlp.*."""
    state = state.get("state_dict", state)
    out = {}
    for k, v in state.items():
        if k.startswith("perceiver."):
            out[f"scan.{k[len('perceiver.'):]}"] = v.clone()
        elif k.startswith("mlp."):
            out[f"proj.{k[len('mlp.'):]}"] = v.clone()
        else:
            raise KeyError(f"unexpected key in vision_connector.pt: {k!r}")
    return out


def load_pretrained_connector(connector, path=None, is_main=True):
    """Load the released connector into `connector`, strict=True (a silent partial load degrades every later stage)."""
    from braindiff.models.encoder import connector_path
    path = path or connector_path()
    mapped = remap_pretrained_connector(torch.load(path, map_location="cpu", weights_only=False))
    connector.load_state_dict(mapped, strict=True)
    if is_main:
        print(f"[connector] loaded {len(mapped)} tensors / {sum(v.numel() for v in mapped.values()):,} params from "
              f"{path}")
    return mapped
