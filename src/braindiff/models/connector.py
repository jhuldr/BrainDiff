"""Per-series connector: one modality's 2016 encoder tokens -> 64 -> Qwen3 width.

Two resamplers, one shared projection:

  scan   trained S1-S2 on real scans, sees only scan features
  delta  a separate module initialised as a copy of `scan`, trained at S3 on 40k unlabeled
         pairs; it only ever sees DiffEncoder output, a signed residual field rather than an
         image. Sharing one resampler would force it to meet delta statistics for the first
         time at S4, on 8,648 pairs.
  proj   shared 768 -> 5120 -> 5120, so both branches land in the same space.

Per-series is load-bearing: each perceiver call sees exactly one modality's 2016 tokens, so
there are no padded keys and no key_padding_mask -- an absent modality is dropped from the
sequence rather than contributing zeros. That is also the distribution vision_connector.pt
was trained on, which makes it a valid init.

Subclasses neurovfm's PerceiverResampler rather than copying it, so every parameter name is
inherited and vision_connector.pt loads with a pure prefix rename. The single override
exists because upstream's forward ends `return queries.to(torch.bfloat16)`, a hard cast that
breaks CPU fp32 shape tests and the numerical-equivalence check.
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
    """[B, N, 768] -> [B, num_queries, 5120], N variable, output fixed."""

    def __init__(self, visual_dim: int = VISUAL_DIM, decoder_dim: int = DECODER_DIM,
                 num_queries: int = NUM_QUERIES, num_layers: int = 6,
                 num_heads: int = 8, dropout: float = 0.0, build_delta: bool = True):
        super().__init__()
        make = lambda: PerceiverResampler(dim=visual_dim, num_queries=num_queries,
                                          num_layers=num_layers, num_heads=num_heads,
                                          dropout=dropout)
        self.scan = make()
        self.delta = make() if build_delta else None
        self.num_queries = num_queries
        # timm's Mlp, the exact class upstream uses -> keys are proj.fc1.* / proj.fc2.*
        # (norm/drop1/drop2 are parameterless). No leading LayerNorm: the resampler's
        # final_norm already normalises this input.
        self.proj = Mlp(in_features=visual_dim, hidden_features=decoder_dim,
                        out_features=decoder_dim, act_layer=nn.GELU, drop=0.0)

    def forward(self, visual: torch.Tensor, is_delta: bool = False) -> torch.Tensor:
        """[B, N, visual_dim] -> [B, num_queries, decoder_dim]."""
        resampler = self.delta if (is_delta and self.delta is not None) else self.scan
        return self.proj(resampler(visual))
