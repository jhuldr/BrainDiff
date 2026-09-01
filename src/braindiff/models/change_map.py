"""Coordinate-tagged change map -- replaces DiffEncoder + connector.delta on the delta path.

The old delta was redundant: the decoder already receives both timepoints as pooled scan
blocks plus the prior report, and the delta pooled its dense computation to 64
non-positional latents, discarding the one thing the pooled decoder cannot recompute --
location. This module computes instead:

  1. cross-timepoint correspondence (windowed cross-attn -> warped prior A_hat), and
  2. the location of change, as a coordinate-tagged change map.

Per modality on the fixed 12x14x12 grid, the dense change field is coarsened to a
4x4x4 = 64-cell map, each cell carrying its 3D coordinate embedding. Output is
[B, M, 64, decoder_dim] -- the same 4-blocks x 64-tokens contract the splice machinery
expects, so nothing downstream changes. Label-free; trained by models/change_map_pretrain.py.

`forward` also returns the dense change field (for the Reconstructor-based contrastive term)
and the pre-projection coarse map and saliency.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from braindiff.models.diff_encoder import WindowedCrossAttention, neighbour_index, TOKEN_GRID


class ChangeMapEncoder(nn.Module):
    def __init__(self, feature_dim: int = 768, decoder_dim: int = 5120,
                 coarse_grid=(4, 4, 4), attn_dim: int = 512, dropout: float = 0.1,
                 grid=TOKEN_GRID):
        super().__init__()
        # change_dim is fixed to feature_dim so the dense field can feed the
        # per-token FiLM Reconstructor (which requires feature_dim, per DiffPretrainModel).
        self.feature_dim = feature_dim
        self.change_dim = feature_dim
        self.decoder_dim = decoder_dim
        self.grid = grid
        self.coarse_grid = coarse_grid
        self.n_cells = coarse_grid[0] * coarse_grid[1] * coarse_grid[2]      # 64
        self.tokens_per_volume = grid[0] * grid[1] * grid[2]                 # 2016
        self.sqrt_d = math.sqrt(feature_dim)

        # --- correspondence (reuse the DiffEncoder's windowed cross-attn) ---
        idx, valid = neighbour_index(grid, 1)
        self.register_buffer("nb_idx", idx, persistent=False)
        self.register_buffer("nb_valid", valid, persistent=False)
        self.cross_attn = WindowedCrossAttention(feature_dim, attn_dim, dropout=dropout)

        # --- dense change feature: [r, relu(r), relu(-r), raw B-A] + saliency scalar ---
        self.change_mlp = nn.Sequential(
            nn.Linear(4 * feature_dim + 1, self.change_dim),
            nn.LayerNorm(self.change_dim), nn.GELU(), nn.Dropout(dropout))

        # --- per-cell saliency gate (for L_saliency; the only absolutely-scaled signal) ---
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.change_dim, self.change_dim // 4), nn.GELU(),
            nn.Linear(self.change_dim // 4, 1))

        # --- coordinate embedding: cell-center xyz in [0,1]^3 -> change_dim ---
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, self.change_dim // 2), nn.GELU(),
            nn.Linear(self.change_dim // 2, self.change_dim))
        self.register_buffer("cell_coords", self._cell_coords(), persistent=False)

        # --- project coarse cell -> decoder width ---
        # The per-token gate is on the read path via GATED POOLING (it weights which
        # tokens dominate each cell, below), so it shapes the read tokens' content --
        # no need to also feed it in as a feature.
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.change_dim), nn.Linear(self.change_dim, decoder_dim),
            nn.GELU(), nn.Linear(decoder_dim, decoder_dim))

    def _cell_coords(self):
        cd, ch, cw = self.coarse_grid
        zs = (torch.arange(cd) + 0.5) / cd
        ys = (torch.arange(ch) + 0.5) / ch
        xs = (torch.arange(cw) + 0.5) / cw
        g = torch.stack(torch.meshgrid(zs, ys, xs, indexing="ij"), dim=-1)
        return g.reshape(-1, 3)                                             # [n_cells, 3]

    def forward(self, f_ref, f_main):
        """f_ref/f_main [B, M, N, feature_dim] (N=2016) ->
        tokens [B, M, n_cells, decoder_dim] + aux dict (dense/coarse/gate/d)."""
        b, m, n, d = f_ref.shape
        a = f_ref.reshape(b * m, n, d)
        bb = f_main.reshape(b * m, n, d)

        # correspondence: align prior to current, then the residual is real change.
        A_hat = a + self.cross_attn(bb, a, self.nb_idx, self.nb_valid)      # [B*M, N, D]
        r = bb - A_hat                                                      # corrected diff
        raw = bb - a                                                        # pointwise diff
        d_sal = (1.0 - F.cosine_similarity(A_hat, bb, dim=-1)).clamp(min=0.0)  # [B*M, N]

        feat_in = torch.cat([r, F.relu(r), F.relu(-r), raw, d_sal.unsqueeze(-1)], dim=-1)
        dense = self.change_mlp(feat_in)                                   # [B*M, N, change_dim]

        # PER-TOKEN saliency gate -- supervised (L_saliency) at fine resolution, where the
        # change signal is sparse (cosine dist 1.74x / AUC 0.903). A per-CELL gate on the
        # 4x4x4 map washes out to uniform (mean target 0.76), which is why v2 localized at
        # chance; the fix is to keep localization at the token grid.
        g_tok = torch.sigmoid(self.gate_mlp(dense))                        # [B*M, N, 1]

        # GATED (attention) pool to the 64 read cells: changed tokens dominate their cell,
        # so the gate is genuinely on the read path -- it decides each cell's CONTENT.
        gD, gH, gW = self.grid
        gd_vol = (dense * g_tok).transpose(1, 2).reshape(b * m, self.change_dim, gD, gH, gW)
        gg_vol = g_tok.transpose(1, 2).reshape(b * m, 1, gD, gH, gW)
        num = F.adaptive_avg_pool3d(gd_vol, self.coarse_grid).reshape(b * m, self.change_dim, self.n_cells)
        den = F.adaptive_avg_pool3d(gg_vol, self.coarse_grid).reshape(b * m, 1, self.n_cells)
        coarse = (num / den.clamp(min=1e-4)).transpose(1, 2)              # [B*M, n_cells, cdim]

        # location: add the per-cell coordinate embedding (the piece the old pool lost).
        coarse = coarse + self.coord_mlp(self.cell_coords).unsqueeze(0)
        tokens = self.out_proj(coarse)                                    # [B*M, n_cells, D]

        aux = {
            "dense": dense.reshape(b, m, n, self.change_dim),
            "coarse": coarse.reshape(b, m, self.n_cells, self.change_dim),
            "gate": g_tok.squeeze(-1).reshape(b, m, n),                    # PER-TOKEN now [B,M,N]
            "d": d_sal.reshape(b, m, n),
        }
        return tokens.reshape(b, m, self.n_cells, self.decoder_dim), aux
