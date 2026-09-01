"""S3 pretrain for the ChangeMapEncoder -- fully LABEL-FREE.

The old delta pooled away location (64 non-positional latents) while the decoder already
gets both timepoints plus the prior report. The ChangeMapEncoder (models/change_map.py)
instead computes cross-timepoint correspondence and emits a coordinate-tagged 4x4x4 change
map. Three label-free terms train it -- no 7-way labels, no masks:

1. L_contrast -- the dense change field must let the Reconstructor rebuild embed_B from
   embed_A better than the same field with a spatial block spliced in from another sample.
   Anti-collapse and token-local.

2. L_map -- the coarse map, which is what S4 reads, must stay pair-discriminative: InfoNCE
   between the pooled 64-cell map and the pooled dense field.

3. L_saliency -- each token's gate tracks the per-token cosine feature distance on the
   12x14x12 grid; duplicate pairs (one scan augmented twice) are pushed to zero gate, which
   is what separates change from scanner nuisance without labels. Deliberately not
   coarsened: a per-cell gate on the 4x4x4 map has a near-uniform target and localized at
   chance -- see _token_distance().

The trainer's 3-tuple contract is kept: l_disc=L_contrast, l_localize=L_saliency,
l_global=L_map, weighted by lambda_disc / lambda_gate_track / lambda_global.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from braindiff.models.diff_encoder import DiffPretrainModel

N_CHANGE_CLASSES = 7        # kept for the trainer's import; unused (label-free)
STABLE_CLASS = 0


class ChangeMapPretrainModel(DiffPretrainModel):

    def __init__(self, *args, spliced_disc=True, splice_frac=0.25,
                 dup_weight=0.5, stable_weight=0.25,
                 num_change_classes=N_CHANGE_CLASSES, **kw):
        super().__init__(*args, **kw)          # builds captioner.change_map, reconstructor,
        self.spliced_disc = spliced_disc       # compress_head, dense_head; calls freeze_vision
        self.splice_frac = splice_frac
        self.dup_weight = dup_weight            # per-row upweight of duplicate (no-change) rows
        self.stable_weight = stable_weight      # unused now (kept for ctor compat)

    def freeze_vision(self):
        """Train only `change_map`. The encoder, connector.scan and shared proj stay
        frozen (proj is shared with the scan branch)."""
        from braindiff.training.freeze import apply_trainable
        apply_trainable(self.captioner, ("change_map",), is_main=False)

    # ---- spatial splice negative (reused from the deltatune contrast) --------------

    def splice_negative(self, delta, generator=None):
        """delta with a random contiguous token block overwritten from another sample."""
        b, n, d = delta.shape
        if b < 2:
            return delta.clone()
        offsets = torch.randint(1, b, (b,), device=delta.device)
        src = (torch.arange(b, device=delta.device) + offsets) % b
        span = max(1, int(n * self.splice_frac))
        start = torch.randint(0, max(1, n - span + 1), (1,), device=delta.device).item()
        out = delta.clone()
        out[:, start:start + span] = delta[src][:, start:start + span]
        return out

    def _token_distance(self, d_tok, joint_bmn):
        """Per-TOKEN cosine-distance target [B,M,N] in [0,1], normalised per
        (sample, modality). Detached. NOT coarsened -- coarsening to 4x4x4 washes the
        sparse change signal into a near-uniform target (measured mean 0.76), which is
        why the per-cell gate localised at chance. The token grid keeps it sparse."""
        with torch.no_grad():
            w = joint_bmn.to(d_tok.dtype)
            dw = d_tok * w
            hi = dw.amax(dim=2, keepdim=True).clamp(min=1e-6)          # per (sample, modality)
            return (dw / hi).clamp(0.0, 1.0)

    # ---- forward -------------------------------------------------------------------

    def forward(self, batch_ref, batch_main, is_dup, change_label=None,
                has_global=None, change_alpha=None):
        """change_label/has_global/change_alpha are accepted for call-signature compat
        but UNUSED -- this objective is label-free."""
        embed_A, mask_A = self.encode(*batch_ref)                    # [B, M*N, 768]
        embed_B, mask_B = self.encode(*batch_main)
        joint = mask_A & mask_B                                      # [B, M*N]
        b, mn, dv = embed_A.shape
        m = self.captioner.num_modalities
        n = mn // m
        cmap = self.captioner.change_map

        _, aux = cmap(embed_A.reshape(b, m, n, dv), embed_B.reshape(b, m, n, dv))
        dense = aux["dense"].reshape(b, mn, cmap.change_dim)         # [B, M*N, 768]
        coarse = aux["coarse"]                                       # [B, M, n_cells, 768]
        gate = aux["gate"]                                          # [B, M, n_cells]
        d_tok = aux["d"]                                            # [B, M, N]

        real = (~is_dup).to(embed_A.dtype)
        real_n = real.sum().clamp(min=1.0)
        with torch.no_grad():
            recon_baseline = ((((embed_B - embed_A) ** 2).mean(dim=[1, 2])) * real).sum() / real_n

        # --- L_contrast: reconstructor + spatial splice on the dense change field ---
        if b > 1:
            cand = torch.stack([dense, self.splice_negative(dense)], dim=1)  # [B,2,M*N,768]
            n_cand = cand.shape[1]
            flat_A = embed_A.unsqueeze(1).expand(-1, n_cand, -1, -1).flatten(0, 1)
            pred, _ = self.reconstructor(flat_A, cand.flatten(0, 1))
            pred = pred.view(b, n_cand, *embed_B.shape[1:])
            dist = ((pred - embed_B.unsqueeze(1)) ** 2).mean(dim=[2, 3])
            logits = -dist / self.disc_temperature
            disc_per = F.cross_entropy(logits, torch.zeros(b, dtype=torch.long,
                                                           device=logits.device),
                                       reduction="none")
            l_contrast = (disc_per * real).sum() / real_n
            disc_acc = (logits.argmax(1) == 0).float().mean()
        else:
            l_contrast = embed_A.new_zeros(()); disc_acc = embed_A.new_zeros(())

        # --- L_map: coarse map (what S4 reads) stays pair-discriminative (InfoNCE) ---
        z = F.normalize(self.compress_head(coarse.mean(dim=[1, 2])), dim=-1)  # [B,256]
        w = joint.unsqueeze(-1).to(dense.dtype)
        d_pool = (dense * w).sum(1) / w.sum(1).clamp(min=1.0)                 # [B,768]
        dvec = F.normalize(self.dense_head(d_pool), dim=-1)
        if b > 1:
            sim = (z @ dvec.t()) / self.disc_temperature
            tgt = torch.arange(b, device=sim.device)
            l_map = (F.cross_entropy(sim, tgt, reduction="none") * real).sum() / real_n
            map_acc = (sim.argmax(1) == tgt).float().mean()
        else:
            l_map = embed_A.new_zeros(()); map_acc = embed_A.new_zeros(())

        # --- L_saliency: PER-TOKEN gate tracks the (sparse) per-token feature distance;
        #     duplicate pairs -> gate 0 (the only change-specific signal). ---
        target = self._token_distance(d_tok, joint.reshape(b, m, n))         # [B,M,N]
        w = joint.reshape(b, m, n).to(gate.dtype)                            # [B,M,N] presence
        g = gate.clamp(1e-6, 1 - 1e-6)                                       # gate is [B,M,N]
        per_tok = F.binary_cross_entropy(g, target, reduction="none") * w
        per_sample = per_tok.sum(dim=[1, 2]) / w.sum(dim=[1, 2]).clamp(min=1.0)
        l_track = (per_sample * real).sum() / real_n
        dup = is_dup.to(gate.dtype)
        dup_act = (gate * w).sum(dim=[1, 2]) / w.sum(dim=[1, 2]).clamp(min=1.0)
        l_dup = (dup_act * dup).sum() / dup.sum().clamp(min=1.0)
        l_saliency = l_track + self.dup_weight * l_dup

        z0 = embed_A.new_zeros(())
        stats = {
            "disc_acc": disc_acc.detach(),
            "delta_norm": dense.norm(dim=-1).mean().detach(),
            "gate_mean": gate.mean().detach(),
            "recon_baseline": recon_baseline.detach(),
            "global_acc": map_acc.detach(),                # reuse the slot for map InfoNCE acc
            "gate_track": l_track.detach(),
            "gate_dup": l_dup.detach(),
            "gate_stable": z0,
            "n_global": z0,
        }
        return l_contrast, l_saliency, l_map, stats
