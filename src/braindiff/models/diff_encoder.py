"""S3 difference-module pretraining.

Self-supervised pretraining of a DiffEncoder on longitudinal MRI pairs, with no report
or LM supervision. A Reconstructor supplies the pretext task (rebuild the follow-up
embedding from the baseline plus delta) and is discarded after S3; only the DiffEncoder
and connector.delta carry into S4.

The front-end is a DeltaDiffCaptioner built with vision_only=True, so no decoder exists
and S3 and S4 share one implementation of the image path.

Inputs are template-space NeuroVFM tokens on a fixed 12x14x12 = 2016 grid.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed modality order (matches the MultiModal model's lowercase file tokens).
MODALITIES = ["T1w", "T1ce", "T2w", "FLAIR"]

# NeuroVFM token grid for a 48x224x192 volume at 4x16x16 patches. Every volume is
# template-space with remove_background=False, so this is the same for all of them.
TOKEN_GRID = (12, 14, 12)


def _pick_heads(dim: int) -> int:
    """Pick an attention head count that divides `dim` (target ~64-dim heads)."""
    for h in [dim // 64, 40, 32, 16, 10, 8, 5, 4, 2, 1]:
        if h >= 1 and dim % h == 0:
            return h
    return 1


def neighbour_index(grid=TOKEN_GRID, window: int = 1):
    """Static (idx [N, K], valid [N, K]) for a +/-`window` token neighbourhood.

The grid is fixed, so this is a constant of the architecture rather than something to
recompute per sample. Token order is row-major over (d, h, w) with w fastest, matching
neurovfm_transforms.tokenize_volume_fast. Out-of-volume neighbours are clamped to a
valid index and flagged False, so the gather never reads out of bounds.
    """
    nd, nh, nw = grid
    base = torch.stack(torch.meshgrid(
        torch.arange(nd), torch.arange(nh), torch.arange(nw), indexing="ij",
    ), dim=-1).reshape(-1, 3)                                       # [N, 3]

    r = torch.arange(-window, window + 1)
    offs = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)

    nb = base[:, None, :] + offs[None, :, :]                        # [N, K, 3]
    hi = torch.tensor(grid) - 1
    valid = ((nb >= 0) & (nb <= hi)).all(dim=-1)                    # [N, K]
    nb = torch.minimum(torch.maximum(nb, torch.zeros(3, dtype=torch.long)), hi)
    idx = nb[..., 0] * (nh * nw) + nb[..., 1] * nw + nb[..., 2]     # [N, K]
    return idx, valid


class WindowedCrossAttention(nn.Module):
    """Each main token attends to the ref tokens in its +/-1 neighbourhood.

Volumes are affinely normalized to a template, not rigidly registered pairwise, so a few
millimetres of shift moves an 8 mm lesion across a 16 mm token boundary and a strict
elementwise difference subtracts different anatomy. Windowed rather than global: at 2016
tokens per volume global cross-attention is quadratic and blurs focal change across
unrelated anatomy.
    """

    def __init__(self, dim: int, attn_dim: int = 512, heads: int = None,
                 dropout: float = 0.0):
        super().__init__()
        heads = heads or _pick_heads(attn_dim)
        assert attn_dim % heads == 0
        self.heads, self.head_dim = heads, attn_dim // heads
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, attn_dim, bias=False)
        self.to_kv = nn.Linear(dim, attn_dim * 2, bias=False)
        self.to_out = nn.Linear(attn_dim, dim, bias=False)
        self.dropout_p = dropout

    def forward(self, main, ref, idx, valid):
        """main/ref [B, N, D]; idx/valid [N, K] -> [B, N, D]."""
        b, n, _ = main.shape
        k_n = idx.shape[1]

        q = self.to_q(self.norm_q(main))                            # [B, N, A]
        k, v = self.to_kv(self.norm_kv(ref)).chunk(2, dim=-1)

        # Gather the neighbourhood. This is the module's memory cost: k/v become
        # K x larger. It sets the S3 batch size -- measure before raising it.
        flat = idx.reshape(-1)
        k = k[:, flat].view(b, n, k_n, self.heads, self.head_dim)
        v = v[:, flat].view(b, n, k_n, self.heads, self.head_dim)

        # Attention written out rather than via scaled_dot_product_attention.
        # SDPA would need (B*N) folded into the batch axis, and its memory-efficient
        # kernel hard-caps batch at 65535 -- at 2016 tokens that caps a call at 32
        # volumes, i.e. batch 8 with 4 modalities, which is a limit on the training
        # config imposed by a kernel detail. K is only 27, so the explicit
        # [B, N, K, heads] score tensor is small and this costs nothing.
        q = q.view(b, n, self.heads, self.head_dim)
        scores = torch.einsum("bnhd,bnkhd->bnkh", q, k) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid[None, :, :, None], float("-inf"))
        w = torch.softmax(scores, dim=2)
        if self.training and self.dropout_p:
            w = F.dropout(w, self.dropout_p)

        out = torch.einsum("bnkh,bnkhd->bnhd", w, v)
        return self.to_out(out.reshape(b, n, -1))


class DiffEncoder(nn.Module):
    """Compress (embed_A, embed_B) into `delta`, a low-dimensional representation of what
changed, while resisting re-encoding embed_B's identity.

The difference is local on the token grid. Windowed cross-attention absorbs residual
misregistration; the pointwise term stays as a residual so exact alignment remains the
default and attention only corrects it. There is no global self-attention over the diff
sequence: connector.delta already mixes globally on this tensor at O(Q*N), and its output
is what reaches the decoder. `local_attn_layers > 0` restores a windowed version as an
ablation.

Neighbourhoods never cross modality boundaries: the input is reshaped to [B*M, N, D]
internally and folded back to [B, M*N, D] on the way out, because the caller's token mask
(`mask_ref & mask_main`) is flat over M*N.
    """

    def __init__(self, feature_dim: int, bottleneck_dim: int = 128,
                 local_attn_layers: int = 0, heads: int = None, metric_dim: int = 32,
                 dropout: float = 0.1, attn_dim: int = 512,
                 grid=TOKEN_GRID, window: int = 1):
        super().__init__()
        self.feature_dim = feature_dim
        self.sqrt_d = math.sqrt(feature_dim)
        self.tokens_per_volume = grid[0] * grid[1] * grid[2]

        # Static neighbourhood: a constant of the architecture, not per-sample state.
        idx, valid = neighbour_index(grid, window)
        self.register_buffer("nb_idx", idx, persistent=False)
        self.register_buffer("nb_valid", valid, persistent=False)

        # Step 1: windowed cross-attention correcting the pointwise difference.
        self.cross_attn = WindowedCrossAttention(feature_dim, attn_dim, heads, dropout)

        # Step 2: light relational features from the three near-redundant scalars.
        self.metric_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, metric_dim),
        )

        # Step 3 (OFF by default): windowed self-attention over the diff sequence.
        # Kept as a flag rather than deleted so the "does local context on the diff
        # help?" question stays answerable, but it is not in the default path.
        self.local_attn = nn.ModuleList(
            WindowedCrossAttention(feature_dim, attn_dim, heads, dropout)
            for _ in range(local_attn_layers)
        )

        # Step 4: fuse a MULTI-CHANNEL, DIRECTIONAL difference with the metric
        # features. All channels are derived from the corrected signed difference, so
        # none is redundant with the ref/main scan blocks the decoder already reads
        # (a generic co-attention over both timepoints would be):
        #   signed diff (B-Â) | magnitude |B-Â| | relu(B-Â) appeared | relu(Â-B) gone
        # The rectified halves give the linear fuse components it cannot form from the
        # signed channel alone, and the appeared/disappeared split is aligned with
        # New-lesion vs Resolved / Progressed vs Improved. cos distance rides in the
        # metric scalars. Held to a fixed bar: must beat raw (B-A) on the probe or
        # revert to the single signed channel.
        self.n_diff_channels = 4
        self.fusion = nn.Sequential(
            nn.Linear(self.n_diff_channels * feature_dim + metric_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-token salience gate. KEPT so the unsupervised stage (DiffPretrainModel)
        # is unchanged, but it is NOT applied to the read tensor: `fused` is returned
        # ungated so change magnitude survives into the perceiver. The supervised
        # stage supervises the POST-perceiver tokens (what S4 reads), not the gate.
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1),
        )

        # bottleneck/project DELETED. The 768->128->768 squeeze (with LayerNorm) was
        # scale-invariant -- a disguised collapse pressure that normalised change
        # magnitude away before the perceiver. `fused` is the read tensor now.
        # `bottleneck_dim` stays in the signature (curriculum forwards it) but is unused.

    def forward(self, embed_A: torch.Tensor, embed_B: torch.Tensor):
        """embed_A/B [B, M*N, D] -> delta [B, M*N, D] (A = ref/prior, B = main)."""
        b, total, d = embed_A.shape
        n = self.tokens_per_volume
        if total % n:
            raise ValueError(
                f"got {total} tokens, not a multiple of {n} — DiffEncoder expects "
                f"whole {n}-token volumes concatenated along the token axis."
            )
        m = total // n

        # Per modality, so a token's neighbourhood is spatial rather than reaching
        # into a different series that happens to sit beside it in the flat layout.
        a = embed_A.reshape(b * m, n, d)
        bb = embed_B.reshape(b * m, n, d)

        # Step 1: pointwise difference + windowed correction for residual
        # misregistration. Pointwise is the residual, so perfect alignment is the
        # default behaviour and attention only has to model the shift.
        diff = (bb - a) + self.cross_attn(bb, a, self.nb_idx, self.nb_valid)

        # Step 2: per-token metric scalars -> small projection.
        l2 = (diff.norm(dim=-1, keepdim=True)) / self.sqrt_d            # [B*M, N, 1]
        cos = F.cosine_similarity(a, bb, dim=-1).unsqueeze(-1)
        dot = (a * bb).sum(dim=-1, keepdim=True) / self.sqrt_d
        metric_features = self.metric_mlp(torch.cat([l2, cos, dot], dim=-1))

        # Step 3: optional windowed self-attention over the diff sequence. Empty by
        # default -- Perceiver_delta is the global mixer on this path.
        diff_contextualized = diff
        for layer in self.local_attn:
            diff_contextualized = diff_contextualized + layer(
                diff_contextualized, diff_contextualized, self.nb_idx, self.nb_valid)

        # Step 4: multi-channel directional difference (embed_B excluded to keep this
        # about change) -> fuse with the metric features. dctx is the corrected
        # signed difference; the four channels are signed / magnitude / appeared /
        # disappeared.
        dctx = diff_contextualized
        channels = torch.cat(
            [dctx, dctx.abs(), F.relu(dctx), F.relu(-dctx), metric_features], dim=-1)
        fused = self.fusion(channels)                                  # [B*M, N, D]

        # Salience gate: computed for the unsupervised stage, NOT applied to `fused`.
        gate = torch.sigmoid(self.gate_mlp(fused))                     # [B*M, N, 1]

        # `fused` IS the delta: no bottleneck/project, so change magnitude is
        # preserved into the perceiver. Index 0 is what captioner._assemble_visual and
        # the supervised objective read.
        #
        # Fold modalities back onto the token axis: the caller's mask
        # (mask_ref & mask_main) is flat over M*N and must keep indexing the token
        # it names. A mismatch here does not raise -- it silently trains on the
        # wrong mask -- so the ordering is asserted in the tests.
        unflatten = lambda t: t.reshape(b, total, t.shape[-1])
        return (unflatten(fused), unflatten(gate),
                unflatten(diff_contextualized), unflatten(fused))


class Reconstructor(nn.Module):
    """S3-only pretext decoder: reconstruct embed_B from embed_A conditioned on delta (and
vice-versa with swapped args). FiLM conditioning forces delta to be used at every layer.
Discarded after S3.
    """

    def __init__(self, feature_dim: int, layers: int = 3, dropout: float = 0.1):
        super().__init__()

        # `delta` arrives already at feature_dim (projected inside DiffEncoder).

        # Step 2: FiLM-modulated transformation layers.
        self.film = nn.ModuleList(
            nn.Linear(feature_dim, feature_dim * 2) for _ in range(layers)
        )
        self.transform = nn.ModuleList(
            nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
            for _ in range(layers)
        )

        # Step 3: output projection.
        self.output = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Step 4: per-token residual gate (NOT pooled over tokens).
        self.residual_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(self, embed_A: torch.Tensor, delta: torch.Tensor):
        # delta is already at feature_dim (projected inside DiffEncoder).
        h = embed_A
        for film, transform in zip(self.film, self.transform):
            scale, shift = film(delta).chunk(2, dim=-1)
            h = transform(h)
            h = h * (1 + scale) + shift

        reconstruction = self.output(h)                                # [B, N, D]

        # Per-token blend: high-delta tokens lean on full reconstruction, near-zero
        # delta tokens lean on the cheap residual path.
        residual_weight = torch.sigmoid(self.residual_mlp(delta))      # [B, N, 1]
        embed_B_pred = (
            residual_weight * (embed_A + delta)
            + (1 - residual_weight) * reconstruction
        )
        return embed_B_pred, residual_weight


class DiffPretrainModel(nn.Module):
    """Container for S3 training: frozen vision front-end + DiffEncoder + Reconstructor + the
trainable connector.delta.

Trainable: `diff_encoder`, `reconstructor`, `captioner.connector.delta`. Frozen: the
encoder, `connector.scan`, and `connector.proj` -- proj is SHARED with the scan branch, so
training it would shift the ref/main blocks' input distribution at S4.
    """

    def __init__(self, bottleneck_dim: int = 128, num_modalities: int = 4,
                 local_attn_layers: int = 0, dropout: float = 0.1,
                 disc_temperature: float = 0.1, disc_negatives: int = 2,
                 attn_dim: int = 512, num_queries: int = 64, device: str = "cuda",
                 vision_lora_r: int = 16, vision_lora_alpha: int = 32):
        super().__init__()
        # Deferred: braindiff.models.captioner imports DiffEncoder from this module.
        from braindiff.models.captioner import DeltaDiffCaptioner_Qwen3

        self.disc_temperature = disc_temperature
        # Each negative is another pass through the Reconstructor, so this
        # multiplies that module's cost by (K+1). Lower the S3 batch size before
        # raising K.
        self.disc_negatives = disc_negatives

        # include_delta=True so the captioner builds BOTH modules S3 pretrains:
        # `diff_encoder` and `connector.delta`. S3 trains the captioner's own
        # instances rather than parallel copies, so the checkpoint keys are already
        # the ones S4 loads and there is nothing to remap.
        # LoRA is BUILT (frozen), not skipped. S1-S2 train the encoder's LoRA, and
        # get_peft_model renames every encoder key -- building without it here
        # matches 0 of the checkpoint's 184 encoder tensors and silently falls back
        # to the bare HF weights, discarding all curriculum adaptation. `r` must
        # match the stage that wrote the checkpoint or the shape filter drops it.
        self.captioner = DeltaDiffCaptioner_Qwen3(
            vision_only=True, single_timepoint=False, include_delta=True,
            use_vision_lora=True,
            vision_lora_r=vision_lora_r, vision_lora_alpha=vision_lora_alpha,
            use_lora=False, num_queries=num_queries,
            # S3 always inherits S2's connector, so never re-init from the
            # released weights -- that would discard two stages of adaptation.
            pretrained_connector=False,
            delta_attn_dim=attn_dim, delta_local_attn_layers=local_attn_layers,
            device=device,
        )
        feature_dim = self.captioner.vision_dim          # encoder width (768)
        self.reconstructor = Reconstructor(feature_dim=feature_dim, dropout=dropout)

        # Heads for L_compress. Separate per side: the compressed and dense deltas
        # live at different scales, and forcing a shared map would make the loss
        # partly about matching norms rather than about retained content.
        self.compress_head = nn.Linear(feature_dim, 256, bias=False)
        self.dense_head = nn.Linear(feature_dim, 256, bias=False)
        self.freeze_vision()

    @property
    def change_map(self):
        return self.captioner.change_map

    def load_pretrained_vision(self, state_dict: dict, label="S2 checkpoint",
                               is_main=True):
        """Warm-start the vision stack from the S2 checkpoint, then re-freeze."""
        from braindiff.training.checkpoint import load_stage_checkpoint
        kept = load_stage_checkpoint(self.captioner, state_dict, label=label,
                                     is_main=is_main)
        self.freeze_vision()
        return kept

    def freeze_vision(self):
        """Freeze the encoder, connector.scan and the shared projection; leave connector.delta and
the DiffEncoder trainable.

`connector.proj` is deliberately frozen: it is SHARED with the scan branch, so training it
here would shift the ref/main blocks' input distribution at S4.
        """
        from braindiff.training.freeze import apply_trainable
        apply_trainable(self.captioner, ("change_map",), is_main=False)

    def delta_state_dict(self):
        """The module that carries into S4 (the ChangeMapEncoder), keyed exactly as the
        captioner's own state dict, so S4's load_state_dict(..., strict=False) picks it
        up with no remapping."""
        return {f"change_map.{k}": v
                for k, v in self.captioner.change_map.state_dict().items()}

    def encode(self, tokens, coords, present):
        """Frozen encoder pass -> (feats [B, M*N, Dv], mask [B, M*N]).

encode_multimodal keeps the modality axis (the connector runs per series), so flatten it
back here: the DiffEncoder and every S3 loss are defined on the flat 4*2016 grid.
        """
        with torch.no_grad():
            self.captioner.vision_encoder.eval()
            feats = self.captioner.encode_multimodal(tokens, coords, present)
        b, m, n, d = feats.shape
        mask = present.unsqueeze(-1).expand(b, m, n)
        return feats.reshape(b, m * n, d), mask.reshape(b, m * n)

    def forward(self, batch_ref, batch_main, is_dup):
        """
        Args:
            batch_ref/main: (tokens [B,M,N,1024], coords [B,M,N,3], present [B,M])
            is_dup:         [B] bool — augmented-duplicate pairs (true delta ~ 0)
        Returns:
            l_recon, l_norm, l_gate, l_antisym, l_disc (unweighted scalars)
        """
        embed_A, mask_A = self.encode(*batch_ref)                      # [B, M*N, Dv]
        embed_B, mask_B = self.encode(*batch_main)
        joint = mask_A & mask_B

        # Forward A->B and backward B->A share the same weights.
        delta_fwd, gate_fwd, _, _ = self.diff_encoder(embed_A, embed_B)
        embed_B_pred, _ = self.reconstructor(embed_A, delta_fwd)

        delta_bwd, gate_bwd, _, _ = self.diff_encoder(embed_B, embed_A)
        embed_A_pred, _ = self.reconstructor(embed_B, delta_bwd)

        # L_recon: real pairs only (duplicates have no meaningful target change).
        real = (~is_dup).to(embed_A.dtype)                             # [B]
        mse_B = ((embed_B_pred - embed_B) ** 2).mean(dim=[1, 2])       # [B]
        mse_A = ((embed_A_pred - embed_A) ** 2).mean(dim=[1, 2])
        recon_per = 0.5 * (mse_B + mse_A)
        l_recon = (recon_per * real).sum() / real.sum().clamp(min=1.0)

        # L_norm / L_gate: DUPLICATE ROWS ONLY.
        #
        # These were previously averaged over the whole batch, which asked every
        # real pair to have a small, mostly-gated-off delta. Combined with the
        # reconstructor's residual path, delta == 0 is then a global optimum, and
        # that is exactly what the S3 logs show happening: ||delta|| fell 32.4 ->
        # 1.32 and the gate to 0.0069, i.e. the module learned "nothing changed"
        # before it ever saw a report. A duplicate pair is the only case where a
        # near-zero delta is the correct answer, so that is the only place these
        # penalties belong.
        dup = is_dup.to(embed_A.dtype)                                 # [B]
        dup_n = dup.sum().clamp(min=1.0)
        norm_per = 0.5 * (delta_fwd.norm(dim=-1).mean(dim=1) + delta_bwd.norm(dim=-1).mean(dim=1))
        l_norm = (norm_per * dup).sum() / dup_n
        gate_per = 0.5 * (gate_fwd.mean(dim=[1, 2]) + gate_bwd.mean(dim=[1, 2]))
        l_gate = (gate_per * dup).sum() / dup_n

        # L_antisym: forward and backward deltas should be near-negatives of each
        # other (swapping the query direction negates the change).
        l_antisym = ((delta_fwd + delta_bwd) ** 2).mean()

        # L_disc: this sample's own delta must reconstruct B better than another
        # sample's delta does, holding A fixed.
        #
        # Reconstruction MSE alone is satisfied by delta ~ 0 whenever B ~ A, which
        # is most of this corpus. The obvious contrastive fix -- rank embed_B_pred
        # against every other sample's embed_B -- does not work: different patients
        # have different anatomy, so embed_A alone already identifies embed_B and
        # the task is solved at delta == 0.
        #
        # Substituting another sample's delta into the SAME embed_A removes that
        # shortcut. Anatomy is now constant across the candidates and only the
        # change signal differs, so the loss can only be driven down by a delta
        # that actually encodes change. If every delta collapsed to zero the
        # candidates would be identical and this term would sit at log(K+1).
        b = embed_A.shape[0]
        if b > 1:
            k = min(self.disc_negatives, b - 1)
            # Offsets in 1..b-1 keep the wrap-around index off the diagonal.
            offsets = torch.randint(1, b, (b, k), device=embed_A.device)
            neg_idx = (torch.arange(b, device=embed_A.device).unsqueeze(1) + offsets) % b

            cand_delta = torch.cat([delta_fwd.unsqueeze(1),
                                    delta_fwd[neg_idx]], dim=1)        # [B, K+1, N, D]
            n_cand = cand_delta.shape[1]
            flat_A = embed_A.unsqueeze(1).expand(-1, n_cand, -1, -1).flatten(0, 1)
            pred_all, _ = self.reconstructor(flat_A, cand_delta.flatten(0, 1))
            pred_all = pred_all.view(b, n_cand, *embed_B.shape[1:])

            dist = ((pred_all - embed_B.unsqueeze(1)) ** 2).mean(dim=[2, 3])  # [B, K+1]
            logits = -dist / self.disc_temperature
            disc_per = F.cross_entropy(
                logits, torch.zeros(b, dtype=torch.long, device=logits.device),
                reduction="none")
            l_disc = (disc_per * real).sum() / real.sum().clamp(min=1.0)
            disc_acc = (logits.argmax(dim=1) == 0).float().mean()
        else:
            l_disc = embed_A.new_zeros(())
            disc_acc = embed_A.new_zeros(())

        # L_compress: the ONLY term that reaches Perceiver_delta.
        #
        # Everything above operates on the dense 2016-token delta, so before this
        # existed the delta resampler received no gradient at S3 at all -- which
        # made the whole stage fail at its stated purpose.
        #
        # Why not simply move L_disc onto the compressed delta, as first planned:
        # that term works by substituting another sample's delta into the SAME
        # embed_A and re-running the Reconstructor, and the Reconstructor FiLMs
        # per-token over N=2016. It cannot consume 64 latents, and there is no
        # inverse of the Perceiver to expand them back. So the retrieval is
        # expressed directly between the compressed and dense deltas instead.
        #
        # The requirement is preservation: whatever distinguishes this pair's change
        # from another pair's must survive the 31x bottleneck. If Perceiver_delta
        # collapsed to a constant, every z would be identical and this term would
        # sit at log(B). That is the concrete anti-collapse property; chance
        # accuracy is 1/B.
        # PER SERIES, exactly as S4 calls it. The connector is per-modality now:
        # each resampler call sees one modality's 2016 tokens, never a concat over
        # all four. Compressing the flat 8064-token grid here instead would train
        # connector.delta on an input distribution S4 never produces -- which is
        # precisely the drift the two-resampler design exists to avoid, and the
        # whole reason S3 pretrains this module rather than S4 learning it cold.
        # Absent modalities contribute no call at all (a resampler with zero keys
        # is a -inf softmax, i.e. NaN), so latents are averaged over present series.
        m = self.captioner.num_modalities
        n = delta_fwd.shape[1] // m
        delta_s = delta_fwd.view(b, m, n, -1)
        pres = joint.view(b, m, n)[:, :, 0]                            # [B, M] bool
        lat = delta_fwd.new_zeros(b, m, self.captioner.num_queries, delta_fwd.shape[-1])
        for mi in range(m):
            idx = pres[:, mi].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            out = self.captioner.connector.delta(delta_s[idx, mi])     # [k, Q, Dv]
            lat[:, mi] = lat[:, mi].index_copy(0, idx, out)
        z = lat.sum(dim=1) / pres.sum(dim=1).clamp(min=1)[:, None, None]  # [B, Q, Dv]
        z = F.normalize(self.compress_head(z.mean(dim=1)), dim=-1)
        w = joint.unsqueeze(-1).to(delta_fwd.dtype)
        d = (delta_fwd * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)   # masked mean
        d = F.normalize(self.dense_head(d), dim=-1)

        if b > 1:
            sim = (z @ d.t()) / self.disc_temperature                  # [B, B]
            tgt = torch.arange(b, device=sim.device)
            comp_per = F.cross_entropy(sim, tgt, reduction="none")
            l_compress = (comp_per * real).sum() / real.sum().clamp(min=1.0)
            compress_acc = (sim.argmax(dim=1) == tgt).float().mean()
        else:
            l_compress = embed_A.new_zeros(())
            compress_acc = embed_A.new_zeros(())

        # Collapse diagnostics: ||delta|| fell 32.4 -> 1.32 and the gate to 0.0069
        # on the last S3 run, so these are watched on val, not just logged.
        stats = {
            "disc_acc": disc_acc.detach(),
            "compress_acc": compress_acc.detach(),
            "delta_norm": delta_fwd.norm(dim=-1).mean().detach(),
            "gate_mean": gate_fwd.mean().detach(),
        }
        return l_recon, l_norm, l_gate, l_antisym, l_disc, l_compress, stats
