"""DeltaDiffCaptioner_Qwen3 -- NeuroVFM encoder + DiffEncoder + per-series connector +
Qwen3-14B decoder.

Pipeline (S4, the full dual case; S1/S2 drop the ref and delta blocks):

  tokens [B,4,2016,1024] x2 timepoints
    -> NeuroVFMEncoder            [B,4,2016,768]   (+ modality_type_embedding)
    -> DiffEncoder (dense grid)   delta [B,4*2016,768]
    -> connector, PER SERIES      [B,12,64,5120]   (+ temporal_embedding per block)
    -> splice at <|image_pad|>    prefix [B,T,5120], left-padded
    -> Qwen3-14B                  logits_to_keep=C+1 -> nll [B,C]

Two consequences of going per-series (see models/connector.py):

  * 12 groups (3 blocks x 4 modalities), so up to 768 visual tokens.
  * Absent modalities are DROPPED, not zero-filled and masked -- a perceiver call with zero
    keys is a -inf softmax, i.e. NaN. So `n_images` varies per sample and the prompt varies
    with it (models/prompts.py::PromptTable).
"""
import math
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from braindiff.models.connector import VisionConnector, DECODER_DIM, VISUAL_DIM
from braindiff.models.diff_encoder import DiffEncoder          # kept for the legacy unsupervised S3 path
from braindiff.models.change_map import ChangeMapEncoder
from braindiff.models.encoder import NeuroVFMEncoder
from braindiff.models.prompts import MODALITIES

DELTA_BOTTLENECK = 128
CONTRASTIVE_DIM = 512
N_TOKENS = 2016


def l2norm(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, dim=-1, eps=1e-4)


def all_gather_batch(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """All-gather along `dim`, keeping gradient only through this rank's slice.

Avoids torch.distributed.nn.functional.all_gather, which registers a custom autograd
Function inside DDP's backward hooks and is fragile with find_unused_parameters=True.
    """
    if not dist.is_initialized():
        return x
    world_size, rank = dist.get_world_size(), dist.get_rank()
    x = x.contiguous()
    with torch.no_grad():
        gathered = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(gathered, x)
    gathered[rank] = x
    return torch.cat(gathered, dim=dim)


class DeltaDiffCaptioner_Qwen3(nn.Module):

    def __init__(
        self,
        # --- decoder ---
        use_lora: bool = False, lora_r: int = 16, lora_alpha: int = 32,
        lora_dropout: float = 0.05, attn_implementation: str = "flash_attention_2",
        use_gradient_checkpointing: bool = True,
        max_seq_len: int = 3072, max_caption_length: int = 512,
        max_prompt_length: int = 384,
        # --- vision encoder ---
        use_vision_lora: bool = True, vision_lora_r: int = 32,
        vision_lora_alpha: int = 64, vision_lora_dropout: float = 0.05,
        vision_lora_trainable: bool = True,
        # --- connector ---
        pretrained_connector: bool = True, num_queries: int = 64,
        # --- topology ---
        include_delta: bool = True, single_timepoint: bool = False,
        vision_only: bool = False,
        delta_attn_dim: int = 512, delta_local_attn_layers: int = 0,
        counterfactual_margin: float = 0.5, num_change_classes: int = 0,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.single_timepoint = single_timepoint
        self.include_delta = bool(include_delta and not single_timepoint)
        self.vision_only = vision_only
        self.num_queries = num_queries
        self.counterfactual_margin = counterfactual_margin
        self.max_seq_len = max_seq_len
        self.max_caption_length = max_caption_length
        self.max_prompt_length = max_prompt_length
        self.num_modalities = len(MODALITIES)
        self.n_blocks = 1 if single_timepoint else (3 if self.include_delta else 2)

        # === VISION ENCODER === frozen; LoRA BUILT at every stage regardless of
        # whether it trains -- get_peft_model renames all 184 keys, so a stage built
        # without it matches 0 of its predecessor's tensors and silently reverts to
        # bare HF weights. trainer/checkpoint.py exists because that happened.
        self.vision_encoder = NeuroVFMEncoder(
            freeze_backbone=True, use_lora=use_vision_lora, lora_r=vision_lora_r,
            lora_alpha=vision_lora_alpha, lora_dropout=vision_lora_dropout,
            lora_trainable=vision_lora_trainable)
        self.vision_dim = self.vision_encoder.output_dim          # 768

        # === DECODER ===
        if vision_only:
            self.decoder = None
            self.decoder_dim = DECODER_DIM
            self.image_pad_id = None
        else:
            from braindiff.models.decoder import Qwen3Decoder
            self.decoder = Qwen3Decoder(
                attn_implementation=attn_implementation,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout)
            self.decoder_dim = self.decoder.hidden_size           # 5120
            self.image_pad_id = self.decoder.image_pad_id
            # The sequence must fit. Prompt is bounded by max_prompt_length, each
            # present group expands one placeholder into num_queries embeddings.
            worst = (self.max_prompt_length
                     + self.n_blocks * self.num_modalities * (num_queries - 1)
                     + max_caption_length)
            assert worst <= max_seq_len, (
                f"worst-case sequence {worst} > max_seq_len {max_seq_len}: "
                f"prompt {max_prompt_length} + visual "
                f"{self.n_blocks * self.num_modalities * num_queries} + caption "
                f"{max_caption_length}")

        # === CONNECTOR === per-series resamplers + shared projection.
        # build_delta=False: the delta path is now the ChangeMapEncoder, which does its
        # own coarsening + projection, so no delta perceiver is built.
        self.connector = VisionConnector(
            visual_dim=self.vision_dim, decoder_dim=self.decoder_dim,
            num_queries=num_queries, build_delta=False)
        if pretrained_connector:
            from braindiff.models.connector_transfer import load_pretrained_connector
            load_pretrained_connector(self.connector)

        # === DELTA === ChangeMapEncoder: correspondence + coordinate-tagged change-map.
        # Replaces DiffEncoder + connector.delta. Emits [B, M, 64, decoder_dim] directly
        # (4x4x4 coarse cells per modality, coordinate-embedded), preserving the 4x64
        # block contract. See models/change_map.py for the redundancy rationale.
        if self.include_delta:
            self.change_map = ChangeMapEncoder(
                feature_dim=self.vision_dim, decoder_dim=self.decoder_dim)

        # === EMBEDDINGS ===
        self.temporal_embedding_ref = nn.Parameter(torch.randn(1, 1, self.decoder_dim) * 0.02)
        self.temporal_embedding_main = nn.Parameter(torch.randn(1, 1, self.decoder_dim) * 0.02)
        self.temporal_embedding_delta = nn.Parameter(torch.randn(1, 1, self.decoder_dim) * 0.02)
        self.modality_type_embedding = nn.Embedding(self.num_modalities, self.vision_dim)
        nn.init.normal_(self.modality_type_embedding.weight, mean=0.0, std=0.02)

        # === CONTRASTIVE HEADS === CLIP-style learned temperature
        self.logit_temperature = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.visual_proj = nn.Linear(self.decoder_dim, CONTRASTIVE_DIM)
        self.text_proj = nn.Linear(self.decoder_dim, CONTRASTIVE_DIM)

        # === CHANGE HEAD === off by default (num_change_classes=0)
        self.num_change_classes = num_change_classes
        if num_change_classes:
            self.change_head = nn.Sequential(
                nn.LayerNorm(self.decoder_dim),
                nn.Linear(self.decoder_dim, 512), nn.GELU(),
                nn.Linear(512, num_change_classes))

        self.to(torch.device(device))

    # ------------------------------------------------------------------ helpers

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.embed(ids)

    # ------------------------------------------------------------------ vision

    def encode_multimodal(self, tokens, coords, present):
        """[B,M,N,1024] -> feats [B,M,N,768].

Keeps the modality axis because the connector runs per series. The ViT runs only on samples
that actually have each modality.
        """
        b, m, n, _ = tokens.shape
        feats = tokens.new_zeros(b, m, n, self.vision_dim, dtype=torch.float32)
        for i in range(m):
            idx = present[:, i].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            enc = self.vision_encoder(tokens[idx, i], coords[idx, i])       # [k,N,768]
            enc = enc + self.modality_type_embedding(
                torch.tensor(i, device=tokens.device, dtype=torch.long))
            feats[:, i] = feats[:, i].index_copy(0, idx, enc)
        return feats

    def _encode_timepoints(self, batch_ref, batch_main):
        """Run the tower once per timepoint. Split out so the counterfactual can
        rebuild from a permuted prior without a second (very expensive) ViT pass."""
        main = self.encode_multimodal(*batch_main)
        ref = None if self.single_timepoint else self.encode_multimodal(*batch_ref)
        return ref, main

    def _assemble_visual(self, f_ref, f_main, present_ref, present_main):
        """Per-series resample + project + temporal tag.

        Returns:
            chunks        [B, G, Q, D]  G = n_blocks * 4, block-major (ref, main, delta)
            block_present [B, G] bool   -- must equal the prompt's placeholder pattern
        """
        b = f_main.shape[0]
        m, q, d = self.num_modalities, self.num_queries, self.decoder_dim
        dev = f_main.device

        blocks = []
        if self.single_timepoint:
            blocks.append(("main", f_main, present_main, self.temporal_embedding_main))
        else:
            blocks.append(("ref", f_ref, present_ref, self.temporal_embedding_ref))
            blocks.append(("main", f_main, present_main, self.temporal_embedding_main))
            if self.include_delta:
                joint = present_ref & present_main
                # ChangeMapEncoder emits the delta block ALREADY resampled + projected:
                # [B, M, 64, decoder_dim], coordinate-tagged. Unlike the scan blocks it
                # does NOT go through the connector below (is_delta branch in the loop).
                cmap, _ = self.change_map(f_ref, f_main)               # [B, M, 64, D]
                blocks.append(("delta", cmap, joint, self.temporal_embedding_delta))

        chunks = torch.zeros(b, self.n_blocks * m, q, d, device=dev,
                             dtype=self.connector.proj.fc1.weight.dtype)
        block_present = torch.zeros(b, self.n_blocks * m, dtype=torch.bool, device=dev)

        for bi, (name, feats, pres, temb) in enumerate(blocks):
            is_delta = (name == "delta")
            for mi in range(m):
                idx = pres[:, mi].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                if is_delta:
                    # ChangeMap already produced [k, 64, D]; no connector call.
                    out = feats[idx, mi]
                else:
                    # One perceiver call per (scan block, modality): exactly one
                    # modality's 2016 tokens, so no padded keys and no mask.
                    out = self.connector(feats[idx, mi], is_delta=False)     # [k,Q,D]
                g = bi * m + mi
                chunks[:, g] = chunks[:, g].index_copy(0, idx, out + temb)
                block_present[idx, g] = True
        return chunks, block_present

    def _splice_prefix(self, prompt_ids, prompt_attn, chunks, block_present):
        """Replace each <|image_pad|> with its 64-token chunk; LEFT-pad the result.

The mismatch ValueError is deliberate: it is the only guard against a prompt/presence
disagreement, which is otherwise a silent misalignment between what the text says and what
the model sees. LEFT padding is load-bearing -- _caption_nll's logits_to_keep assumes the
supervised span is the contiguous tail.
        """
        b = prompt_ids.shape[0]
        text_embeds = self._embed_tokens(prompt_ids)
        parts_e, parts_a = [], []
        for i in range(b):
            pos = (prompt_ids[i] == self.image_pad_id).nonzero(as_tuple=True)[0]
            groups = block_present[i].nonzero(as_tuple=True)[0]
            if len(pos) != len(groups):
                raise ValueError(
                    f"sample {i}: {len(pos)} <|image_pad|> placeholders but "
                    f"{len(groups)} present visual groups")
            e, a, last = [], [], 0
            for j, p in enumerate(pos):
                e.append(text_embeds[i, last:p])
                a.append(prompt_attn[i, last:p])
                e.append(chunks[i, groups[j]].to(text_embeds.dtype))
                a.append(torch.ones(self.num_queries, dtype=prompt_attn.dtype,
                                    device=prompt_attn.device))
                last = p + 1
            e.append(text_embeds[i, last:])
            a.append(prompt_attn[i, last:])
            parts_e.append(torch.cat(e, 0))
            parts_a.append(torch.cat(a, 0))

        T = max(x.shape[0] for x in parts_e)
        prefix = text_embeds.new_zeros(b, T, text_embeds.shape[-1])
        attn = torch.zeros(b, T, dtype=prompt_attn.dtype, device=prompt_attn.device)
        for i, (e, a) in enumerate(zip(parts_e, parts_a)):
            prefix[i, -e.shape[0]:] = e
            attn[i, -a.shape[0]:] = a
        return prefix, attn

    # ------------------------------------------------------------------ losses

    def _caption_nll(self, prefix_embeds, prefix_attn, labels, attention_mask):
        """Per-token caption NLL, [B, C].

Kept per-token, not reduced, because three consumers need the shape: content-weighted CE
(weights applied before reduction), the counterfactual hinge (a per-sample mean), and the
content-token val CE (a token subset). HF's labels= path returns one scalar.

logits_to_keep = C+1 restricts the LM head to the supervised tail (+1 because predicting
the first caption token needs the hidden state one position earlier). Without it Qwen3
builds fp32 logits over the whole sequence, [B, L, 151936].
        """
        C = labels.shape[1]
        inputs_embeds = torch.cat([prefix_embeds, self._embed_tokens(labels)], dim=1)
        combined_attn = torch.cat([prefix_attn, attention_mask], dim=1)
        out = self.decoder(inputs_embeds=inputs_embeds.to(self.decoder.dtype_),
                           attention_mask=combined_attn,
                           logits_to_keep=C + 1, return_dict=True)
        logits = out.logits.float()[:, :-1, :]                              # [B, C, V]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1), reduction="none").view_as(labels)

    def _encode_sentences(self, sentence_ids, sentence_attn):
        b, s, ls = sentence_ids.shape
        ids = sentence_ids.reshape(b * s, ls)
        attn = sentence_attn.reshape(b * s, ls).unsqueeze(-1).float()
        emb = self._embed_tokens(ids).float()
        pooled = (emb * attn).sum(1) / attn.sum(1).clamp(min=1)
        return l2norm(self.text_proj(pooled))

    def _compute_logit_matrix(self, visual_tokens, text_latents_all, token_mask,
                              temp, chunk_size=64):
        """VL-CABS: each sentence query attends over one image's visual tokens,
        softmax-pools, and cosine-compares. Chunking is pure tiling."""
        device = visual_tokens.device
        N_T = text_latents_all.shape[0]
        visual_tokens = self.visual_proj(visual_tokens)
        visual_f32 = torch.nan_to_num(visual_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
        text_f32 = torch.nan_to_num(text_latents_all.float(), nan=0.0, posinf=0.0, neginf=0.0)
        visual_norm = l2norm(visual_f32)
        neg_mask = (~token_mask).unsqueeze(0)
        logits = torch.zeros(N_T, visual_f32.shape[0], device=device, dtype=torch.float32)
        for s0 in range(0, N_T, chunk_size):
            s1 = min(s0 + chunk_size, N_T)
            ct = text_f32[s0:s1]
            attn = torch.einsum('c d, i n d -> c i n', ct, visual_norm)
            attn = F.softmax(attn.masked_fill(neg_mask, -1e4), dim=-1)
            pooled = torch.einsum('c i n, i n d -> c i d', attn, visual_f32)
            logits[s0:s1] = torch.einsum('c i d, c d -> c i', l2norm(pooled), ct) * temp.float()
        return logits.clamp(-100.0, 100.0)

    def sentence_contrastive_loss(self, visual_tokens, token_mask, text_latents,
                                  sentence_mask, num_sentences_per_image):
        device = visual_tokens.device
        temp = self.logit_temperature.float().clamp(0.0, 4.0).exp()
        text_all = all_gather_batch(text_latents)
        mask_all = (all_gather_batch(sentence_mask.view(-1).float()).bool()
                    if sentence_mask is not None else None)
        local = self._compute_logit_matrix(visual_tokens, text_all, token_mask, temp)
        logits = all_gather_batch(local, dim=1).float()
        B_g, N_T = logits.shape[1], text_all.shape[0]
        s2i = torch.arange(B_g, device=device).repeat_interleave(num_sentences_per_image)
        membership = F.one_hot(s2i, num_classes=B_g).bool()

        pos = logits[torch.arange(N_T, device=device), s2i]
        neg_ok = ~membership
        if mask_all is not None:
            neg_ok = neg_ok & mask_all.view(N_T, 1).expand(N_T, B_g)
        masked = logits.clone()
        masked[~neg_ok] = -1e4
        lse_neg = torch.logsumexp(masked, dim=0)[s2i]
        loss_i_all = -pos + torch.logsumexp(torch.stack([pos, lse_neg], 1), 1)
        loss_t_all = F.cross_entropy(logits, s2i, reduction='none')
        if mask_all is not None:
            den = mask_all.float().sum().clamp(min=1.0)
            loss_t = torch.where(mask_all, loss_t_all, torch.zeros_like(loss_t_all)).sum() / den
            loss_i = torch.where(mask_all, loss_i_all, torch.zeros_like(loss_i_all)).sum() / den
        else:
            loss_t, loss_i = loss_t_all.mean(), loss_i_all.mean()
        return loss_t + loss_i

    # ------------------------------------------------------------------ forward

    def forward(self, tokens_main, coords_main, present_main,
                tokens_ref=None, coords_ref=None, present_ref=None,
                labels=None, attention_mask=None,
                prompt_table=None, prompt_ids=None, prompt_attn=None,
                sentence_input_ids=None, sentence_attn=None, sentence_mask=None,
                swap_perm=None, swap_valid=None,
                content_mask=None, token_weights=None):
        """Returns (loss, contrastive_loss, cf_loss, change_logits, content_loss).

        Only small scalars come back, so DDP's _DDPSink does not clone a large output.
        """
        dev = tokens_main.device
        f_ref, f_main = self._encode_timepoints(
            (tokens_ref, coords_ref, present_ref), (tokens_main, coords_main, present_main))
        chunks, block_present = self._assemble_visual(f_ref, f_main, present_ref, present_main)

        if prompt_ids is None:
            assert prompt_table is not None, "need prompt_ids or a PromptTable"
            prompt_ids, prompt_attn, table_bp = prompt_table.batch(
                present_main.tolist(),
                None if self.single_timepoint else present_ref.tolist(), device=dev)
            assert torch.equal(table_bp, block_present), \
                "PromptTable presence disagrees with the assembled visual blocks"
        prefix, prefix_attn = self._splice_prefix(prompt_ids, prompt_attn, chunks, block_present)

        # visual_tokens/token_mask for the contrastive term: flatten groups back to a
        # token axis, keeping BrainDiff's [B, n_visual, D] contract.
        b, G, Q, D = chunks.shape
        visual_tokens = chunks.reshape(b, G * Q, D)
        token_mask = block_present.unsqueeze(-1).expand(b, G, Q).reshape(b, G * Q)

        loss = contrastive_loss = cf_loss = content_loss = None
        change_logits = None

        if labels is not None:
            nll = self._caption_nll(prefix, prefix_attn, labels, attention_mask)   # [B,C]
            keep = attention_mask.float()
            w = keep if token_weights is None else keep * token_weights
            loss = (nll * w).sum() / w.sum().clamp(min=1.0)

            if content_mask is not None:
                cm = keep * content_mask.float()
                content_loss = (nll * cm).sum() / cm.sum().clamp(min=1.0)

            if swap_perm is not None:
                # Reuse the already-encoded features; only the prompt and the delta
                # change. PromptTable makes the swapped prompt free, so every swap
                # with a different classification is valid.
                #
                # WHICH block gets permuted differs by stage, and this is the whole
                # point of the term. Dual: swap the PRIOR, so the question is "does
                # the report still hold against a different baseline". Single: there
                # is no prior, so the only scan IS the swapped block -- swap `main`
                # and the question becomes plain grounding, "does this report still
                # hold against a different patient's scan". Guarding the whole branch
                # on `not single_timepoint` (as it was) indexes nothing at S2 and
                # leaves cf_loss None, so cf_weight, counterfactual_margin and the
                # per-step make_swap_perm call are all silently inert there.
                perm = swap_perm.to(dev)
                if self.single_timepoint:
                    cf_pres_main, cf_pres_ref = present_main[perm], None
                    cf_chunks, cf_bp = self._assemble_visual(
                        None, f_main[perm], None, cf_pres_main)
                else:
                    cf_pres_main, cf_pres_ref = present_main, present_ref[perm]
                    cf_chunks, cf_bp = self._assemble_visual(
                        f_ref[perm], f_main, cf_pres_ref, cf_pres_main)
                if prompt_table is not None:
                    cf_ids, cf_attn, _ = prompt_table.batch(
                        cf_pres_main.tolist(),
                        None if self.single_timepoint else cf_pres_ref.tolist(), device=dev)
                else:
                    cf_ids, cf_attn = prompt_ids, prompt_attn
                cf_prefix, cf_pattn = self._splice_prefix(cf_ids, cf_attn, cf_chunks, cf_bp)
                nll_swap = self._caption_nll(cf_prefix, cf_pattn, labels, attention_mask)
                per_true = (nll * keep).sum(1) / keep.sum(1).clamp(min=1.0)
                per_swap = (nll_swap * keep).sum(1) / keep.sum(1).clamp(min=1.0)
                hinge = F.relu(self.counterfactual_margin - (per_swap - per_true))
                v = swap_valid.float().to(dev)
                cf_loss = (hinge * v).sum() / v.sum().clamp(min=1.0)

        if sentence_input_ids is not None:
            text_latents = self._encode_sentences(sentence_input_ids, sentence_attn)
            contrastive_loss = self.sentence_contrastive_loss(
                visual_tokens, token_mask, text_latents, sentence_mask,
                sentence_input_ids.shape[1])

        if self.num_change_classes:
            pooled = (visual_tokens * token_mask.unsqueeze(-1)).sum(1) / \
                     token_mask.sum(1, keepdim=True).clamp(min=1)
            change_logits = self.change_head(pooled)

        return loss, contrastive_loss, cf_loss, change_logits, content_loss

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate_caption_batch(self, tokens_main, coords_main, present_main,
                               tokens_ref=None, coords_ref=None, present_ref=None,
                               prompt_table=None, prompt_ids=None, prompt_attn=None,
                               max_new_tokens=512, num_beams=1,
                               repetition_penalty=1.0, no_repeat_ngram_size=0,
                               min_new_tokens=5, **kw):
        # repetition_penalty MUST stay 1.0 for reports. It was 1.5, and radiology
        # prose is the worst case for it: a single reference report says "normal"
        # eight times ("observed as normal", "in normal configuration", "are
        # normal"). The penalty divides the logit of every token already emitted,
        # so each successive normality phrase is suppressed and the model, blocked
        # from writing the correct template, spends the probability mass on
        # invented pathology instead -- observed at S2 epoch 9 generating acute
        # infarcts and a nasopharyngeal mass for a scan whose reference reads
        # "mild cortical atrophy". Val CE was still improving that same epoch,
        # which is how the sampler was identified as the cause rather than the
        # model. test_overfit.py has always passed 1.0, which is why greedy
        # generation there reproduces targets verbatim.
        """Either a PromptTable (S2/S4) or explicit prompt_ids (S1, which has a
        per-row instruction the table cannot key on)."""
        dev = tokens_main.device
        f_ref, f_main = self._encode_timepoints(
            (tokens_ref, coords_ref, present_ref), (tokens_main, coords_main, present_main))
        chunks, bp = self._assemble_visual(f_ref, f_main, present_ref, present_main)
        if prompt_ids is not None:
            ids, attn = prompt_ids, prompt_attn
        else:
            assert prompt_table is not None, "need prompt_ids or a PromptTable"
            ids, attn, _ = prompt_table.batch(
                present_main.tolist(),
                None if self.single_timepoint else present_ref.tolist(), device=dev)
        prefix, pattn = self._splice_prefix(ids, attn, chunks, bp)
        out = self.decoder.generate(
            prefix, pattn, max_new_tokens=max_new_tokens, num_beams=num_beams,
            do_sample=False, repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens, early_stopping=(num_beams > 1), **kw)
        return [strip_think(t) for t in
                self.decoder.tokenizer.batch_decode(out, skip_special_tokens=True)]


def strip_think(text: str) -> str:
    """Drop a leading <think>...</think> block. No-op when absent.

The released config suppresses thinking with `/no_think` in the system prompt rather than
an empty block in the template, so a stray block is possible; scoring one as report text
would be a pure false negative.
    """
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
