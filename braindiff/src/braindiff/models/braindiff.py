"""Stage 2 comparison-report model, status-first.

  tokens [V, 2016, 1024] per series and timepoint (data/datasets.py)
    -> NeuroVFM blocks 1-12 + norm FROZEN; modality embedding + per-scan connector trainable   [V, 64, 5120]
    -> splice at <|image_pad|>: prior blocks, current blocks, edited prior report, CURRENT_UNKNOWN, instruction
    -> Qwen3-14B + LoRA (attention q/k/v/o, r 16, alpha 32; initialised from Stage 1-D)
    -> caption CE on '<status line>\\n<target report>'

split_status cuts the status line from every generation, so every scorer sees the report alone.
Weights: `init` = a Stage 1-D checkpoint for training, or load_checkpoint() with the release file or epochK.pt + `init`.
"""
import re

import torch
import torch.nn as nn

from braindiff.models.connector import VisionConnector, load_pretrained_connector
from braindiff.models.decoder import Qwen3Decoder
from braindiff.models.encoder import NeuroVFMEncoder, TopBlocks
from braindiff.models.prompts import MODALITIES, PriorReportPrompts
from braindiff.models.stage1 import Q, SpliceLM

TOP = tuple(f"encoder.model.blocks.{i}." for i in (9, 10, 11)) + ("encoder.model.norm.",)
GROUPS = {"encoder blocks 10-12 + norm": lambda k: k.startswith(TOP), "connector": lambda k: k.startswith("connector."),
          "modality embedding": lambda k: k.startswith("modality_type_embedding."),
          "decoder LoRA A": lambda k: k.startswith("decoder.") and "lora_A" in k,
          "decoder LoRA B": lambda k: k.startswith("decoder.") and "lora_B" in k}


def group_of(k):
    hit = [g for g, f in GROUPS.items() if f(k)]
    assert len(hit) == 1, (k, hit)
    return hit[0]


def copy_checked(model, sd, required, what, is_main=True):
    """Copy `sd` into `model`. Every key must exist in the model, each group in `required` must be complete, and
    every copied tensor is read back."""
    own = dict(model.named_parameters())
    unexpected = sorted(set(sd) - set(own))
    assert not unexpected, f"{what}: {len(unexpected)} keys not in the model, e.g. {unexpected[:5]}"
    n_sd, n_own = {}, {}
    for k in sd:
        n_sd[group_of(k)] = n_sd.get(group_of(k), 0) + 1
    for k in own:
        if any(f(k) for f in GROUPS.values()):
            n_own[group_of(k)] = n_own.get(group_of(k), 0) + 1
    for g in required:
        assert n_sd.get(g, 0) == n_own.get(g, 0) > 0, f"{what}: group '{g}' has {n_sd.get(g, 0)} / {n_own.get(g, 0)} tensors"
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(own[k].dtype))
    bad = [k for k, v in sd.items() if not torch.equal(own[k].detach().cpu(), v.to(own[k].dtype).cpu())]
    assert not bad, f"{what}: {len(bad)} tensors differ after the copy, e.g. {bad[:3]}"
    if is_main:
        print(f"[load] {what}: {len(sd)} tensors " + ", ".join(f"{g} {n_sd.get(g, 0)}/{n_own.get(g, 0)}" for g in GROUPS),
              flush=True)
    return n_sd


class BrainDiff(TopBlocks, SpliceLM, nn.Module):
    STATUS_RE = re.compile(r"^((?:(?:NEW|WORSENED|IMPROVED|RESOLVED)\.\s*)*No (?:other )?interval change\.)\s*\n?", re.I)

    def __init__(self, init=None, lora_r=16, lora_alpha=32, max_prompt=1536, max_report_tokens=384, is_main=True,
                 train_connector=True, status_first=True):
        super().__init__()
        self.status_first, self.last_status = status_first, None
        self.encoder = NeuroVFMEncoder()
        self.init_top_blocks()
        self.modality_type_embedding = nn.Embedding(len(MODALITIES), self.encoder.model.norm.weight.shape[0])
        self.connector = VisionConnector()
        load_pretrained_connector(self.connector, is_main=is_main)
        sd_dec = {}
        if init:
            # Stage 1-D checkpoint: vision path + decoder LoRA (copied once the decoder exists)
            sd = torch.load(init, map_location="cpu", weights_only=False)["model"]
            sd_dec = {k: v for k, v in sd.items() if k.startswith("decoder.")}
            sd = {k: v for k, v in sd.items() if not k.startswith("decoder.")}
            copy_checked(self, sd, ("encoder blocks 10-12 + norm", "connector", "modality embedding"),
                         f"vision path from {init}", is_main)
        self.decoder = Qwen3Decoder(use_lora=True, lora_r=lora_r, lora_alpha=lora_alpha)
        if init:
            copy_checked(self, sd_dec, ("decoder LoRA A", "decoder LoRA B"), f"decoder LoRA from {init}", is_main)
        self.image_pad_id = self.decoder.image_pad_id
        self.eot = self.decoder.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.prompts = PriorReportPrompts(self.decoder.tokenizer, max_prompt_length=max_prompt,
                                          max_report_tokens=max_report_tokens)
        self.train_connector = train_connector
        vis = ("connector.", "modality_type_embedding.")
        for n, p in self.named_parameters():
            p.requires_grad = "lora_" in n or (train_connector and n.startswith(vis))
        if is_main:
            tr = {n: p for n, p in self.named_parameters() if p.requires_grad}
            nv = sum(p.numel() for n, p in tr.items() if n.startswith(vis))
            print(f"[freeze] trainable {len(tr)} tensors {sum(p.numel() for p in tr.values()):,} params (LoRA"
                  f"{f' + connector / modality embedding {nv:,}' if train_connector else ''}); encoder frozen", flush=True)

    def trainable_keys(self):
        return {k for k, p in self.named_parameters() if p.requires_grad}

    def load_checkpoint(self, path, init=None, is_main=True, ck=None):
        """Release file (keys = trainable set + encoder blocks 10-12 + norm) or a training epochK.pt (keys = trainable
        set; the model must then have been built with its Stage 1-D `init`). Exact key sets, every group complete."""
        ck = ck if ck is not None else torch.load(path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        top = {k for k, _ in self.named_parameters() if k.startswith(TOP)}
        want = self.trainable_keys() | (top if set(sd) & top else set())
        missing, unexpected = sorted(want - set(sd)), sorted(set(sd) - want)
        assert not missing and not unexpected, f"{path}: missing {len(missing)} {missing[:3]}, unexpected " \
                                               f"{len(unexpected)} {unexpected[:3]}"
        assert set(sd) & top or init, f"{path} has no encoder blocks: build the model with its Stage 1-D init"
        req = [g for g in GROUPS if g != "encoder blocks 10-12 + norm" or set(sd) & top]
        copy_checked(self, sd, req, path, is_main)
        return ck

    def train(self, mode=True):
        super().train(mode)
        self.encoder.model.eval()
        self.connector.train(mode and self.train_connector)
        return self

    def series(self, tokens, mods):
        """tokens [S, 2016, 1024] -> connector outputs [S, 64, D]; no gradient through the encoder."""
        with torch.no_grad():
            f = self.encode(tokens)
        f = f + self.modality_type_embedding(mods)[:, None, :].to(f.dtype)
        return self.connector(f)

    def assemble(self, za, item_a, mods_a, zb, item_b, mods_b, P):
        """prior-slot series (za, item_a, mods_a) + current-slot series -> chunks [P, 8, 64, D], presence [P, 8]."""
        M = len(MODALITIES)
        out = za.new_zeros(P, 2 * M, Q, za.shape[-1])
        bp = torch.zeros(P, 2 * M, dtype=torch.bool, device=out.device)
        for z, it, md, t in ((za, item_a, mods_a, 0), (zb, item_b, mods_b, 1)):
            out[it, t * M + md] = z.to(out.dtype)
            bp[it, t * M + md] = True
        return out, bp

    def prefix(self, ch, bp, priors):
        M = len(MODALITIES)
        ids, attn, tbp = self.prompts.set_reports(priors).batch(bp[:, M:].tolist(), bp[:, :M].tolist(), device=ch.device)
        assert torch.equal(tbp, bp), "prompt placeholders disagree with the visual blocks"
        return self.splice(ids, attn, ch, bp)

    def feats(self, b):
        return self.series(b["ta"], b["mods_a"]), self.series(b["tb"], b["mods"])

    def visual(self, b):
        za, zb = self.feats(b)
        return self.assemble(za, b["item_a"], b["mods_a"], zb, b["item"], b["mods"], b["n_items"])

    def forward(self, b, priors):
        """priors: per-pair prior report text. -> (loss, stats dict)."""
        nll, keep = self.nll(*self.prefix(*self.visual(b), priors), b["text"])
        loss = (nll * keep).sum() / keep.sum().clamp(min=1)
        return loss, {"ce": float(loss)}

    def split_status(self, texts):
        """Cut the leading status line (kept in self.last_status, "" if none was written); no-op unless status_first."""
        if not self.status_first:
            return texts
        m = [self.STATUS_RE.match(t) for t in texts]
        self.last_status = [x.group(1) if x else "" for x in m]
        return [t[x.end():].strip() if x else t for t, x in zip(texts, m)]

    @torch.no_grad()
    def generate(self, b, priors, max_new_tokens=512, **gen_kw):
        """Greedy decoding; gen_kw (repetition_penalty) go to HF generate and apply to generated tokens only."""
        prefix, pattn = self.prefix(*self.visual(b), priors)
        out = self.decoder.generate(prefix, pattn, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=[self.eot, self.decoder.tokenizer.eos_token_id], **gen_kw)
        return self.split_status([t.strip() for t in self.decoder.tokenizer.batch_decode(out, skip_special_tokens=True)])

    @torch.no_grad()
    def caption_nll(self, b, priors):
        """-> (summed token NLL, token count) of the targets: the val caption CE health check."""
        nll, keep = self.nll(*self.prefix(*self.visual(b), priors), b["text"])
        return float((nll * keep).sum()), float(keep.sum())


def load_model(ckpt, init=None, is_main=True, **kw):
    """BrainDiff built from a checkpoint's cfg (or the release defaults) with its weights loaded and checked."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    keys = ("lora_r", "lora_alpha", "max_prompt", "max_report_tokens", "train_connector", "status_first")
    model = BrainDiff(init, is_main=is_main, **{k: cfg[k] for k in keys if k in cfg}, **kw)
    model.load_checkpoint(ckpt, init, is_main, ck=ck)
    return model, ck
