"""Stage 1-D model: the vision path and the decoder LoRA learn the change task and single-study tasks through the decoder.

  tokens [V, 2016, 1024] per series and timepoint
    -> NeuroVFM blocks 1-9 (frozen, no_grad) -> blocks 10-12 + final norm (trainable, fp32 masters)
    -> per-scan connector (trainable) + a learned modality-type embedding added before it   [V, 64, 5120]
    -> splice at <|image_pad|>: "Prior study: T1w<..> ..." / "Current study: ..." + the status instruction
    -> Qwen3-14B + LoRA (attention q/k/v/o, r 16, alpha 32; trainable) -> caption CE on the target text

Prior vs current is told apart by the prompt's block labels alone (no slot embeddings). Single-study tasks fill the
current slot only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from braindiff.models.connector import VisionConnector, load_pretrained_connector
from braindiff.models.decoder import Qwen3Decoder
from braindiff.models.encoder import NeuroVFMEncoder, TopBlocks
from braindiff.models.prompts import (INSTRUCTION_STATUS, MODALITIES, PromptTable, block_present_matrix, build_chatml,
                                      build_user_turn, left_pad)

Q = 64
GROUPS = ("blocks", "connector", "modality", "lora")


class SpliceLM:
    """Mixin (needs self.decoder, self.image_pad_id, self.eot): visual chunks into the prompt, caption NLL."""

    def splice(self, ids, attn, chunks, bp):
        """<|image_pad|> -> its 64-token chunk, left-padded."""
        te = self.decoder.embed(ids)
        pe, pa = [], []
        for i in range(ids.shape[0]):
            pos = (ids[i] == self.image_pad_id).nonzero(as_tuple=True)[0]
            groups = bp[i].nonzero(as_tuple=True)[0]
            e, a, last = [], [], 0
            for j, p in enumerate(pos):
                e += [te[i, last:p], chunks[i, groups[j]].to(te.dtype)]
                a += [attn[i, last:p], torch.ones(Q, dtype=attn.dtype, device=attn.device)]
                last = p + 1
            e.append(te[i, last:]); a.append(attn[i, last:])
            pe.append(torch.cat(e)); pa.append(torch.cat(a))
        T = max(x.shape[0] for x in pe)
        prefix = te.new_zeros(len(pe), T, te.shape[-1]); pattn = attn.new_zeros(len(pe), T)
        for i, (e, a) in enumerate(zip(pe, pa)):
            prefix[i, -len(e):] = e; pattn[i, -len(a):] = a
        return prefix, pattn

    def captions(self, texts, device):
        ids = [self.decoder.tokenizer(t, add_special_tokens=False).input_ids + [self.eot] for t in texts]
        L = max(len(x) for x in ids)
        lab = torch.full((len(ids), L), self.decoder.tokenizer.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            lab[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
        return lab.to(device), att.to(device)

    def logits(self, prefix, pattn, lab, latt):
        """-> logits [P, C, V] predicting each target token `lab` after the prefix."""
        emb = torch.cat([prefix, self.decoder.embed(lab).to(prefix.dtype)], 1)
        out = self.decoder(inputs_embeds=emb.to(self.decoder.dtype_), attention_mask=torch.cat([pattn, latt], 1),
                           logits_to_keep=lab.shape[1] + 1, return_dict=True)
        return out.logits[:, :-1]

    def nll(self, prefix, pattn, texts):
        """-> per-token NLL [P, C] and mask [P, C] of `texts` after the prefix."""
        lab, latt = self.captions(texts, prefix.device)
        lg = self.logits(prefix, pattn, lab, latt)
        nll = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), lab.reshape(-1), reduction="none").view_as(lab)
        return nll, latt.float()


class Stage1Text(TopBlocks, SpliceLM, nn.Module):
    def __init__(self, trainable=GROUPS, is_main=True):
        super().__init__()
        assert set(trainable) <= set(GROUPS), trainable
        self.encoder = NeuroVFMEncoder()
        self.init_top_blocks()
        vit = self.encoder.model
        self.modality_type_embedding = nn.Embedding(len(MODALITIES), vit.norm.weight.shape[0])
        nn.init.normal_(self.modality_type_embedding.weight, mean=0.0, std=0.02)
        self.connector = VisionConnector()
        load_pretrained_connector(self.connector, is_main=is_main)
        self.decoder = Qwen3Decoder(use_lora="lora" in trainable)
        self.image_pad_id = self.decoder.image_pad_id
        self.prompts = PromptTable(self.decoder.tokenizer, instruction=INSTRUCTION_STATUS)
        self.eot = self.decoder.tokenizer.convert_tokens_to_ids("<|im_end|>")
        for p in self.parameters():
            p.requires_grad = False
        groups = {"blocks": [p for m in list(vit.blocks[9:]) + [vit.norm] for p in m.parameters()],
                  "connector": list(self.connector.parameters()),
                  "modality": list(self.modality_type_embedding.parameters()),
                  "lora": [p for n, p in self.decoder.named_parameters() if "lora_" in n]}
        self.groups = {k: groups[k] for k in trainable}
        for ps in self.groups.values():
            for p in ps:
                p.requires_grad = True
        if is_main:
            for k, ps in self.groups.items():
                print(f"[freeze]   {k:<10} {len(ps):>4} tensors  {sum(p.numel() for p in ps):>12,} params", flush=True)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.model.eval()            # no dropout / drop-path, no upstream per-block checkpointing
        return self

    def visual(self, b):
        """batch -> per-pair chunks [P, 8, 64, D] (prior T1w..FLAIR, current T1w..FLAIR) + presence [P, 8]."""
        ia, ma = b.get("item_a", b["item"]), b.get("mods_a", b["mods"])
        S = b["ta"].shape[0]
        f = self.encode(torch.cat([b["ta"], b["tb"]]))
        if "modality" in self.groups:
            f = f + self.modality_type_embedding(torch.cat([ma, b["mods"]]))[:, None, :].to(f.dtype)
        z = self.connector(f)                                                # [Sa + Sb, 64, D]
        P, M = b["n_items"], len(MODALITIES)
        out = z.new_zeros(P, 2 * M, Q, z.shape[-1])
        bp = torch.zeros(P, 2 * M, dtype=torch.bool, device=z.device)
        for (it, md, lo, hi, t) in ((ia, ma, 0, S, 0), (b["item"], b["mods"], S, z.shape[0], 1)):
            out[it, t * M + md] = z[lo:hi]
            bp[it, t * M + md] = True
        return out, bp

    def prefix(self, b):
        ch, bp = self.visual(b)
        M = len(MODALITIES)
        if "prompt" in b:
            ids, attn, tbp = self.task_prompts(bp[:, :M].tolist(), bp[:, M:].tolist(), b["prompt"], ch.device)
        else:
            pres = bp[:, :M].tolist()
            ids, attn, tbp = self.prompts.batch(pres, pres, device=ch.device)
        assert torch.equal(tbp, bp), "prompt placeholders disagree with the visual blocks"
        return self.splice(ids, attn, ch, bp)

    def task_prompts(self, pr, pm, prompts, device):
        """Single-study tasks: 'Study: ...' + the row's instruction (no prior present); a pair item gets the pair prompt."""
        rows = []
        for r, m, ins in zip(pr, pm, prompts):
            single = not any(r)
            assert single == (ins is not None), "a task item needs its prompt and no prior; a pair neither"
            rows.append(self.decoder.tokenizer(build_chatml(build_user_turn(
                m, None if single else r, ins if single else self.prompts.instruction)), add_special_tokens=False).input_ids)
        return left_pad(rows, self.decoder.tokenizer.pad_token_id,
                        bps=[block_present_matrix(m, r) for r, m in zip(pr, pm)], device=device)

    def forward(self, b):
        prefix, pattn = self.prefix(b)
        nll, keep = self.nll(prefix, pattn, b["text"])
        return (nll * keep).sum() / keep.sum().clamp(min=1)

    @torch.no_grad()
    def generate(self, b, max_new_tokens=96):
        """-> greedy generations after the batch's prompt."""
        prefix, pattn = self.prefix(b)
        out = self.decoder.generate(prefix, pattn, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=[self.eot, self.decoder.tokenizer.eos_token_id])
        return [t.strip() for t in self.decoder.tokenizer.batch_decode(out, skip_special_tokens=True)]

    @torch.no_grad()
    def score(self, b, none_text, max_new_tokens=96):
        """-> (sum NLL of `none_text` per pair [P] (higher = more likely changed), greedy generations)."""
        prefix, pattn = self.prefix(b)
        nll, keep = self.nll(prefix, pattn, [none_text] * b["n_items"])
        out = self.decoder.generate(prefix, pattn, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=[self.eot, self.decoder.tokenizer.eos_token_id])
        txt = self.decoder.tokenizer.batch_decode(out, skip_special_tokens=True)
        return (nll * keep).sum(1).float().cpu(), [t.strip() for t in txt]
