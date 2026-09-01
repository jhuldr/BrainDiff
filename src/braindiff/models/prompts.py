"""ChatML prompt construction for the Qwen3 decoder.

Format matches neurovfm/data/text.py::process_text byte-for-byte, asserted at token level by
braindiff/eval/verify_prompt_format.py for the one case both can express (all placeholders
contiguous at the head of the first user turn). process_text itself is not called: it
hardcodes that contiguous layout, and placeholders are interleaved here with per-block,
per-modality labels.

Interleaving is needed because absent series are dropped (see models/connector.py), so
position alone no longer identifies a group's modality -- a study missing T1ce has the same
placeholder count as one missing FLAIR. The learned temporal embeddings stay as well; they
are 15k params and the delta block has no natural text description.

PromptTable is what makes the S4 counterfactual work: the prompt is a pure function of
(present_ref, present_main), so all <=256 variants are cached. Swapping the prior changes
the placeholder count and the table returns a correct prompt for the swapped pair, so
partner selection keeps only the different-classification constraint.
"""
from functools import lru_cache
from typing import Sequence, Tuple

import torch

MODALITIES = ("T1w", "T1ce", "T2w", "FLAIR")

# Verbatim from mlinslab/neurovfm-llm/config.json -> language_model_cf.system_prompt.
# Kept exactly so we stay in the distribution the released connector was trained in.
SYSTEM_PROMPT = (
    "You are an expert neuro-radiologist AI assistant. Analyze the provided "
    "neuroimaging study and answer the user's request. /no_think"
)

# Instructions carried over verbatim from BrainDiff models/MultiModal/model.py:237-289,
# minus the "you are provided with N image blocks" framing -- the labelled placeholders
# now say that structurally, so repeating it in prose is redundant.
_STRUCTURE = (
    "Findings: Lesions: [primary lesions/acute infarct/hemorrhage] "
    "Structural Effects: [secondary mass effect/edema/hydrocephalus/etc.] "
    "Background Findings: [chronic contextual findings] "
    "Impression: [clinical interpretation]. "
    "What are the key findings and your impression?"
)
INSTRUCTION_SINGLE = (
    "What are the key findings and your impression? Structure your response as - "
    "Findings: [observations] Impression: [clinical interpretation]. "
    "Ensure impressions follow directly from findings."
)
INSTRUCTION_DUAL = (
    "Describe the interval changes observed between the two timepoints, "
    "using the following structure: " + _STRUCTURE
)

BLOCK_LABELS = {
    "ref": "Prior study",
    "main": "Current study",
    "delta": "Interval change (current minus prior)",
}
# Block order is load-bearing: it fixes the order chunks are spliced in, and
# models/captioner.py indexes its [B, 12, ...] tensor as block-major in this order.
DUAL_BLOCKS = ("ref", "main", "delta")
SINGLE_BLOCKS = ("main",)


def _series_span(label: str, present: Sequence[bool]) -> str:
    """'Prior study: T1w<|vision_start|><|image_pad|><|vision_end|> T2w<...>'"""
    parts = [f"{m}<|vision_start|><|image_pad|><|vision_end|>"
             for m, ok in zip(MODALITIES, present) if ok]
    return f"{label}: " + " ".join(parts)


def build_user_turn(present_main: Sequence[bool],
                    present_ref: Sequence[bool] = None,
                    include_delta: bool = False,
                    instruction: str = None) -> str:
    """The user-turn content, placeholders interleaved with block/modality labels."""
    single = present_ref is None
    lines = []
    if single:
        lines.append(_series_span("Study", present_main))
    else:
        joint = [a and b for a, b in zip(present_ref, present_main)]
        lines.append(_series_span(BLOCK_LABELS["ref"], present_ref))
        lines.append(_series_span(BLOCK_LABELS["main"], present_main))
        if include_delta:
            lines.append(_series_span(BLOCK_LABELS["delta"], joint))
    if instruction is None:
        instruction = INSTRUCTION_SINGLE if single else INSTRUCTION_DUAL
    lines.append(instruction)
    return "\n".join(lines)


def build_chatml(user_content: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """System turn + user turn + the assistant header the caption continues from.

    Byte-identical to process_text's prompt-side output; the caption is appended
    separately by the model (see the prefix/caption split in models/captioner.py).
    """
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def block_present_matrix(present_main, present_ref=None, include_delta=False):
    """-> [n_blocks * 4] bool, block-major in DUAL_BLOCKS / SINGLE_BLOCKS order.

The delta block for modality m exists iff m is present at BOTH timepoints. This must agree
with the placeholder count build_user_turn emits or the splice raises;
verify_prompt_format.py asserts that agreement over real CSV rows.
    """
    pm = [bool(x) for x in present_main]
    if present_ref is None:
        return pm
    pr = [bool(x) for x in present_ref]
    out = pr + pm
    if include_delta:
        out += [a and b for a, b in zip(pr, pm)]
    return out


class PromptTable:
    """Cached (present_ref, present_main) -> (prompt_ids [P], block_present [G]).

At most 16x16 = 256 variants per stage, tokenized on first use. This is what lets the
counterfactual swap the prior freely: look up the swapped pair, no re-tokenization.
    """

    def __init__(self, tokenizer, single_timepoint: bool = False,
                 include_delta: bool = True, instruction: str = None,
                 system_prompt: str = SYSTEM_PROMPT, max_prompt_length: int = 384):
        self.tok = tokenizer
        self.single = single_timepoint
        self.include_delta = include_delta and not single_timepoint
        self.instruction = instruction
        self.system_prompt = system_prompt
        self.max_prompt_length = max_prompt_length
        self.image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        assert self.image_pad_id is not None and self.image_pad_id >= 0
        self.n_blocks = 1 if single_timepoint else (3 if self.include_delta else 2)
        self._cache = {}

    @property
    def n_groups(self) -> int:
        return self.n_blocks * len(MODALITIES)

    def _build(self, key):
        pr, pm = key
        text = build_chatml(
            build_user_turn(pm, None if self.single else pr,
                            include_delta=self.include_delta,
                            instruction=self.instruction),
            self.system_prompt,
        )
        ids = self.tok(text, add_special_tokens=False).input_ids
        # Raise, never truncate. BrainDiff's 200->320 prompt cap once silently
        # destroyed all `caref` conditioning by truncating mid-prompt.
        if len(ids) > self.max_prompt_length:
            raise ValueError(
                f"prompt is {len(ids)} tokens, over max_prompt_length="
                f"{self.max_prompt_length}; raise the cap rather than truncating")
        bp = block_present_matrix(pm, None if self.single else pr, self.include_delta)
        n_pad = sum(1 for t in ids if t == self.image_pad_id)
        assert n_pad == sum(bp), (
            f"prompt has {n_pad} <|image_pad|> but block_present has {sum(bp)} True")
        return torch.tensor(ids, dtype=torch.long), torch.tensor(bp, dtype=torch.bool)

    def get(self, present_main, present_ref=None):
        pm = tuple(bool(x) for x in present_main)
        pr = tuple(bool(x) for x in present_ref) if present_ref is not None else None
        key = (pr, pm)
        if key not in self._cache:
            self._cache[key] = self._build(key)
        return self._cache[key]

    def batch(self, present_main, present_ref=None, device=None):
        """Left-padded [B, P] prompt ids + [B, P] attention + [B, G] block presence."""
        rows = [self.get(present_main[i], None if present_ref is None else present_ref[i])
                for i in range(len(present_main))]
        P = max(len(ids) for ids, _ in rows)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.full((len(rows), P), pad, dtype=torch.long)
        attn = torch.zeros((len(rows), P), dtype=torch.long)
        bp = torch.zeros((len(rows), self.n_groups), dtype=torch.bool)
        for i, (row_ids, row_bp) in enumerate(rows):
            n = len(row_ids)
            ids[i, -n:] = row_ids          # LEFT pad -- the whole prefix/caption split
            attn[i, -n:] = 1               # depends on the supervised span being the tail
            bp[i] = row_bp
        if device is not None:
            ids, attn, bp = ids.to(device), attn.to(device), bp.to(device)
        return ids, attn, bp
