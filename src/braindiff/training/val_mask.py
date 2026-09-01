"""Content-token masks for the validation metric.

Val cross-entropy is dominated by tokens the language prior predicts on its own, which is
why stage 3's val loss bottoms at epoch 5 while the best RadGraph checkpoint is epoch 10.
Restricting the metric to tokens that require the image is the same forward pass, reduced
differently.

Two definitions of "requires the image":

1. `build_content_masks` (no model needed): keep tokens belonging to a matched
   anatomy/pathology entity or change cue.
2. `load_surprisal_mask`: keep the top-k most surprising tokens under the blind no-image
   model. Measured rather than assumed, but needs that control to exist first.

Either way the mask is fixed for the run: validation captions are not augmented, so token
positions are stable and masks are built once at startup and indexed by dataset position.
"""
import re

import torch

from braindiff.eval.temporal_score import _change_pattern, _entity_pattern
from braindiff.data.report_text import split_report_sentences

# A sentence asserting normality or absence. These are the tokens the language
# prior predicts for free -- 89.3% of generated reports contain "are normal",
# almost exactly the reference rate (88.1%), while the findings that make a
# report that patient's report go unwritten.
#
# Defined by inversion (boilerplate, not content) on purpose: enumerating every
# way to describe a finding is open-ended, whereas the ways to say "nothing here"
# are a short closed list.
_BOILERPLATE_RE = re.compile(
    # "... are normal", "... is of normal signal intensity/form/width"
    r"\b(?:are|is|were|was)\s+(?:of\s+)?normal\b"
    r"|\bof\s+normal\b"
    r"|\bwithin\s+normal\s+limits\b"
    r"|\bnormal\s+(?:in\s+)?(?:size|width|appearance|limits|calibre|caliber|form)\b"
    r"|\bunremarkable\b"
    # absence, with room for a multi-word subject: "No acute infarct was
    # detected", "No findings suggestive of thrombus were detected".
    r"|\b(?:no|not)\b[^.]{0,45}?\b(?:detected|observed|identified|seen|noted)\b"
    r"|\bno\s+(?:significant\s+)?(?:patholog\w+|abnormal\w*|evidence|deviation)\b"
    r"|\b(?:is|are)\s+(?:open|patent|intact|symmetric)\b",
    re.IGNORECASE,
)

# A sentence carrying a measurement is describing something real, whatever else
# it says. Without this, "A 12 mm cyst is present, with no enhancement observed"
# is demoted by the absence clause -- the exact failure mode worth guarding.
_MEASUREMENT_RE = re.compile(r"\d+\s*(?:mm|cm)\b", re.IGNORECASE)


def _boilerplate_spans(text):
    """Character spans of sentences that only assert normality or absence.

Scoped to whole sentences: phrase-level scoping would demote the "no enhancement" clause of
a sentence otherwise describing a real lesion. Segmentation reuses `split_report_sentences`
(syntok), which the S2 loader already runs per item; it returns strings rather than offsets,
so each is located with a moving cursor.
    """
    spans, cursor = [], 0
    for sent in split_report_sentences(text):
        i = text.find(sent, cursor)
        if i < 0:                      # re-segmentation drift; skip rather than guess
            continue
        cursor = i + len(sent)
        if _BOILERPLATE_RE.search(sent) and not _MEASUREMENT_RE.search(sent):
            spans.append((i, cursor))
    return spans


def build_token_weights(caption, tokenizer, max_length, content_weight):
    """[max_length] float CE weights for one caption.

Boilerplate tokens weigh 1.0, everything else `content_weight`, then the vector is rescaled
so the mean over real tokens is 1.0. That keeps the loss magnitude -- and therefore the
effective learning rate and the cosine schedule -- identical to the unweighted run, and
makes content_weight=1.0 an exact no-op.

Mirrors TokenizeCaption: ONE slot reserved for Qwen3's <|im_end|>, so weight position t
lines up with input_ids[t] and with the per-token nll from _caption_nll. Padding stays 0.0.
    """
    weights = torch.zeros(max_length, dtype=torch.float)
    encoded = tokenizer(
        caption, add_special_tokens=False, truncation=True,
        max_length=max_length - 1, return_offsets_mapping=True,
    )
    spans = _boilerplate_spans(caption)
    n_core = len(encoded["input_ids"])

    for t, (start, end) in enumerate(encoded["offset_mapping"]):
        if start == end:                       # whitespace-only piece, no span to judge
            weights[t] = 1.0
            continue
        boiler = any(start < b_end and end > b_start for b_start, b_end in spans)
        weights[t] = 1.0 if boiler else content_weight

    # The turn terminator is fully predictable -- never content.
    n_real = min(n_core + 1, max_length)
    weights[n_core:n_real] = 1.0

    total = weights[:n_real].sum()
    if total > 0:
        weights[:n_real] *= n_real / total
    return weights


def _content_spans(text):
    """Character spans of every matched entity and change cue."""
    spans = [m.span() for m in _entity_pattern().finditer(text.lower())]
    spans += [m.span() for m in _change_pattern().finditer(text.lower())]
    return spans


def build_content_masks(captions, tokenizer, max_length):
    """[N, max_length] bool -- True on tokens overlapping a clinical term.

Uses the tokenizer's offset mapping rather than decoding token-by-token, so a term that
splits into several word pieces marks all of them.
    """
    masks = torch.zeros(len(captions), max_length, dtype=torch.bool)
    for i, caption in enumerate(captions):
        encoded = tokenizer(
            caption, add_special_tokens=False, truncation=True,
            max_length=max_length - 2, return_offsets_mapping=True,
        )
        spans = _content_spans(caption)
        if not spans:
            continue
        for t, (start, end) in enumerate(encoded["offset_mapping"]):
            if start == end:
                continue
            if any(start < s_end and end > s_start for s_start, s_end in spans):
                masks[i, t] = True
    return masks


def load_surprisal_mask(path, keep_fraction=0.4):
    """Top-`keep_fraction` most surprising tokens from a saved [N, C] surprisal tensor produced by
running the blind no-image model over the val set.

Thresholded per sample rather than globally, so a long report cannot crowd out a short one.
    """
    surprisal = torch.load(path)
    k = max(1, int(keep_fraction * surprisal.shape[1]))
    cutoff = surprisal.topk(k, dim=1).values[:, -1:]
    return surprisal >= cutoff
