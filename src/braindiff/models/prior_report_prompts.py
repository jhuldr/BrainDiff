"""Prompts that carry the prior report text, built per sample.

PromptTable caches on (present_ref, present_main) because the prompt is a pure function of
those. A prior report is not -- it varies per row -- so each row is tokenised individually.
Tokenising ~250 tokens per sample is negligible against a 14B decoder forward.

Only `batch`, the one method the captioner calls, is required, so this duck-types
PromptTable.

Budget: S4's prompt is max 205 tokens against a 384 cap and the report adds a median ~250,
so `max_prompt_length` has to rise. Qwen3 has sliding_window null and 40960 positions, so
the prefix (P + 63k, k<=12) is nowhere near a limit. The report is truncated to
`max_report_tokens` and the truncation rate is counted and printed rather than quietly
applied.
"""
import torch

from braindiff.models.prompts import (MODALITIES, SYSTEM_PROMPT, build_chatml, build_user_turn,
                            block_present_matrix)

PRIOR_HEADER = "Prior report:"
NO_PRIOR = "Prior report: not available."
# Stated explicitly because the prior report is the easiest thing in the context to
# copy, and on a Stable pair copying is also the correct answer -- so BLEU/METEOR
# cannot distinguish paraphrase from understanding. This line names the actual task:
# the current study's report does not exist and has to be read off the images.
CURRENT_UNKNOWN = ("The report for the current (follow-up) study is NOT provided. "
                   "Infer the current findings from the images above and the interval "
                   "change, and use the prior report only as the baseline to compare "
                   "against.")


def user_turn_with_prior(present_main, present_ref, prior_report, include_delta):
    """Prior report goes BEFORE the instruction, after the image placeholders.

Placing it before the images would separate the placeholders from the instruction that
refers to them; placing it after the instruction buries the question.
    """
    base = build_user_turn(present_main, present_ref, include_delta=include_delta)
    head, _, instruction = base.rpartition("\n")
    text = f"{PRIOR_HEADER} {prior_report.strip()}" if prior_report.strip() else NO_PRIOR
    return f"{head}\n{text}\n{CURRENT_UNKNOWN}\n{instruction}"


class PriorReportPrompts:
    """Build per-sample prompts carrying each row's prior report."""

    def __init__(self, tokenizer, include_delta=True, max_prompt_length=1024,
                 max_report_tokens=384, system_prompt=SYSTEM_PROMPT):
        self.tok = tokenizer
        self.include_delta = include_delta
        self.max_prompt_length = max_prompt_length
        self.max_report_tokens = max_report_tokens
        self.system_prompt = system_prompt
        self.image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.n_blocks = 3 if include_delta else 2
        self.n_truncated = 0
        self.n_seen = 0
        self._reports = None

    @property
    def n_groups(self):
        return self.n_blocks * len(MODALITIES)

    def set_reports(self, reports):
        """Bind this batch's prior reports; call before the model forward."""
        self._reports = list(reports)
        return self

    def _clip(self, report):
        """Fit the report to budget while keeping the Impression.

15.5% of prior reports exceed 384 tokens (p50 258, p90 431, max 1128). Head-truncation
drops the tail, which in a radiology report is the Impression -- the most informative
section. So the Impression is reserved and the remaining budget is filled from the start
of Findings.
        """
        self.n_seen += 1
        ids = self.tok(report, add_special_tokens=False).input_ids
        if len(ids) <= self.max_report_tokens:
            return report
        self.n_truncated += 1

        head, sep, impression = report.rpartition("Impression:")
        if not sep:
            return self.tok.decode(ids[:self.max_report_tokens])
        imp_text = (sep + impression).strip()
        imp_ids = self.tok(imp_text, add_special_tokens=False).input_ids
        if len(imp_ids) >= self.max_report_tokens:
            # Impression alone overruns: keep its head, since it opens with the verdict.
            return self.tok.decode(imp_ids[:self.max_report_tokens])
        room = self.max_report_tokens - len(imp_ids) - 4      # 4 for the elision marker
        head_ids = self.tok(head, add_special_tokens=False).input_ids
        return self.tok.decode(head_ids[:max(room, 0)]) + " ... " + imp_text

    def _build_row(self, present_main, present_ref, report):
        text = build_chatml(
            user_turn_with_prior(present_main, present_ref, self._clip(report),
                                 self.include_delta),
            self.system_prompt)
        ids = self.tok(text, add_special_tokens=False).input_ids
        if len(ids) > self.max_prompt_length:
            raise ValueError(
                f"prompt is {len(ids)} tokens, over max_prompt_length="
                f"{self.max_prompt_length}; raise the cap rather than truncating")
        bp = block_present_matrix(present_main, present_ref, self.include_delta)
        n_pad = sum(1 for t in ids if t == self.image_pad_id)
        assert n_pad == sum(bp), f"{n_pad} <|image_pad|> vs {sum(bp)} present blocks"
        return torch.tensor(ids, dtype=torch.long), torch.tensor(bp, dtype=torch.bool)

    def batch(self, present_main, present_ref=None, device=None):
        assert present_ref is not None, "prior-report prompts are dual-timepoint only"
        reports = self._reports if self._reports is not None else [""] * len(present_main)
        assert len(reports) == len(present_main), (
            f"{len(present_main)} rows but {len(reports)} prior reports -- "
            f"set_reports() was called for a different batch")
        rows = [self._build_row(present_main[i], present_ref[i], reports[i])
                for i in range(len(present_main))]

        # LEFT pad, exactly as PromptTable does: the prefix/caption split depends on
        # the supervised span being the contiguous tail.
        P = max(len(i) for i, _ in rows)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        ids = torch.full((len(rows), P), pad, dtype=torch.long)
        attn = torch.zeros((len(rows), P), dtype=torch.long)
        bp = torch.zeros((len(rows), self.n_groups), dtype=torch.bool)
        for i, (row_ids, row_bp) in enumerate(rows):
            n = len(row_ids)
            ids[i, -n:] = row_ids
            attn[i, -n:] = 1
            bp[i] = row_bp
        if device is not None:
            ids, attn, bp = ids.to(device), attn.to(device), bp.to(device)
        return ids, attn, bp
