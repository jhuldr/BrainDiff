"""ChatML prompts for the Qwen3 decoder: system prompt, labelled image placeholders, the Stage 1-D status task, the Stage 2
prior-report prompt and the status line.

Placeholders carry per-block, per-modality labels because absent series are dropped. Prompts are left-padded; the
prefix/caption split depends on the supervised span being the contiguous tail.
"""
import torch

MODALITIES = ("T1w", "T1ce", "T2w", "FLAIR")
STATUSES = ("NEW", "WORSENED", "IMPROVED", "RESOLVED")

# Verbatim from mlinslab/neurovfm-llm/config.json (the released connector's distribution).
SYSTEM_PROMPT = ("You are an expert neuro-radiologist AI assistant. Analyze the provided "
                 "neuroimaging study and answer the user's request. /no_think")
BLOCK_LABELS = {"ref": "Prior study", "main": "Current study"}

# Stage 1-D change task (fmt "status"): statuses only, closed by CLOSE; NONE_TEXT for a pair without change.
CLOSE, NONE_TEXT = "No other interval change.", "No interval change."
INSTRUCTION_STATUS = ("Which interval changes are present between the prior and current studies? Answer with the "
                      "statuses present (NEW, WORSENED, IMPROVED, RESOLVED), then end with 'No other interval change.', "
                      "e.g. 'NEW. IMPROVED. No other interval change.' If nothing changed, answer 'No interval change.'")

# Stage 2: the prior report goes after the image placeholders and before the instruction.
PRIOR_HEADER = "Prior report:"
NO_PRIOR = "Prior report: not available."
CURRENT_UNKNOWN_FOLLOWUP = ("The report for the current (follow-up) study is NOT provided. Read the current "
                            "findings off the images above, and use the prior report only as the baseline to "
                            "compare against.")
INSTRUCTION_FOLLOWUP = ("Write the report for the current study as a follow-up to the prior study, using the "
                        "following structure: Findings: [current findings, each stated as new, worsened, improved, "
                        "resolved or unchanged compared with the prior] Impression: [clinical interpretation].")


def status_target(statuses):
    """Stage 1-D target: 'NEW. IMPROVED. No other interval change.' / 'No interval change.'"""
    return " ".join([f"{s}." for s in STATUSES if s in statuses] + [CLOSE]) if statuses else NONE_TEXT


def status_line(r):
    """Status-first line in Stage 1-D's format from a Stage-2 pair's st_* labels (the caption's first line)."""
    on = [s for s in STATUSES if r[f"st_{s}"]]
    return " ".join(f"{s}." for s in on) + f" {CLOSE}" if on else NONE_TEXT


def _series_span(label, present):
    """'Prior study: T1w<|vision_start|><|image_pad|><|vision_end|> T2w<...>'"""
    return f"{label}: " + " ".join(f"{m}<|vision_start|><|image_pad|><|vision_end|>"
                                   for m, ok in zip(MODALITIES, present) if ok)


def build_user_turn(present_main, present_ref=None, instruction=None):
    """User-turn content: placeholders interleaved with block / modality labels, then the instruction."""
    if present_ref is None:
        lines = [_series_span("Study", present_main)]
    else:
        lines = [_series_span(BLOCK_LABELS["ref"], present_ref), _series_span(BLOCK_LABELS["main"], present_main)]
    return "\n".join(lines + [instruction])


def build_chatml(user_content, system_prompt=SYSTEM_PROMPT):
    """System turn + user turn + the assistant header the caption continues from."""
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def block_present_matrix(present_main, present_ref=None):
    """-> [n_blocks * 4] bool, block-major (prior, current); must agree with the placeholder count."""
    pm = [bool(x) for x in present_main]
    return pm if present_ref is None else [bool(x) for x in present_ref] + pm


def left_pad(rows, pad, n_groups=None, bps=None, device=None):
    """[ids per row] -> left-padded ids [B, P], attention [B, P] (+ block presence [B, G])."""
    P = max(len(r) for r in rows)
    ids = torch.full((len(rows), P), pad, dtype=torch.long)
    attn = torch.zeros((len(rows), P), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, -len(r):] = torch.as_tensor(r, dtype=torch.long)
        attn[i, -len(r):] = 1
    bp = torch.stack([torch.as_tensor(b, dtype=torch.bool) for b in bps]) if bps is not None else None
    if device is not None:
        ids, attn = ids.to(device), attn.to(device)
        bp = bp.to(device) if bp is not None else None
    return ids, attn, bp


class PromptTable:
    """Cached (present_ref, present_main) -> (prompt ids, block presence) for the Stage 1-D pair prompt. Raises rather
    than truncating: a capped prompt silently destroys all conditioning."""

    def __init__(self, tokenizer, instruction=INSTRUCTION_STATUS, system_prompt=SYSTEM_PROMPT, max_prompt_length=384):
        self.tok, self.instruction, self.system_prompt = tokenizer, instruction, system_prompt
        self.max_prompt_length = max_prompt_length
        self.image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.n_groups = 2 * len(MODALITIES)
        self._cache = {}

    def _build(self, key):
        pr, pm = key
        ids = self.tok(build_chatml(build_user_turn(pm, pr, self.instruction), self.system_prompt),
                       add_special_tokens=False).input_ids
        if len(ids) > self.max_prompt_length:
            raise ValueError(f"prompt is {len(ids)} tokens, over max_prompt_length={self.max_prompt_length}")
        bp = block_present_matrix(pm, pr)
        assert sum(t == self.image_pad_id for t in ids) == sum(bp), "placeholders disagree with block presence"
        return torch.tensor(ids, dtype=torch.long), torch.tensor(bp, dtype=torch.bool)

    def get(self, present_main, present_ref):
        key = (tuple(bool(x) for x in present_ref), tuple(bool(x) for x in present_main))
        if key not in self._cache:
            self._cache[key] = self._build(key)
        return self._cache[key]

    def batch(self, present_main, present_ref, device=None):
        """Left-padded [B, P] prompt ids + [B, P] attention + [B, G] block presence."""
        rows = [self.get(present_main[i], present_ref[i]) for i in range(len(present_main))]
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        return left_pad([r for r, _ in rows], pad, bps=[b for _, b in rows], device=device)


def user_turn_with_prior(present_main, present_ref, prior_report, instruction=INSTRUCTION_FOLLOWUP,
                         current_unknown=CURRENT_UNKNOWN_FOLLOWUP):
    """Placeholders, then the prior report, then CURRENT_UNKNOWN, then the instruction."""
    head, _, instruction = build_user_turn(present_main, present_ref, instruction).rpartition("\n")
    text = f"{PRIOR_HEADER} {prior_report.strip()}" if prior_report.strip() else NO_PRIOR
    return f"{head}\n{text}\n{current_unknown}\n{instruction}"


class PriorReportPrompts:
    """Per-sample Stage 2 prompts carrying each pair's (edited) prior report; set_reports() binds a batch's reports.
    _clip keeps the Impression and fills the rest of the budget from the start of Findings."""

    def __init__(self, tokenizer, max_prompt_length=1536, max_report_tokens=384, system_prompt=SYSTEM_PROMPT):
        self.tok, self.system_prompt = tokenizer, system_prompt
        self.max_prompt_length, self.max_report_tokens = max_prompt_length, max_report_tokens
        self.image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.n_groups = 2 * len(MODALITIES)
        self.n_truncated = self.n_seen = 0
        self._reports = None

    def set_reports(self, reports):
        self._reports = list(reports)
        return self

    def _clip(self, report):
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
        if len(imp_ids) >= self.max_report_tokens:      # Impression alone overruns: keep its head (the verdict)
            return self.tok.decode(imp_ids[:self.max_report_tokens])
        room = self.max_report_tokens - len(imp_ids) - 4      # 4 for the elision marker
        head_ids = self.tok(head, add_special_tokens=False).input_ids
        return self.tok.decode(head_ids[:max(room, 0)]) + " ... " + imp_text

    def text(self, present_main, present_ref, report):
        """The full ChatML prompt string of one pair."""
        return build_chatml(user_turn_with_prior(present_main, present_ref, self._clip(report)), self.system_prompt)

    def _build_row(self, present_main, present_ref, report):
        ids = self.tok(self.text(present_main, present_ref, report), add_special_tokens=False).input_ids
        if len(ids) > self.max_prompt_length:
            raise ValueError(f"prompt is {len(ids)} tokens, over max_prompt_length={self.max_prompt_length}")
        bp = block_present_matrix(present_main, present_ref)
        assert sum(t == self.image_pad_id for t in ids) == sum(bp), "placeholders disagree with block presence"
        return torch.tensor(ids, dtype=torch.long), torch.tensor(bp, dtype=torch.bool)

    def batch(self, present_main, present_ref, device=None):
        reports = self._reports if self._reports is not None else [""] * len(present_main)
        assert len(reports) == len(present_main), "set_reports() was called for a different batch"
        rows = [self._build_row(present_main[i], present_ref[i], reports[i]) for i in range(len(present_main))]
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        return left_pad([r for r, _ in rows], pad, bps=[b for _, b in rows], device=device)
