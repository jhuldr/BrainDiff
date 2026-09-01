"""Report segmentation shared by the single and dual braindiff.data.

Both stages need the same sentence splitter and the same section-aware reordering
augmentation, so they live in one place.
"""
import random
import re

import syntok.segmenter as segmenter

# Section headers used by the S4 targets ("Findings: Lesions: ... Structural
# Effects: ... Background Findings: ... Impression: ...") and the S2 reports,
# which only carry Findings/Impression (96.7% / 87.1% of rows).
SECTION_HEADERS = (
    "Findings", "Lesions", "Structural Effects", "Background Findings", "Impression",
)
_HEADER_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:",
    re.IGNORECASE,
)

# Sentence openers that only make sense relative to whatever came before --
# either true anaphora ("This lesion...") or additive/contrastive connectives
# ("Additionally...", "Otherwise..."). Measured on 400 stage-2 reports: 1.1% of
# non-initial sentences, in 12% of reports. Small, but stage 2 is raw
# radiologist prose rather than the normalized stage-4 targets, and it is the
# split where every report gets reordered.
_REFERRING_RE = re.compile(
    r"^\W*("
    r"this|these|those|it|its|they|their|such|said|same"
    r"|the (?:lesion|mass|latter|former|above|aforementioned|same)"
    r"|additionally|furthermore|moreover|also|in addition|besides"
    r"|however|otherwise|nevertheless|conversely|by contrast|in contrast"
    r"|therefore|thus|hence|consequently|as a result"
    r")(?!\w)",
    re.IGNORECASE,
)


def group_dependent_sentences(sentences):
    """Chunk sentences so a back-referring one travels with what it refers to.

Pinning such a sentence to its index would not help: the sentence in front of it would still
change. Gluing it to its predecessor keeps the pair intact wherever it lands, and
consecutive referring sentences accumulate into one block. A referring sentence in first
position becomes its own block, and the caller keeps it anchored.
    """
    blocks = []
    for i, sentence in enumerate(sentences):
        if i > 0 and _REFERRING_RE.match(sentence):
            blocks[-1].append(sentence)
        else:
            blocks.append([sentence])
    return blocks


def split_report_sentences(text):
    """Segment a report into sentences with syntok — handles decimals ('1.5 cm')
    and abbreviations ('e.g.', 'vs.') that a naive '.' split breaks."""
    out = []
    for paragraph in segmenter.process(text):
        for sentence in paragraph:
            s = "".join(tok.spacing + tok.value for tok in sentence).strip()
            if s:
                out.append(s)
    return out


def split_sections(text):
    """Report -> [(header_or_None, body)] in document order.

syntok glues a header to the first sentence of its section, so shuffling its output directly
would drag headers around. Splitting on headers first is what makes the reordering safe.
    """
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [(None, text)]

    sections = []
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(0), text[m.end():end].strip()))
    return sections


def reorder_report_sections(text, rng=random):
    """Shuffle sentences within each section, never across them.

The targets are near-templates -- 63% of S4 reports contain the same "Structural Effects: No
meaningful interval structural effect" verbatim -- so reordering is one of the few
augmentations that adds real surface variety. But section order carries meaning: Findings
must precede Impression, and a lesion sentence must not migrate into Background Findings.

Reports with no recognizable header are returned unchanged rather than globally shuffled.
Sentences referring back to their predecessor are not shuffled independently; see
group_dependent_sentences.
    """
    sections = split_sections(text)
    if len(sections) == 1 and sections[0][0] is None:
        return text

    parts = []
    for header, body in sections:
        if body:
            body = _reorder_body(body, rng)
        parts.append(f"{header} {body}".strip() if header else body)
    return " ".join(p for p in parts if p)


def _reorder_body(body, rng):
    """Shuffle one section's blocks, or return it untouched if that would destabilize the split.

The guard compares the shuffled join against the unshuffled join of the same blocks -- not
against raw text, and per section rather than per report. Comparing to raw text would reject
almost everything, since joining bullet-style items with a space already re-segments them;
comparing whole reports would too, because reordering changes which sentence carries the
header.

The real risk it catches: some stage-2 items have no terminal punctuation, so moving one
mid-body makes syntok merge it with its neighbour and can strand a referring sentence from
its referent. Those sections are skipped; the rest of the report is still augmented.
    """
    sentences = split_report_sentences(body)
    blocks = group_dependent_sentences(sentences)
    # A leading referring sentence points back at the previous section, so its
    # block stays first and only the remainder moves.
    anchor = 1 if (sentences and _REFERRING_RE.match(sentences[0])) else 0
    movable = blocks[anchor:]
    if len(movable) < 2:
        return body

    rng.shuffle(movable)
    ordered = blocks[:anchor] + movable
    out = " ".join(s for block in ordered for s in block)
    baseline = " ".join(s for block in blocks for s in block)
    if sorted(split_report_sentences(out)) != sorted(split_report_sentences(baseline)):
        return body
    return out


# Stage 2's target is "Pathologies: <labels>\n<report>" (mutli_single_dataloader.py).
# Selection and the offline RadGraph pass score the report only, so the numbers stay
# comparable with the S2 checkpoints trained before the pcls prefix existed.
_PATHOLOGY_PREFIX_RE = re.compile(r"^\s*Pathologies\s*:[^\n]*\n?", re.IGNORECASE)


def strip_pathology_prefix(text: str) -> str:
    """Drop a leading 'Pathologies: ...' line. No-op when absent, so S1/S4 text and
    any checkpoint that has not learned to emit the line pass through unchanged."""
    if not text:
        return text
    return _PATHOLOGY_PREFIX_RE.sub("", text, count=1).lstrip()
