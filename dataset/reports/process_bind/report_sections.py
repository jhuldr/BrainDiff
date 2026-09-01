#!/usr/bin/env python3
"""
Section handling for raw BIND radiology reports.

`dataloaders/MultiModal/report_text.py::split_sections` cannot be reused here: its
SECTION_HEADERS are the normalized S2/S4 set (Findings, Lesions, Structural
Effects, Background Findings, Impression) and BIND's raw prose does not use them.
Header frequencies measured over 4,000 sampled reports per site, 2026-08-04:

    I0001   TECHNIQUE 57.6%  COMPARISON 56.2%  FINDINGS 55.5%  ATTESTATION 15.0%
            HISTORY 9.2%  INDICATION 9.1%  IMPRESSION 4.8%
    I0004   NARRATIVE 100%  FINDINGS 98.2%  COMPARISON 94.6%  IMPRESSION 94.0%
            ACCESSION NUMBER 92.5%  PROCEDURE COMMENTS 72.1%  CLINICAL HISTORY 56.5%

Two jobs here:

1. Strip the COMPARISON section. MR-RATE's process_long.ipynb instead *dropped*
   any report mentioning a prior, so each timepoint's report would stand alone.
   In BIND that would cost 48,035/81,666 I0001 reports (59%). Removing just the
   section keeps the row and still leaves the report free of prior-study
   references, which is what the pairing actually requires.

2. Read the exam type out of the report body, for `has_contrast` and as a
   cross-check on the `Type` column.

   Note on rows with `Type == "not available"` (31,893 I0001, 9,590 I0004):
   these are NOT recoverable. Measured 2026-08-04, every such row also has
   `Report_txt == "not available"` -- literally that string, not prose
   (Type-NA-but-report-present = 0 at both sites). There is no text to parse,
   so build_pairs.py drops them outright.
"""

import re

# Headers observed at either site. Order matters only for readability; the regex
# alternation is anchored on the colon.
SECTION_HEADERS = (
    "NARRATIVE",
    "TECHNIQUE",
    "COMPARISON STUDIES",
    "COMPARISON",
    "FINDINGS OF CLINICAL SIGNIFICANCE",
    "FINDINGS",
    "IMPRESSION",
    "END OF IMPRESSION",
    "ATTESTATION",
    "CLINICAL INDICATION",
    "CLINICAL HISTORY",
    "ADDITIONAL HISTORY",
    "HISTORY",
    "INDICATIONS",
    "INDICATION",
    "PROCEDURE COMMENTS",
    "RECOMMENDATION",
    "ACCESSION NUMBER",
    "SUMMARY",
    "SYNOPSIS FOR CLINICAL MANAGEMENT",
    "EXAM",
)

_HEADER_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:",
    re.IGNORECASE,
)

# Sections that describe a prior study or the dictation apparatus rather than the
# current exam's findings.
_COMPARISON_HEADERS = {"COMPARISON", "COMPARISON STUDIES"}

_EXAM_TYPE_RE = re.compile(
    r"\b((?:MRI?|CT|NM|PET|MRA)\s+[A-Z][A-Z /&-]{2,60}?)\s*(?:[:.]|$)"
)

_MR_BRAIN_RE = re.compile(r"\b(?:MRI?|MR)\s+(?:NS\s+)?(?:FAST\s+)?BRAIN\b", re.IGNORECASE)

# Exams that mention "brain" but are not the structural brain MRI we want.
_EXCLUDE_RE = re.compile(
    r"\b(ANGIOGRAPH|ANGIO|MRA|FUNCTIONAL|SPECTROSCOP|PERFUSION ONLY|DATSCAN"
    r"|OUTSIDE \(NO INTERPRETATION\)|PET|SPECT)\b",
    re.IGNORECASE,
)


def split_sections(text):
    """
    Report -> [(header_or_None, body)] in document order.

    Text before the first header is returned with a header of None.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    matches = list(_HEADER_RE.finditer(text))

    if not matches:
        return [(None, text)]

    sections = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        sections.append((match.group(1).upper(), body))

    return sections


def strip_comparison(text):
    """
    Remove COMPARISON sections, returning (cleaned_text, had_comparison).

    Reports with no header structure at all are returned unchanged; there is no
    reliable way to excise a prior-study reference from unstructured prose, and
    guessing would corrupt the findings.
    """
    sections = split_sections(text)

    if not sections:
        return text, False

    had_comparison = any(header in _COMPARISON_HEADERS for header, _ in sections)

    if not had_comparison:
        return text, False

    kept = []
    for header, body in sections:
        if header in _COMPARISON_HEADERS:
            continue
        kept.append(f"{header}: {body}" if header else body)

    return "\n\n".join(part for part in kept if part.strip()), True


def exam_type_from_report(text):
    """
    Recover the exam name from the report body.

    Used for the 31,142 I0001 rows whose `Type` column reads "not available".
    The TECHNIQUE/NARRATIVE section leads with the exam name
    ("TECHNIQUE: MRI BRAIN WITH AND WITHOUT CONTRAST"), so read that first and
    fall back to the first modality-prefixed phrase anywhere in the report.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    for header, body in split_sections(text):
        if header in {"TECHNIQUE", "NARRATIVE", "EXAM"} and body:
            match = _EXAM_TYPE_RE.search(body[:200])
            if match:
                return match.group(1).strip().upper()

    match = _EXAM_TYPE_RE.search(text[:400])

    return match.group(1).strip().upper() if match else ""


def is_mr_brain(exam_type):
    """True if an exam-type string names a structural brain MRI."""
    if not isinstance(exam_type, str) or not exam_type.strip():
        return False

    if _EXCLUDE_RE.search(exam_type):
        return False

    return bool(_MR_BRAIN_RE.search(exam_type))


def has_contrast(exam_type, text=""):
    """
    True if the exam is a with-and-without-contrast study.

    This gates the run-order T1ce fallback in generate_dual_bind.py: without an
    explicit statement that contrast was given, a later T1w series is just a later
    T1w series, not a post-contrast one.
    """
    blob = f"{exam_type or ''} {(text or '')[:400]}".upper()

    if re.search(r"\bW\s*/?\s*O\s+(?:IV\s+)?CONTRAST\b", blob) and "AND" not in blob:
        return False

    return bool(
        re.search(
            r"\bW(?:ITH)?\s+AND\s+W(?:ITH)?\s*/?O(?:UT)?\b|\bWWO\b"
            r"|\bWITH\s+AND\s+WITHOUT\b|\bW\s*&\s*WO\b",
            blob,
        )
    )


def _self_test():
    text = (
        "TECHNIQUE: MRI BRAIN WITH AND WITHOUT CONTRAST\n\n"
        "COMPARISON: CT head *****; MRI brain *****.\n\n"
        "FINDINGS:\n\nThere is a right transfrontal ventriculostomy catheter.\n\n"
        "IMPRESSION: Stable."
    )

    cleaned, had = strip_comparison(text)
    assert had
    assert "COMPARISON" not in cleaned
    assert "ventriculostomy" in cleaned
    assert "IMPRESSION" in cleaned

    assert exam_type_from_report(text) == "MRI BRAIN WITH AND WITHOUT CONTRAST"
    assert is_mr_brain("MRI BRAIN WITH AND WITHOUT CONTRAST")
    assert is_mr_brain("MR BRAIN WO IV CONTRAST")
    assert is_mr_brain("MRI NS FAST BRAIN WO CONTRAST LIMITED")
    assert not is_mr_brain("MRI ANGIO BRAIN")
    assert not is_mr_brain("MRA HEAD WITHOUT CONTRAST")
    assert not is_mr_brain("CT HEAD")
    assert not is_mr_brain("NM PET BRAIN ALZHEIMER DEMENTIA")

    assert has_contrast("MR BRAIN W AND WO CONTRAST")
    assert has_contrast("MRI BRAIN WWO CONTRAST")
    assert not has_contrast("MR BRAIN WO IV CONTRAST")
    assert not has_contrast("MRI BRAIN")

    # A report with no headers is returned untouched.
    plain = "No headers here, just prose about a prior study."
    assert strip_comparison(plain) == (plain, False)

    print("report_sections self-test passed")


if __name__ == "__main__":
    _self_test()
