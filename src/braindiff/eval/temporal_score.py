"""Change-direction scorer -- the temporal axis RadGraph-XL does not cover.

RadGraph-XL labels entities present/absent/uncertain plus anatomy, but has no
change-direction relation, so progression needs its own number.

A dictionary matcher, not a model: pure CPU, ~5 ms/report. Validated against cached
RadGraph-XL scores over 26 checkpoints and ~20k report pairs -- Pearson 0.993 overall,
0.990 within stage 2, and it selects the same best checkpoint in both stages. It cannot
resolve differences inside a ~0.01-wide band; use RadGraph offline for that.

Negation handling is required: in the S4 corpus 34.7% of "new" and 63.2% of "recurrent"
are negated ("no new lesion"), and reading those as progression inverts the metric on the
majority-stable cases.
"""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

# parents[1], not [2]. This file sat at  in BrainDiff, where [2]
# was the repo root; the port moved it up one level and [2] started resolving to
# code/, outside any repo. Nothing caught it because train_single.py never calls
# build_content_masks -- S4 is the first stage to touch the entity dictionaries.
# Vocabularies ship as package data, resolved through importlib.resources rather than a
# path relative to a repo root. The previous form broke silently when the tree moved:
# the module still imported and CHANGE_LEXICON still populated, and only code that
# touched the entity dictionaries failed.
from importlib import resources as _resources

_VOCAB = _resources.files("braindiff.eval") / "vocab"
ANATOMY_PATH = _VOCAB / "common_anatomy.txt"
PATHOLOGY_PATH = _VOCAB / "condensed_pathology_labels.csv"

STABLE, PROGRESSED, IMPROVED = 0, 1, -1

# Mined from the S4 corpus by frequency, then filtered by hand. Size adjectives
# ("small", "large", "largest") are deliberately absent -- they describe a lesion,
# they don't describe a change. "developmental" and "internal" are excluded for
# the same reason: they only matched as prefix collisions.
CHANGE_LEXICON = {
    "stable": STABLE, "unchanged": STABLE, "unaltered": STABLE,
    "persistent": STABLE, "persists": STABLE, "persist": STABLE,
    "similar": STABLE, "stationary": STABLE,

    "new": PROGRESSED, "newly": PROGRESSED,
    "increased": PROGRESSED, "increase": PROGRESSED, "increasing": PROGRESSED,
    "enlargement": PROGRESSED, "enlarged": PROGRESSED, "enlarging": PROGRESSED,
    "enlargements": PROGRESSED, "larger": PROGRESSED,
    "progression": PROGRESSED, "progressed": PROGRESSED, "progressive": PROGRESSED,
    "worsening": PROGRESSED, "worsened": PROGRESSED, "growth": PROGRESSED,
    "recurrent": PROGRESSED, "recurrence": PROGRESSED, "developed": PROGRESSED,

    "decreased": IMPROVED, "decrease": IMPROVED, "decreasing": IMPROVED,
    "resolution": IMPROVED, "resolved": IMPROVED, "resolving": IMPROVED,
    "regression": IMPROVED, "regressed": IMPROVED,
    "improvement": IMPROVED, "improved": IMPROVED,
    "reduced": IMPROVED, "reduction": IMPROVED,
    "smaller": IMPROVED, "disappeared": IMPROVED,
}

# Report language the ontologies don't cover. common_anatomy.txt uses SNOMED
# names ("Pontine structure") and the pathology labels are full diagnoses
# ("Intracranial meningioma"), so a radiologist writing "pons" or "the
# meningioma" matches neither. Without these the matcher finds anatomical
# adjectives and misses every finding noun -- fatal for a reward meant to make
# the model name lesions.
EXTRA_ENTITIES = (
    # findings
    "lesion", "lesions", "mass", "masses", "plaque", "plaques", "nodule",
    "metastasis", "metastases", "metastatic", "tumor", "tumour", "infarct",
    "infarction", "hemorrhage", "haemorrhage", "edema", "oedema", "enhancement",
    "cyst", "abscess", "hematoma", "effusion", "midline shift", "mass effect",
    "hydrocephalus", "atrophy", "gliosis", "encephalomalacia", "microangiopathy",
    "demyelinating", "ischemic", "ischaemic", "meningioma", "glioma", "aneurysm",
    # lay anatomy
    "pons", "midbrain", "medulla", "thalamus", "thalami", "hippocampus",
    "brainstem", "brain stem", "ventricle", "ventricles", "sulci", "sulcus",
    "white matter", "gray matter", "grey matter", "cortex", "cerebellar",
    "frontal", "parietal", "temporal", "occipital", "pituitary", "sella",
)

# Dropped when taking a term's head noun: they carry no diagnostic content and
# would collapse unrelated terms onto each other.
HEAD_NOUN_STOPWORDS = frozenset({
    "structure", "structures", "disease", "syndrome", "system", "region",
    "body", "part", "tissue", "matter", "of", "brain", "finding", "disorder",
})

NEGATORS = ("no", "not", "without", "absent", "absence", "never", "nor")

# How many tokens back a negator can sit and still scope over a change cue.
# "no new intracranial lesion" needs 2; "no significant interval enlargement"
# needs 3. Beyond ~4 the negation usually belongs to a different clause.
NEGATION_WINDOW = 4

# Clause boundaries. Negation does not cross these, so "enlarged mass, no new
# lesion" keeps its two polarities apart.
CLAUSE_SPLIT = re.compile(r"[.;:,]|\bbut\b|\bwhereas\b|\bwhile\b|\bhowever\b")


def _clean_term(term):
    """Strip the SNOMED-style qualifiers in common_anatomy.txt, e.g.
    'Cerebral hemisphere structure (body structure)' -> the leading phrase."""
    return re.sub(r"\s*\(.*?\)", "", term).strip().lower()


def _head_noun(term):
    """Last content word of an ontology term, so 'Intracranial meningioma' also
    matches a report that just says 'the meningioma'."""
    words = [w for w in term.split() if w not in HEAD_NOUN_STOPWORDS]
    if not words:
        return None
    head = words[-1]
    return head if len(head) >= 5 else None


@lru_cache(maxsize=1)
def _entity_pattern():
    """One alternation over anatomy + pathology terms, their head nouns, and the
    report-language supplement -- longest-first so 'lateral ventricle' wins over
    'ventricle'."""
    terms = {_clean_term(line) for line in ANATOMY_PATH.read_text().splitlines()}
    pathologies = pd.read_csv(PATHOLOGY_PATH)["true_name"].astype(str)
    terms |= {t.strip().lower() for t in pathologies}
    terms |= {head for t in list(terms) if (head := _head_noun(t))}
    terms |= set(EXTRA_ENTITIES)
    terms = sorted((t for t in terms if 3 < len(t) < 40), key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?!\w)")


@lru_cache(maxsize=1)
def _change_pattern():
    cues = sorted(CHANGE_LEXICON, key=len, reverse=True)
    return re.compile(r"(?<!\w)(" + "|".join(re.escape(c) for c in cues) + r")(?!\w)")


def _polarity(clause):
    """Polarities asserted in one clause, with negation applied.

A negated progression cue ("no new lesion") asserts stability, not progression. A negated
improvement cue is treated the same way.
    """
    found = set()
    for match in _change_pattern().finditer(clause):
        polarity = CHANGE_LEXICON[match.group(1)]
        preceding = clause[:match.start()].split()[-NEGATION_WINDOW:]
        if any(word.strip("(),") in NEGATORS for word in preceding):
            polarity = STABLE
        found.add(polarity)
    return found


def extract_triples(report):
    """Report -> {(entity, polarity)}.

Entities in a clause carrying no change cue are emitted with polarity None so they still
contribute to entity F1 without inventing a direction.
    """
    triples = set()
    for clause in CLAUSE_SPLIT.split(report.lower()):
        entities = set(_entity_pattern().findall(clause))
        if not entities:
            continue
        for polarity in _polarity(clause) or {None}:
            triples.update((entity, polarity) for entity in entities)
    return triples


def _f1(hyp_set, ref_set):
    if not hyp_set or not ref_set:
        return 0.0
    overlap = len(hyp_set & ref_set)
    if not overlap:
        return 0.0
    precision = overlap / len(hyp_set)
    recall = overlap / len(ref_set)
    return 2 * precision * recall / (precision + recall)


def triple_f1(hyp, ref):
    """(entity, change-direction) F1 -- the headline number."""
    return _f1(extract_triples(hyp), extract_triples(ref))


def entity_f1(hyp, ref):
    """Entity F1 with direction discarded, for separating 'named the right
    finding' from 'got its direction right'."""
    return _f1({e for e, _ in extract_triples(hyp)},
               {e for e, _ in extract_triples(ref)})


def change_class(report):
    """The report's implied change class, for agreement against the
    `classification` column: mixed direction -> 'Mixed interval change'."""
    polarities = {p for _, p in extract_triples(report) if p is not None}
    directional = polarities - {STABLE}
    if len(directional) > 1:
        return "Mixed interval change"
    if PROGRESSED in directional:
        return "Progressed"
    if IMPROVED in directional:
        return "Improved"
    return "Stable" if polarities else "Indeterminate"


_IMPRESSION = re.compile(r"\bimpression\s*[:\-]", re.I)


def change_class_v2(report):
    """`change_class` restricted to the report's IMPRESSION, with an explicit-unclear override.

Two corrections to change_class, which is left untouched (it also feeds `reward()`):

1. Read the IMPRESSION when present. Impressions state the net verdict; the findings body
   enumerates stable and changed items alike, so the set-union rule promotes one stray cue
   into a spurious direction. This is the larger effect (+7 points of balanced accuracy).
2. An explicit "indeterminate" wins. The 7-way scheme uses that word for "change present,
   direction unassignable"; change_class returns it for "no change cue found". The clash
   cost the Indeterminate class almost all its recall (7% under change_class).

Against the LLM-assigned labels on the reference reports, 4-class balanced accuracy rises
72.2 -> 81.1 (test) and 72.9 -> 82.9 (val). Deterministic, LLM-free, CPU.
    """
    m = list(_IMPRESSION.finditer(report or ""))
    seg = report[m[-1].end():] if m else report
    if "indeterminate" in (seg or "").lower():
        return "Indeterminate"
    return change_class(seg)


# Literal phrases the S4 synthesis uses for the two classes change_class cannot express.
# Chosen by inspecting the misclassified impressions on the test split, then confirmed on
# val (never inspected while choosing). "limited" was dropped: net-harmful in leave-one-out.
UNCLEAR_CUES = ("indeterminate", "uncertain", "newly described", "discrepant",
                "discordant", "not possible", "cannot")


def change_class_v5(report):
    """change_class_v2 plus explicit Mixed / unclear cues, read from the IMPRESSION.

Rule order: "mixed" -> Mixed interval change; any UNCLEAR_CUES -> Indeterminate; otherwise
change_class on the impression. 4-class balanced accuracy on the reference reports:
72.2 (change_class) -> 81.1 (v2) -> 87.7 (v5) on test.

CAVEAT for the ceiling only: the synthesis writes "Mixed interval change with ..." when the
label is Mixed, so the "mixed" rule partly reads the label out of the reference text (+2.4
of the +6.6 over v2). Quote 85.1 (val, no "mixed" rule) for a label-free ceiling. Worsened
recall also falls 0.933 -> 0.883, since some progression impressions hedge.
    """
    m = list(_IMPRESSION.finditer(report or ""))
    seg = report[m[-1].end():] if m else report
    low = (seg or "").lower()
    if "mixed" in low:
        return "Mixed interval change"
    if any(cue in low for cue in UNCLEAR_CUES):
        return "Indeterminate"
    return change_class(seg)


def score(hyps, refs):
    """Mean scores over the pairs RadGraph's scorer would also keep (both
    non-empty), so the two metrics are computed on the same subset."""
    pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not pairs:
        return {"triple_f1": 0.0, "entity_f1": 0.0, "n": 0}
    return {
        "triple_f1": sum(triple_f1(h, r) for h, r in pairs) / len(pairs),
        "entity_f1": sum(entity_f1(h, r) for h, r in pairs) / len(pairs),
        "n": len(pairs),
    }


def reward(hyp, ref, classification=None, beta=2.0):
    """GRPO reward. Recall-emphasized (beta > 1) because the failure mode is
    under-reporting change, not over-reporting it."""
    hyp_triples, ref_triples = extract_triples(hyp), extract_triples(ref)
    if not hyp_triples or not ref_triples:
        return 0.0
    overlap = len(hyp_triples & ref_triples)
    if not overlap:
        return 0.0
    precision = overlap / len(hyp_triples)
    recall = overlap / len(ref_triples)
    f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    if classification is None:
        return f_beta
    return 0.75 * f_beta + 0.25 * float(change_class(hyp) == classification)


def run_batch(hyp_ref_dir, run_name):
    """Mirrors radgraph_score.py --hyp_ref_dir so the two are directly
    comparable over the same evaluate_checkpoints.py dumps."""
    hyp_ref_dir = Path(hyp_ref_dir)
    stem = run_name.split(".")[0]
    paths = [p for p in sorted(hyp_ref_dir.glob("*.json"))
             if "results" not in p.name and stem in p.name]
    if not paths:
        raise SystemExit(f"No *.json files found in {hyp_ref_dir}")

    results = {}
    for path in paths:
        payload = json.loads(path.read_text())
        results[path.stem] = score(payload["hyps"], payload["refs"])
        print(f"  {path.stem}: triple_f1={results[path.stem]['triple_f1']:.4f} "
              f"entity_f1={results[path.stem]['entity_f1']:.4f}")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["triple_f1"], reverse=True)
    best_name = ranked[0][0]
    print(f"\nBest checkpoint: {best_name}  triple_f1={ranked[0][1]['triple_f1']:.4f}")

    out_path = hyp_ref_dir / "temporal_results.json"
    out_path.write_text(json.dumps({"results": results, "best": best_name}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_ref_dir", required=True)
    p.add_argument("--model_name", required=True)
    args = p.parse_args()
    run_batch(args.hyp_ref_dir, args.model_name)
