"""Change-direction scorer -- the temporal axis RadGraph-XL does not cover.

Dictionary matcher, CPU only. Negation-aware: "no new lesion" asserts stability, not progression.
"""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

# pathology_names.txt = the 29 unique MR-RATE pathology names (the per-study label csv is not redistributed).
VOCAB = Path(__file__).resolve().parent / "vocab"
ANATOMY_PATH = VOCAB / "common_anatomy.txt"
PATHOLOGY_PATH = VOCAB / "pathology_names.txt"

STABLE, PROGRESSED, IMPROVED = 0, 1, -1

# Size adjectives ("small", "large") are deliberately absent: they describe a lesion, not a change.
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

# Report words the ontologies miss (SNOMED "Pontine structure" vs "pons"); without these no finding noun matches.
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

# Dropped when taking a head noun: they would collapse unrelated terms onto each other.
HEAD_NOUN_STOPWORDS = frozenset({
    "structure", "structures", "disease", "syndrome", "system", "region",
    "body", "part", "tissue", "matter", "of", "brain", "finding", "disorder",
})

NEGATORS = ("no", "not", "without", "absent", "absence", "never", "nor")

# Tokens back a negator can scope over a change cue ("no significant interval enlargement" needs 3).
NEGATION_WINDOW = 4

# Clause boundaries; negation does not cross them.
CLAUSE_SPLIT = re.compile(r"[.;:,]|\bbut\b|\bwhereas\b|\bwhile\b|\bhowever\b")


def _clean_term(term):
    """Strip SNOMED qualifiers: 'Cerebral hemisphere structure (body structure)' -> the leading phrase."""
    return re.sub(r"\s*\(.*?\)", "", term).strip().lower()


def _head_noun(term):
    """Last content word, so 'Intracranial meningioma' also matches 'the meningioma'."""
    words = [w for w in term.split() if w not in HEAD_NOUN_STOPWORDS]
    if not words:
        return None
    head = words[-1]
    return head if len(head) >= 5 else None


@lru_cache(maxsize=1)
def _entity_pattern():
    """One longest-first alternation over anatomy + pathology terms, head nouns and the supplement."""
    terms = {_clean_term(line) for line in ANATOMY_PATH.read_text().splitlines()}
    pathologies = PATHOLOGY_PATH.read_text().splitlines()
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
    """Polarities asserted in one clause; a negated progression or improvement cue asserts stability."""
    found = set()
    for match in _change_pattern().finditer(clause):
        polarity = CHANGE_LEXICON[match.group(1)]
        preceding = clause[:match.start()].split()[-NEGATION_WINDOW:]
        if any(word.strip("(),") in NEGATORS for word in preceding):
            polarity = STABLE
        found.add(polarity)
    return found


def extract_triples(report):
    """Report -> {(entity, polarity)}; entities in a clause without a change cue get polarity None."""
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
    """Entity F1 with direction discarded."""
    return _f1({e for e, _ in extract_triples(hyp)},
               {e for e, _ in extract_triples(ref)})


def change_class(report):
    """Implied change class for agreement with `classification`; mixed direction -> 'Mixed interval change'."""
    polarities = {p for _, p in extract_triples(report) if p is not None}
    directional = polarities - {STABLE}
    if len(directional) > 1:
        return "Mixed interval change"
    if PROGRESSED in directional:
        return "Progressed"
    if IMPROVED in directional:
        return "Improved"
    return "Stable" if polarities else "Indeterminate"


def score(hyps, refs):
    """Mean scores over pairs with both reports non-empty (the subset RadGraph keeps)."""
    pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not pairs:
        return {"triple_f1": 0.0, "entity_f1": 0.0, "n": 0}
    return {
        "triple_f1": sum(triple_f1(h, r) for h, r in pairs) / len(pairs),
        "entity_f1": sum(entity_f1(h, r) for h, r in pairs) / len(pairs),
        "n": len(pairs),
    }


def reward(hyp, ref, classification=None, beta=2.0):
    """GRPO reward; recall-emphasized (beta > 1)."""
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
    """Mirrors radgraph_score.py --hyp_ref_dir over the same dumps."""
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


# F1temp (paper Table 2): Libra's temporal-entity F1, verbatim from libra/eval/temporal_f1.py (github.com/X-iZhang/Libra,
# commit 0d6c56a4, Apache-2.0). Not the dictionary scorer above.

KEYWORDS = {
    "bigger", "change", "cleared", "constant", "decrease", "decreased", "decreasing", "elevated", "elevation",
    "enlarged", "enlargement", "enlarging", "expanded", "greater", "growing", "improved", "improvement",
    "improving", "increase", "increased", "increasing", "larger", "new", "persistence", "persistent",
    "persisting", "progression", "progressive", "reduced", "removal", "resolution", "resolved", "resolving",
    "smaller", "stability", "stable", "stably", "unchanged", "unfolded", "worse", "worsen", "worsened",
    "worsening", "unaltered"
}


def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[_-]+', ' ', text)
    text = re.sub(r'\(___, __, __\)', '', text)
    text = re.sub(r'---, ---, ---', '', text)
    text = re.sub(r'\(__, __, ___\)', '', text)
    text = re.sub(r'[_-]+', ' ', text)
    text = re.sub(r'[^\w\s.,:;()-]', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text


def extract_entities(text, keywords):
    text = clean_text(text)
    pattern = r'\b(' + '|'.join(re.escape(word) for word in keywords) + r')\b'
    return {match.group().lower() for match in re.finditer(pattern, text.lower())}


def calculate_tem_score(prediction_text, reference_text, epsilon=1e-10):
    """-> {"f1", "prediction_entities", "reference_entities"} (Libra calculate_tem_score)."""
    if isinstance(reference_text, list):
        reference_entities = set()
        for ref in reference_text:
            reference_entities.update(extract_entities(ref, KEYWORDS))
    else:
        reference_entities = extract_entities(reference_text, KEYWORDS)
    prediction_entities = extract_entities(prediction_text, KEYWORDS)
    if len(reference_entities) == 0:
        f1 = 1.0 if len(prediction_entities) == 0 else epsilon
        return {"f1": f1, "prediction_entities": prediction_entities, "reference_entities": reference_entities}
    true_positives = len(prediction_entities & reference_entities)
    precision = (true_positives + epsilon) / (len(prediction_entities) + epsilon)
    recall = (true_positives + epsilon) / (len(reference_entities) + epsilon)
    tem_score = (2 * precision * recall) / (precision + recall + epsilon)
    return {"f1": tem_score, "prediction_entities": prediction_entities, "reference_entities": reference_entities}


def f1temp(refs, hyps):
    """Per-pair F1temp, as baseline_metrics.scorer("f1temp")."""
    return [calculate_tem_score(h, r)["f1"] for r, h in zip(refs, hyps)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_ref_dir", required=True)
    p.add_argument("--model_name", required=True)
    args = p.parse_args()
    run_batch(args.hyp_ref_dir, args.model_name)
