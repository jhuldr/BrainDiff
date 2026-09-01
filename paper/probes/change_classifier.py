"""Interval-change classifier — assign a direction of change to a radiology report.

A rule classifier, not a model: it reads the report's Impression, honours an explicit statement
of indeterminacy, and otherwise matches a 41-term change lexicon. No GPU, no API, no second
network whose own errors would contaminate whatever you are measuring. Deterministic, ~5 ms per
report, and it is what drives the reversal probe and the confusion figure.

Three rule sets:

    v1  the original: set-union over the whole report
    v2  restricted to the Impression, with an explicit-indeterminacy override   (the paper's)
    v5  v2 plus explicit mixed / unclear cue lists

Seven classes -- Stable, New lesion, Indeterminate, Progressed, Improved, Mixed interval change,
Resolved -- optionally collapsed to four by clinical direction:

    Stable · Improved (+Resolved) · Worsened (Progressed +New lesion) · Mixed/unclear

## Classify your own reports

    python paper/probes/change_classifier.py --reports mine.json
    python paper/probes/change_classifier.py --csv mine.csv --text-col report --out pred.json

`mine.json` is a list of strings, or of objects with a `report` field and optionally a `label`.
With labels present the tool also reports agreement: per-class recall, balanced accuracy and a
confusion matrix.

## Evaluate predictions you already have

    python paper/probes/change_classifier.py --pred pred.json --labels truth.json

Both are `{id: class}` maps. This is the path that needs no report text, and it is how the
paper's Figure 3 numbers are checked in `tests/test_probes.py`.
"""
import argparse
import csv as _csv
import json
import sys

from braindiff.eval.temporal_score import change_class, change_class_v2, change_class_v5

_csv.field_size_limit(10 ** 7)

CLASSIFIERS = {"v1": change_class, "v2": change_class_v2, "v5": change_class_v5}
SEVEN = ["Stable", "New lesion", "Indeterminate", "Progressed", "Improved",
         "Mixed interval change", "Resolved"]
COLLAPSE = {"Stable": "Stable",
            "Improved": "Improved", "Resolved": "Improved",
            "Progressed": "Worsened", "New lesion": "Worsened",
            "Mixed interval change": "Mixed/unclear", "Indeterminate": "Mixed/unclear"}
FOUR = ["Stable", "Improved", "Worsened", "Mixed/unclear"]


def collapse(c):
    return COLLAPSE.get(c, "Mixed/unclear")


def confusion(true, pred, keys, classes, four=True):
    """-> (matrix, per-class recall, balanced accuracy, n)."""
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    n = 0
    for k in keys:
        t, p = true.get(k), pred.get(k)
        if t is None or p is None:
            continue
        t, p = (collapse(t), collapse(p)) if four else (t, p)
        if t not in idx or p not in idx:
            continue
        m[idx[t]][idx[p]] += 1
        n += 1
    recall = [(m[i][i] / s if (s := sum(m[i])) else float("nan")) for i in range(len(classes))]
    valid = [r for r in recall if r == r]
    return m, recall, (sum(valid) / len(valid) if valid else float("nan")), n


def report_table(m, recall, bal, n, classes):
    w = max(len(c) for c in classes) + 2
    print(f"\n  n = {n}   balanced accuracy = {bal:.4f}\n")
    print(" " * (w + 2) + "".join(f"{c[:12]:>14s}" for c in classes) + f"{'recall':>10s}")
    for i, c in enumerate(classes):
        print(f"  {c:<{w}s}" + "".join(f"{v:14d}" for v in m[i]) + f"{recall[i]:10.3f}")


def load_reports(a):
    """-> (ids, reports, labels or None)."""
    if a.csv:
        rows = list(_csv.DictReader(open(a.csv)))
        ids = [r.get(a.id_col) or str(i) for i, r in enumerate(rows)]
        texts = [r[a.text_col] for r in rows]
        labels = [r[a.label_col] for r in rows] if a.label_col else None
        return ids, texts, labels
    d = json.load(open(a.reports))
    if d and isinstance(d[0], str):
        return [str(i) for i in range(len(d))], d, None
    ids = [str(r.get("id", i)) for i, r in enumerate(d)]
    texts = [r["report"] for r in d]
    labels = [r["label"] for r in d] if "label" in d[0] else None
    return ids, texts, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--reports", help="JSON: list of strings, or of {report, label?, id?}")
    src.add_argument("--csv", help="CSV of reports")
    src.add_argument("--pred", help="JSON {id: class} of predictions you already have")
    ap.add_argument("--labels", help="JSON {id: class} of ground truth (with --pred)")
    ap.add_argument("--text-col", default="report")
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--classifier", choices=sorted(CLASSIFIERS), default="v2")
    ap.add_argument("--classes", choices=["4", "7"], default="4")
    ap.add_argument("--out", help="write {id: class} predictions here")
    a = ap.parse_args()

    classes = FOUR if a.classes == "4" else SEVEN
    four = a.classes == "4"

    if a.pred:
        pred = json.load(open(a.pred))
        if not a.labels:
            print(json.dumps(pred, indent=2)[:2000])
            return
        true = json.load(open(a.labels))
        keys = [k for k in pred if k in true]
        m, rec, bal, n = confusion(true, pred, keys, classes, four)
        report_table(m, rec, bal, n, classes)
        return

    ids, texts, labels = load_reports(a)
    cls = CLASSIFIERS[a.classifier]
    pred = {i: cls(t) for i, t in zip(ids, texts)}
    print(f"classified {len(pred)} report(s) with change_class_{a.classifier}")
    if a.out:
        json.dump(pred, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    elif not labels:
        for i in ids[:20]:
            print(f"  {i:>16s}  {collapse(pred[i]) if four else pred[i]}")
        if len(ids) > 20:
            print(f"  ... {len(ids)-20} more (use --out to write them all)")

    if labels:
        true = dict(zip(ids, labels))
        m, rec, bal, n = confusion(true, pred, ids, classes, four)
        report_table(m, rec, bal, n, classes)


if __name__ == "__main__":
    main()
