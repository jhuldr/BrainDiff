"""Timepoint-reversal probe — for any system that writes longitudinal comparison reports.

The probe asks whether a model's asserted direction of change actually depends on which study
is the prior. Generate a report for a pair, generate a second with the two timepoints swapped,
and check whether the asserted direction flips. A grounded model flips; a model paraphrasing the
prior report is indifferent.

Nothing here is specific to BrainDiff. The input is a JSON list of generations:

    [{"gt": 3, "fwd": "Findings: ...", "rev": "Findings: ..."}, ...]

      gt   the 7-way ground-truth change label, as an index into CHANGE_CLASSES
      fwd  the report generated with the true (prior, current) ordering
      rev  the report generated with the two studies swapped

Produce that file with whatever model you like, then:

    python paper/probes/reversal_probe.py --reports mysystem.json
    python paper/probes/reversal_probe.py --reports armA.json armB.json --labels nocf cf

Definitions, unchanged from the paper: only ground-truth CHANGE cases count (New lesion,
Progressed, Improved, Resolved -- a reversal has no direction to flip on a Stable or
Mixed/unclear pair), and among those only the ones whose FORWARD report asserts a direction at
all. A signed flip is `sign(direction(rev)) != sign(direction(fwd))`.

Direction is assigned by the LLM-free rule classifier in `braindiff.eval.temporal_score`, so
scoring is deterministic and costs no GPU.
"""
import argparse
import json

import numpy as np

from braindiff.eval.temporal_score import change_class, change_class_v2, change_class_v5

# Indices into CHANGE_CLASSES = (Stable, New lesion, Indeterminate, Progressed, Improved,
# Mixed interval change, Resolved). Directional change cases only.
CHANGE_IDX = {1, 3, 4, 6}
POS = {"Progressed", "New lesion"}
NEG = {"Improved", "Resolved"}
CLASSIFIERS = {"v1": change_class, "v2": change_class_v2, "v5": change_class_v5}


def direction(report, cls):
    c = cls(report)
    return 1 if c in POS else (-1 if c in NEG else 0)


def flip_array(rows, cls):
    """-> float array, one entry per ground-truth change case.

    1.0 flipped, 0.0 held, NaN where the forward report asserts no direction and the pair is
    therefore uninformative. NaN rather than dropping keeps the array index-aligned across
    arms, so two systems can be compared pair-by-pair.
    """
    out = []
    for r in rows:
        if int(r["gt"]) not in CHANGE_IDX:
            continue
        df = direction(r["fwd"], cls)
        if df == 0:
            out.append(np.nan)
            continue
        out.append(1.0 if np.sign(direction(r["rev"], cls)) != np.sign(df) else 0.0)
    return np.array(out, float)


def ci(v, resamples=10000, seed=0):
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.choice(len(v), len(v), True)].mean() for _ in range(resamples)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), int(len(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reports", nargs="+", required=True,
                    help="one JSON file per system or arm, each a list of {gt, fwd, rev}")
    ap.add_argument("--labels", nargs="*", default=None, help="display names, in the same order")
    ap.add_argument("--classifier", choices=sorted(CLASSIFIERS), default="v2")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flips-out", nargs="*", default=None,
                    help="optional: write each flip array as JSON, for downstream analysis")
    a = ap.parse_args()

    labels = a.labels or [f.rsplit("/", 1)[-1] for f in a.reports]
    assert len(labels) == len(a.reports), "--labels must match --reports in length"
    cls = CLASSIFIERS[a.classifier]

    arrays = []
    print(f"classifier: change_class_{a.classifier}   {a.resamples} resamples, seed {a.seed}\n")
    print(f"{'arm':28s} {'flip rate':>22s} {'n':>6s} {'cases':>7s}")
    for label, path in zip(labels, a.reports):
        rows = json.load(open(path))
        f = flip_array(rows, cls)
        arrays.append(f)
        m, lo, hi, n = ci(f, a.resamples, a.seed)
        print(f"{label:28s} {m:8.4f} [{lo:.4f}, {hi:.4f}] {n:6d} {len(f):7d}")

    if len(arrays) == 2:
        fa, fb = arrays
        rng = np.random.default_rng(a.seed)
        va, vb = fa[np.isfinite(fa)], fb[np.isfinite(fb)]
        bs = np.array([vb[rng.choice(len(vb), len(vb), True)].mean()
                       - va[rng.choice(len(va), len(va), True)].mean() for _ in range(a.resamples)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"\n{'difference (unpaired)':28s} {vb.mean()-va.mean():+8.4f} [{lo:+.4f}, {hi:+.4f}]")
        both = np.isfinite(fa) & np.isfinite(fb)
        if both.sum():
            m, plo, phi, n = ci(fb[both] - fa[both], a.resamples, a.seed)
            print(f"{'difference (paired)':28s} {m:+8.4f} [{plo:+.4f}, {phi:+.4f}] {n:6d}")

    if a.flips_out:
        for path, arr in zip(a.flips_out, arrays):
            json.dump({"flips": arr.tolist()}, open(path, "w"))
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
