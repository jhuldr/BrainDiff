"""Prior-report x image factorial — for any system conditioned on a prior report.

Decomposes where a longitudinal report generator's score actually comes from, by crossing two
interventions: the prior report (present / withheld) and the images (the patient's own / another
patient's). Four conditions, and the three quantities that matter:

    image effect            own scans  -  another patient's scans, at fixed report availability
    prior-report effect     report present - report withheld, at fixed image identity
    interaction             how much the image effect shrinks when the report is available

A system that reads the images loses score when they are swapped. One that paraphrases the prior
loses little, and its interaction is large and negative.

Nothing here is specific to BrainDiff. Supply four files, one per cell. Each may be either:

    {"rg_er": [...]}                per-report scores you already have -- no GPU, no text
    {"hyps": [...], "refs": [...]}  generations, scored here with RadGraph-XL

Rows must be index-aligned across the four files: cell i is the same study in each. With
`hyps`/`refs` that is checked by comparing the reference text; with bare score arrays it can
only be assumed, so keep the generation order fixed.

    python paper/probes/factorial_probe.py \\
        --present-own   real.json      --present-other   vis_roll.json \\
        --withheld-own  noreport.json  --withheld-other  nr_vis_roll.json

Reproducing the paper's Table 6 from the shipped caches:

    python paper/probes/factorial_probe.py \\
        --present-own  paper/cache/perreport/C_braindiff_real.json \\
        --present-other paper/cache/perreport/C_braindiff_visroll.json \\
        --withheld-own paper/cache/perreport/C_braindiff_noreport.json \\
        --withheld-other paper/cache/perreport/C_braindiff_nr_visroll.json
"""
import argparse
import json
import os

import numpy as np


def load(path):
    """-> (per-report score array, reference texts or None)."""
    d = json.load(open(path))
    if "rg_er" in d:
        return np.array(d["rg_er"], float), [r.strip() for r in d["refs"]] if "refs" in d else None
    if "hyps" not in d or "refs" not in d:
        raise SystemExit(f"{path}: expected 'rg_er', or 'hyps' and 'refs'")
    from radgraph import F1RadGraph
    f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl",
                    cuda=int(os.environ.get("RG_CUDA", "0")))
    hyps, refs = d["hyps"], d["refs"]
    keep = [i for i, (h, r) in enumerate(zip(hyps, refs)) if h and h.strip() and r and r.strip()]
    _, rl, *_ = f1(hyps=[hyps[i] for i in keep], refs=[refs[i] for i in keep])
    arr = np.full(len(hyps), np.nan)
    for j, i in enumerate(keep):
        arr[i] = rl[1][j]
    return arr, [r.strip() for r in refs]


def ci(v, resamples, seed):
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    bs = np.array([v[rng.choice(len(v), len(v), True)].mean() for _ in range(resamples)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), int(len(v))


def paired(a, b, resamples, seed):
    d = (a - b)[np.isfinite(a) & np.isfinite(b)]
    return ci(d, resamples, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for f in ("present-own", "present-other", "withheld-own", "withheld-other"):
        ap.add_argument(f"--{f}", required=True)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cells = {}
    refs = {}
    for key, path in [("present-own", a.present_own), ("present-other", a.present_other),
                      ("withheld-own", a.withheld_own), ("withheld-other", a.withheld_other)]:
        cells[key], refs[key] = load(path)

    n = {len(v) for v in cells.values()}
    assert len(n) == 1, f"cells differ in length: { {k: len(v) for k, v in cells.items()} }"
    known = [r for r in refs.values() if r is not None]
    if len(known) > 1:
        assert all(r == known[0] for r in known), \
            "reference texts differ between cells -- the four files are not index-aligned"

    R, S = a.resamples, a.seed
    print(f"n = {n.pop()} rows per cell, {R} resamples, seed {S}\n")
    print(f"{'':22s} {'own scans':>24s} {'other patient':>24s}")
    for cond in ("present", "withheld"):
        o = ci(cells[f"{cond}-own"], R, S)
        t = ci(cells[f"{cond}-other"], R, S)
        print(f"prior report {cond:9s} {o[0]:8.4f} [{o[1]:.4f}, {o[2]:.4f}] "
              f"{t[0]:8.4f} [{t[1]:.4f}, {t[2]:.4f}]")

    print()
    eff = {}
    for cond in ("present", "withheld"):
        m, lo, hi, k = paired(cells[f"{cond}-own"], cells[f"{cond}-other"], R, S)
        eff[cond] = cells[f"{cond}-own"] - cells[f"{cond}-other"]
        print(f"image effect, report {cond:9s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={k}")
    for img in ("own", "other"):
        m, lo, hi, k = paired(cells[f"present-{img}"], cells[f"withheld-{img}"], R, S)
        print(f"prior-report effect, {img:6s} scans {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={k}")
    m, lo, hi, k = ci(eff["present"] - eff["withheld"], R, S)
    print(f"\ninteraction (present - withheld image effect) {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={k}")


if __name__ == "__main__":
    main()
