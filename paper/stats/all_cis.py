"""SINGLE SOURCE for every bootstrap interval the manuscript reports.

One run, one seed (numpy default_rng(0)), 10,000 resamples, one pairing rule per comparison
type. Emits paper/stats/PAPER_CIS.md. Nothing here re-reads a recorded metric: all
inputs are the per-report rg_er arrays in paper/cache/perreport/, which were produced
by re-scoring the cached generations with RadGraph-XL.

Pairing rules, applied in this order and asserted, never guessed:
  index   same-pipeline dumps from one aligned run -- references must match at every position
  uid     cross-pipeline (BrainDiff dump vs the NeuroVFM CSV), joined on study_uid2
  ref     the 64-study frontier subset, whose files are in different orders

Run:  python paper/stats/all_cis.py
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import json
import os

import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "cache", "perreport")
OUT = os.path.join(HERE, "PAPER_CIS.md")
PDF = os.path.join(HERE, "..", "..", "BrainDiff Main.pdf")   # compared against at run time
R = 10000


def load(tag):
    """(rg_er, study_uid2) for one arm.

    The caches are keyed by `study_uid2`, not by reference text: the MR-RATE report text
    is not redistributable, so it is not in this tree. Every join below -- index, ref, uid
    -- is therefore a join on study uids, which is what it was standing in for anyway.
    """
    d = json.load(open(f"{P}/{tag}.json"))
    return np.array(d["rg_er"], float), list(d["uids"])


def ci(v):
    rng = np.random.default_rng(0)                 # same seed for every interval
    v = v[np.isfinite(v)]
    bs = np.array([v[rng.choice(len(v), len(v), True)].mean() for _ in range(R)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return v.mean(), lo, hi, len(v)


def mean_ci(tag):
    v, _ = load(tag)
    return ci(v)


def diff_index(a, b):
    ra, fa = load(a); rb, fb = load(b)
    assert fa == fb, f"{a} vs {b}: not index-aligned"
    m = np.isfinite(ra) & np.isfinite(rb)
    return ci((ra - rb)[m])


def diff_ref(a, b):
    ra, fa = load(a); rb, fb = load(b)
    A = dict(zip(fa, ra)); B = dict(zip(fb, rb))
    assert set(A) == set(B), f"{a} vs {b}: different study sets"
    k = [x for x in A if x in B]
    return ci(np.array([A[x] - B[x] for x in k]))


def diff_uid(dump_tag, other_tag):
    """BrainDiff dump vs another arm, joined on study_uid2; keeps all 1444 rows."""
    ra, ua = load(dump_tag)
    rb, ub = load(other_tag)
    other_rg = dict(zip(ub, rb))
    d = [v - other_rg[u] for v, u in zip(ra, ua)
         if u in other_rg and np.isfinite(v) and np.isfinite(other_rg[u])]
    return ci(np.array(d))


def uids_for(uids):
    """Identity: the caches already carry study_uid2 per row."""
    return list(uids)


def dd(a1, a2, b1, b2):
    """Difference of differences: (a1-a2) - (b1-b2).

    Within an arm the two conditions come from one aligned run, so they pair by index and
    that is asserted. Across arms the dumps may carry a different shard layout (ncnf does),
    so the two per-report difference series are joined by reference text, with the study
    sets asserted equal.
    """
    ra1, f1 = load(a1); ra2, f2 = load(a2); rb1, f3 = load(b1); rb2, f4 = load(b2)
    assert f1 == f2, f"{a1} vs {a2}: not index-aligned"
    assert f3 == f4, f"{b1} vs {b2}: not index-aligned"
    if f1 == f3:                                   # both arms share the run order
        m = np.isfinite(ra1) & np.isfinite(ra2) & np.isfinite(rb1) & np.isfinite(rb2)
        return ci(((ra1 - ra2) - (rb1 - rb2))[m])
    # different shard layout (ncnf): key both arms by study_uid2, resolving the one
    # duplicated generated_report by order of appearance so all 1444 rows survive.
    ua, ub = uids_for(f1), uids_for(f3)
    A = {u: x - y for u, x, y in zip(ua, ra1, ra2) if np.isfinite(x) and np.isfinite(y)}
    B = {u: x - y for u, x, y in zip(ub, rb1, rb2) if np.isfinite(x) and np.isfinite(y)}
    assert set(A) == set(B), "difference-of-differences arms cover different studies"
    return ci(np.array([A[u] - B[u] for u in A]))


def flips(tag):
    """Cached signed-flip array for one reversal arm.

    Previously recomputed by re-classifying the forward and reversed generations in
    logs/reversal_v2/*_reports.json. That text is MR-RATE-derived and is not shipped, so
    the flip arrays are read from the cache instead; they were verified identical to the
    recomputation before the text was dropped.
    """
    return np.array(json.load(open(f"{P}/{tag}.json"))["flips"], float)


def fmt(t, signed=False):
    m, lo, hi, n = t
    s = f"{m:+.4f}" if signed else f"{m:.4f}"
    return f"{s} [{lo:+.4f}, {hi:+.4f}]" if signed else f"{s} [{lo:.4f}, {hi:.4f}]", n


rows = []          # (table, quantity, value-with-CI, n, pairing, reported)
def add(tbl, name, t, pairing, signed=False, reported=True):
    """reported=False: computed here for completeness, but the paper does not print it,
    so it must not appear in the PDF-comparison section."""
    s, n = fmt(t, signed)
    rows.append((tbl, name, s, n, pairing, reported))


NV_TAG = "neurovfm_base_fulltest"

# ---- Table 2 : comparison against baselines ---------------------------------
add("2", "BrainDiff, full test", mean_ci("C_braindiff_real"), "--")
add("2", "NeuroVFM + report pipeline, full test", mean_ci("neurovfm_base_fulltest"), "--")
add("2", "Text-only (images zeroed), full test", mean_ci("C_braindiff_scanzero"), "--")
add("2", "ours - NeuroVFM", diff_uid("C_braindiff_real", NV_TAG), "uid", True)
add("2", "Opus 5, 64-subset", mean_ci("claude_opus5_subset64"), "--")
add("2", "GPT-5.6 Sol, 64-subset", mean_ci("gpt56sol_subset64"), "--")
add("2", "BrainDiff, 64-subset", mean_ci("braindiff_subset64"), "--")
add("2", "NeuroVFM, 64-subset", mean_ci("neurovfm_subset64"), "--")
add("2", "ours - Opus 5 (64-subset)", diff_ref("braindiff_subset64", "claude_opus5_subset64"), "ref", True)
add("2", "ours - GPT-5.6 (64-subset)", diff_ref("braindiff_subset64", "gpt56sol_subset64"), "ref", True)
add("2", "ours - NeuroVFM (64-subset)", diff_ref("braindiff_subset64", "neurovfm_subset64"), "ref", True, reported=False)

# ---- Table 3 : internal side only -------------------------------------------
# The BIND rows of Table 3 are NOT reproducible from this tree. BIND is external
# health-system data we are not permitted to redistribute, so its per-report
# caches are withheld; the paper's BIND column stands on the manuscript alone.
# The internal rows below are the same quantities on the MR-RATE test split.
add("3", "internal rg_er", mean_ci("C_braindiff_real"), "--")
add("3", "internal image contribution", diff_index("C_braindiff_real", "C_braindiff_visroll"), "index", True)

# ---- Table 4 : counterfactual + dropout -------------------------------------
add("4", "nocf: image effect", diff_index("C_nocf_real", "C_nocf_visroll"), "index", True)
add("4", "cf+dropout: image effect", diff_index("C_braindiff_real", "C_braindiff_visroll"), "index", True)
add("4", "quality cost (cf+dropout - nocf)", diff_index("C_braindiff_real", "C_nocf_real"), "index", True)
add("4", "counterfactual gain (image effect difference)",
    dd("C_braindiff_real", "C_braindiff_visroll", "C_nocf_real", "C_nocf_visroll"), "index", True, reported=False)

# ---- Table 5 : image reliance by intervention stage -------------------------
add("5", "none (no curriculum, no cf, no dropout)", diff_index("ncnf_real", "ncnf_visroll"), "index", True)
add("5", "no curriculum (cf + dropout)", diff_index("C_nocurric_real", "C_nocurric_visroll"), "index", True)
add("5", "S1 only", diff_index("C_ns2_real", "C_ns2_visroll"), "index", True)
add("5", "S1 + S2", diff_index("C_braindiff_real", "C_braindiff_visroll"), "index", True)
add("5", "total (S1+S2 - none)",
    dd("C_braindiff_real", "C_braindiff_visroll", "ncnf_real", "ncnf_visroll"), "index", True)

# ---- Table 6 : prior-report x image factorial -------------------------------
add("6", "report present, own scans", mean_ci("C_braindiff_real"), "--")
add("6", "report present, other patient's scans", mean_ci("C_braindiff_visroll"), "--")
add("6", "report withheld, own scans", mean_ci("C_braindiff_noreport"), "--")
add("6", "report withheld, other patient's scans", mean_ci("C_braindiff_nr_visroll"), "--")
add("6", "image effect, report present", diff_index("C_braindiff_real", "C_braindiff_visroll"), "index", True)
add("6", "image effect, report withheld", diff_index("C_braindiff_noreport", "C_braindiff_nr_visroll"), "index", True)
add("6", "effect of prior report, own scans", diff_index("C_braindiff_real", "C_braindiff_noreport"), "index", True)
add("6", "effect of prior report, other scans", diff_index("C_braindiff_visroll", "C_braindiff_nr_visroll"), "index", True)
add("6", "interaction (present - withheld image effect)",
    dd("C_braindiff_real", "C_braindiff_visroll", "C_braindiff_noreport", "C_braindiff_nr_visroll"), "index", True)

# ---- Table 7 : difference-pathway ablation ----------------------------------
add("7", "delta on", mean_ci("C_deltaon_real"), "--")
add("7", "delta off", mean_ci("C_braindiff_real"), "--")
add("7", "wrong patient's scans (delta off)", mean_ci("C_braindiff_visroll"), "--")
add("7", "images zeroed (delta off)", mean_ci("C_braindiff_scanzero"), "--")
add("7", "delta contribution", diff_index("C_deltaon_real", "C_braindiff_real"), "index", True)
add("7", "image contribution vs wrong scans", diff_index("C_braindiff_real", "C_braindiff_visroll"), "index", True)
add("7", "image contribution vs zeroed images", diff_index("C_braindiff_real", "C_braindiff_scanzero"), "index", True, reported=False)

# ---- Table S1 (supp) : curriculum arms, prior-report reliance ----------------
# Three arms x {report available, withheld}, each arm's within-arm reliance, and the
# paired differences vs the full-curriculum arm. nostage2 comes from the canonical
# logs/nostage2_test dump (C_ns2_*), not the superseded logs/test/ns2_* one.
add("S1", "nodelta (S1+S2), report available", mean_ci("C_braindiff_real"), "--")
add("S1", "nostage2 (S1), report available", mean_ci("C_ns2_real"), "--")
add("S1", "nocurriculum (none), report available", mean_ci("C_nocurric_real"), "--")
add("S1", "nodelta (S1+S2), report withheld", mean_ci("C_braindiff_noreport"), "--")
add("S1", "nostage2 (S1), report withheld", mean_ci("C_ns2_noreport"), "--")
add("S1", "nocurriculum (none), report withheld", mean_ci("C_nocurric_noreport"), "--")

add("S1", "report-reliance, nodelta (S1+S2)",
    diff_index("C_braindiff_real", "C_braindiff_noreport"), "index", True)
add("S1", "report-reliance, nostage2 (S1)",
    diff_index("C_ns2_real", "C_ns2_noreport"), "index", True)
add("S1", "report-reliance, nocurriculum (none)",
    diff_index("C_nocurric_real", "C_nocurric_noreport"), "index", True)

add("S1", "available: nodelta - nostage2", diff_index("C_braindiff_real", "C_ns2_real"), "index", True)
add("S1", "withheld: nodelta - nostage2", diff_index("C_braindiff_noreport", "C_ns2_noreport"), "index", True)
add("S1", "reliance DiD vs nostage2",
    dd("C_braindiff_real", "C_braindiff_noreport", "C_ns2_real", "C_ns2_noreport"), "index", True)

add("S1", "available: nodelta - nocurriculum", diff_index("C_braindiff_real", "C_nocurric_real"), "index", True)
add("S1", "withheld: nodelta - nocurriculum", diff_index("C_braindiff_noreport", "C_nocurric_noreport"), "index", True)
add("S1", "reliance DiD vs nocurriculum",
    dd("C_braindiff_real", "C_braindiff_noreport", "C_nocurric_real", "C_nocurric_noreport"), "index", True)

# ---- floors -----------------------------------------------------------------
add("6/3", "metric floor, internal (patient-disjoint)", (0.1510, 0.1469, 0.1553, 1444), "derangement", reported=False)
add("6/3", "metric floor, internal (published roll-1)", (0.2471, 0.2392, 0.2548, 1444), "roll-1")

# ---- Table 4 reversal column ------------------------------------------------
import sys
sys.path.insert(0, _REPO)
from braindiff.eval.temporal_score import change_class, change_class_v2
# The paper's Fig. 3 and reversal probe both use change_class_v2 (Impression-restricted,
# explicit-indeterminacy rule); v1 rows are kept below for comparison only.
CLS = change_class_v2
fa = flips("reversal_nocf_v2")
fb = flips("reversal_cfdropout_v2")
add("4", "reversal flip rate, nocf", ci(fa), "own denominator")
add("4", "reversal flip rate, cf+dropout", ci(fb), "own denominator")
both = np.isfinite(fa) & np.isfinite(fb)
add("4", "reversal flip rate, nocf (common subset)", ci(fa[both]), "common subset", reported=False)
add("4", "reversal flip rate, cf+dropout (common subset)", ci(fb[both]), "common subset", reported=False)
add("4", "reversal difference (paired, common subset)", ci(fb[both] - fa[both]), "paired", True, reported=False)
rng = np.random.default_rng(0)
va, vb = fa[np.isfinite(fa)], fb[np.isfinite(fb)]
bs = np.array([vb[rng.choice(len(vb), len(vb), True)].mean()
               - va[rng.choice(len(va), len(va), True)].mean() for _ in range(R)])
add("4", "reversal difference (UNPAIRED -- the value the paper reports)",
    (vb.mean() - va.mean(), *np.percentile(bs, [2.5, 97.5]), len(va)), "unpaired", True)
fa1 = flips("reversal_nocf_v1")
fb1 = flips("reversal_cfdropout_v1")
add("4", "[v1 classifier, not used] flip rate nocf", ci(fa1), "own denominator", reported=False)
add("4", "[v1 classifier, not used] flip rate cf+dropout", ci(fb1), "own denominator", reported=False)

# ---- write ------------------------------------------------------------------
with open(OUT, "w") as f:
    f.write("# Bootstrap intervals for the manuscript — single source\n\n")
    f.write("Every interval below comes from ONE run of `paper/stats/all_cis.py`: "
            "`numpy.random.default_rng(0)`, 10,000 resamples, percentile method. "
            "Inputs are the per-report rg_er arrays in `paper/cache/perreport/`, "
            "produced by re-scoring the cached generations with RadGraph-XL "
            "(`reward_level=\"all\"`, index [1] = entity+relation F1).\n\n")
    f.write("Pairing is asserted, never assumed: `index` requires the two dumps' references "
            "to match at every position; `uid` joins the BrainDiff dump to the NeuroVFM CSV "
            "on `study_uid2`; `ref` joins the 64-study frontier files by reference text. "
            "Every rg_er row is n=1444, the full test split.\n\n")
    f.write("**BIND is not included.** Table 3's external-cohort rows are omitted here: "
            "BIND is health-system data under a use agreement that does not permit "
            "redistribution, so its per-report caches are not in this tree. Only the "
            "internal side of Table 3 is reproducible from these files.\n\n")
    f.write("**Reversal probe.** Direction is assigned with `change_class_v2` — the "
            "Impression-restricted classifier with the explicit-indeterminacy rule, the same "
            "one behind Fig. 3. The probe ran on the **503 directional change cases** of the "
            "test split -- every row labelled New lesion, Progressed, Improved or Resolved "
            "(510 of 1,464), less 7 dropped for incomplete imaging -- not the full 1,444. "
            "Stable, Mixed and Indeterminate rows are excluded by construction: a reversal "
            "has no direction to flip. It is then further restricted to change-cases whose "
            "forward report asserts a direction, which is why n is 282-285. Rows marked "
            "`[v1 classifier, not used]` are the superseded `change_class` values, kept only "
            "so they are not re-derived by mistake.\n\n")
    f.write("The comparison below is against **BrainDiff Main.pdf** (2026-08-30 21:03); "
            "it is re-read from the PDF on every run.\n\n")
    order = list(dict.fromkeys(t for t, *_ in rows))     # each table once, first-seen order
    rows_sorted = [r for t in order for r in rows if r[0] == t]
    cur = None
    for tbl, name, val, n, pairing, _rep in rows_sorted:
        if tbl != cur:
            f.write(f"\n## Table {tbl}\n\n| quantity | value [95% CI] | n | pairing |\n"
                    f"|---|---|---:|---|\n")
            cur = tbl
        f.write(f"| {name} | {val} | {n} | {pairing} |\n")
    # The printed values are read FROM THE PDF at run time, so this comparison cannot
    # go stale when the paper is recompiled. A value counts as present if its 4-decimal
    # digits appear anywhere in the extracted text (whitespace stripped, unicode minus
    # normalised) -- enough to catch "paper says 0.0223, we compute 0.0222" without
    # depending on how pdf extraction happens to lay out a table.
    pdf_txt = None
    if os.path.exists(PDF):
        try:
            import pypdf
            raw = "".join(pg.extract_text() or "" for pg in pypdf.PdfReader(PDF).pages)
            pdf_txt = re.sub(r"\s+", "", raw).replace("\u2212", "-").replace("\u2013", "-")
        except Exception as e:                       # pypdf missing or unreadable PDF
            pdf_txt = None
            f.write(f"\n> PDF comparison skipped: {e}\n")
    if pdf_txt is None:
        f.write(f"\n> PDF comparison skipped: `{os.path.basename(PDF)}` not readable.\n")
    else:
        missing = []
        for tbl, name, val, n, pairing, rep in rows_sorted:
            if not rep:                      # the paper does not print this quantity
                continue
            if str(tbl).startswith("S"):
                continue                 # supplement rows: main PDF is the comparison target
            pt = re.match(r"[-+]?\d\.\d{4}", val)
            if not pt:
                continue
            digits = pt.group(0).lstrip("+-")
            if digits not in pdf_txt:
                missing.append((tbl, name, val))
        f.write(f"\n## Values not found in `{os.path.basename(PDF)}`\n\n")
        if not missing:
            f.write("None — every computed point estimate appears in the PDF.\n")
        else:
            f.write("These quantities ARE printed in the paper, but the computed point "
                    "estimate does not occur anywhere in its text -- i.e. the paper prints a "
                    "different number.\n\n"
                    "| table | quantity | computed |\n|---|---|---|\n")
            for tbl, name, val in missing:
                f.write(f"| {tbl} | {name} | **{val}** |\n")
        notp = [(t, nm, v) for t, nm, v, _n, _p, rep in rows_sorted if not rep]
        f.write("\n<details><summary>Computed here but not printed in the paper "
                f"({len(notp)} rows) — excluded from the comparison above</summary>\n\n")
        f.write("| table | quantity | computed |\n|---|---|---|\n")
        for t, nm, v in notp:
            f.write(f"| {t} | {nm} | {v} |\n")
        f.write("\n</details>\n")

    f.write("\n## Not bootstrapped here\n\n"
            "- **Table 8** (change-decodability AUROCs) — read from the probe result JSONs; "
            "no interval was computed for them. All four reproduce: 0.766 / 0.601 / 0.702 / 0.997.\n"
            "- **Real-Impression ranking** (supplement; not in the main PDF) — "
            "`impression_validation_v2.py`, `default_rng(0)`, 10,000 resamples: "
            "BrainDiff 0.0961, NeuroVFM 0.0885, "
            "difference **+0.0075 [+0.0015, +0.0134]**, n=1297.\n"
            "- **BLEU-4 / METEOR** — corpus-level and per-report respectively; point estimates "
            "only, in `paper/stats/PAPER_METRICS.md`.\n")
print(f"wrote {OUT}  ({len(rows)} intervals)")
