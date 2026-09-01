"""Single source for every rg_er / BLEU-4 / METEOR POINT ESTIMATE in the paper.

The companion to all_cis.py, which owns the confidence intervals. This owns the three
report-quality metrics themselves, table by table, and the paired differences between arms.

Everything is read from the cached per-report arrays in paper/cache/perreport/, keyed
by `study_uid2`. So:

  * rg_er     per-report array from the cache (RadGraph-XL, reward_level="all", index [1]).
  * BLEU-4    cached corpus-level value, method-1 smoothing.
  * METEOR    cached per-report average.

BLEU-4 and METEOR were previously recomputed here from the cached generations. The MR-RATE
report text is not redistributable and is not in this tree, so both are now stored in the
cache as the scalars they were computed to. They are deterministic functions of the text, so
the stored values are exactly what the recomputation produced. Each row names its dump.

  conda activate BrainDiff && python paper/stats/all_metrics.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "cache", "perreport")
OUT = os.path.join(HERE, "PAPER_METRICS.md")
_cache = {}


def load(tag):
    if tag not in _cache:
        d = json.load(open(f"{P}/{tag}.json"))
        _cache[tag] = d
    return _cache[tag]


def metrics(tag):
    """(rg_er, bleu4, meteor, n, source) for one arm."""
    d = load(tag)
    rg = np.array(d["rg_er"], float)
    return (float(np.nanmean(rg)), d["bleu4"], d["meteor"],
            d["n"], d.get("source", "--"))


def paired(a, b):
    """Difference of the two arms' metrics, index-paired with the reference assert.

    rg_er is differenced at full precision (per-report mean difference). BLEU-4 and METEOR
    are differenced on the values ROUNDED TO 4 DECIMALS -- the convention the manuscript's
    tables use, since a reader subtracting two printed cells must get the printed difference.
    The two conventions disagree in the last digit where the unrounded difference falls near
    a rounding boundary (Table 7 image contribution: 0.02134 -> 0.0213 unrounded, but
    0.2281 - 0.2067 = 0.0214 on the printed cells).
    """
    da, db = load(a), load(b)
    assert da["uids"] == db["uids"], f"{a} vs {b}: not index-aligned; refusing to difference"
    ma, mb = metrics(a), metrics(b)
    return (ma[0] - mb[0],
            round(ma[1], 4) - round(mb[1], 4),
            round(ma[2], 4) - round(mb[2], 4),
            ma[3], "index")


ROWS = []


def add(table, name, vals, kind):
    ROWS.append((table, name, vals, kind))


# ---- Table 2 : baselines ----------------------------------------------------
add("2", "BrainDiff, full test", metrics("C_braindiff_real"), "arm")
add("2", "NeuroVFM + report pipeline, full test", metrics("neurovfm_base_fulltest"), "arm")
add("2", "Text-only (images zeroed), full test", metrics("C_braindiff_scanzero"), "arm")
add("2", "Opus 5, 64-subset", metrics("claude_opus5_subset64"), "arm")
add("2", "GPT-5.6 Sol, 64-subset", metrics("gpt56sol_subset64"), "arm")
add("2", "BrainDiff, 64-subset", metrics("braindiff_subset64"), "arm")
add("2", "NeuroVFM, 64-subset", metrics("neurovfm_subset64"), "arm")

# ---- Table 3 : internal side only -------------------------------------------
# BIND's rows are withheld -- external health-system data we cannot redistribute,
# so its per-report caches are not in this tree. See PAPER_CIS.md for the note.
add("3", "internal (full test)", metrics("C_braindiff_real"), "arm")
add("3", "internal image contribution", paired("C_braindiff_real", "C_braindiff_visroll"), "diff")

# ---- Table 4 : counterfactual + dropout -------------------------------------
add("4", "nocf, own scans", metrics("C_nocf_real"), "arm")
add("4", "nocf, other patient's scans", metrics("C_nocf_visroll"), "arm")
add("4", "cf+dropout, own scans", metrics("C_braindiff_real"), "arm")
add("4", "cf+dropout, other patient's scans", metrics("C_braindiff_visroll"), "arm")
add("4", "nocf: image effect", paired("C_nocf_real", "C_nocf_visroll"), "diff")
add("4", "cf+dropout: image effect", paired("C_braindiff_real", "C_braindiff_visroll"), "diff")
add("4", "quality cost (cf+dropout - nocf)", paired("C_braindiff_real", "C_nocf_real"), "diff")

# ---- Table 5 : curriculum, image effect -------------------------------------
add("5", "none: real / wrong scans", metrics("ncnf_real"), "arm")
add("5", "no curriculum: real", metrics("C_nocurric_real"), "arm")
add("5", "S1 only: real", metrics("C_ns2_real"), "arm")
add("5", "none (no curriculum, no cf, no dropout)", paired("ncnf_real", "ncnf_visroll"), "diff")
add("5", "no curriculum (cf + dropout)", paired("C_nocurric_real", "C_nocurric_visroll"), "diff")
add("5", "S1 only", paired("C_ns2_real", "C_ns2_visroll"), "diff")
add("5", "S1 + S2", paired("C_braindiff_real", "C_braindiff_visroll"), "diff")

# ---- Table 6 : prior report x image factorial -------------------------------
add("6", "report present, own scans", metrics("C_braindiff_real"), "arm")
add("6", "report present, other patient's scans", metrics("C_braindiff_visroll"), "arm")
add("6", "report withheld, own scans", metrics("C_braindiff_noreport"), "arm")
add("6", "report withheld, other patient's scans", metrics("C_braindiff_nr_visroll"), "arm")
add("6", "image effect, report present", paired("C_braindiff_real", "C_braindiff_visroll"), "diff")
add("6", "image effect, report withheld", paired("C_braindiff_noreport", "C_braindiff_nr_visroll"), "diff")
add("6", "effect of prior report, own scans", paired("C_braindiff_real", "C_braindiff_noreport"), "diff")

# ---- Table 7 : difference pathway -------------------------------------------
add("7", "delta on", metrics("C_deltaon_real"), "arm")
add("7", "delta off", metrics("C_braindiff_real"), "arm")
add("7", "wrong patient's scans (delta off)", metrics("C_braindiff_visroll"), "arm")
add("7", "images zeroed (delta off)", metrics("C_braindiff_scanzero"), "arm")
add("7", "delta contribution", paired("C_deltaon_real", "C_braindiff_real"), "diff")
add("7", "image contribution vs wrong scans", paired("C_braindiff_real", "C_braindiff_visroll"), "diff")

# ---- Table S1 : curriculum arms, prior-report reliance ----------------------
add("S1", "nodelta (S1+S2), report available", metrics("C_braindiff_real"), "arm")
add("S1", "nostage2 (S1), report available", metrics("C_ns2_real"), "arm")
add("S1", "nocurriculum (none), report available", metrics("C_nocurric_real"), "arm")
add("S1", "nodelta (S1+S2), report withheld", metrics("C_braindiff_noreport"), "arm")
add("S1", "nostage2 (S1), report withheld", metrics("C_ns2_noreport"), "arm")
add("S1", "nocurriculum (none), report withheld", metrics("C_nocurric_noreport"), "arm")
add("S1", "report-reliance, nodelta", paired("C_braindiff_real", "C_braindiff_noreport"), "diff")
add("S1", "report-reliance, nostage2", paired("C_ns2_real", "C_ns2_noreport"), "diff")
add("S1", "report-reliance, nocurriculum", paired("C_nocurric_real", "C_nocurric_noreport"), "diff")


def fmt(v, kind):
    s = "+" if kind == "diff" else ""
    return f"{v:{s}.4f}"


def main():
    with open(OUT, "w") as f:
        f.write("# rg_er / BLEU-4 / METEOR for the manuscript -- single source\n\n")
        f.write("Every value below comes from ONE run of `paper/stats/all_metrics.py`, "
                "reading the cached per-report arrays in `paper/cache/perreport/`.\n\n"
                "`rg_er` is read from the cache (RadGraph-XL, `reward_level=\"all\"`, index [1] "
                "= entity+relation F1). `BLEU-4` is corpus-level with method-1 smoothing and "
                "`METEOR` is per-report averaged; both are cached scalars, computed from the "
                "generations before the MR-RATE text was removed from this tree. No GPU, no "
                "API calls, no regeneration.\n\n"
                "Rows marked *diff* are the paired difference between two arms, index-paired "
                "with the study uids asserted equal at every position. rg_er is differenced "
                "at full precision; BLEU-4 and METEOR are differenced on the 4-decimal values, "
                "matching the manuscript's tables so that subtracting two printed cells gives "
                "the printed difference. BLEU-4 is corpus-level, so its differences are not "
                "per-report paired quantities and carry no interval here.\n\n"
                "**BIND is not included.** Table 3's external-cohort rows are omitted: BIND is "
                "health-system data under a use agreement that does not permit "
                "redistribution, so its per-report caches are not in this tree. Only the "
                "internal side of Table 3 is reproducible from these files.\n\n"
                "Confidence intervals live in `PAPER_CIS.md`, which owns rg_er intervals only. "
                "The two files share the same caches, so rg_er agrees between them by "
                "construction.\n\n")
        order, seen = [], set()
        for t, *_ in ROWS:
            if t not in seen:
                seen.add(t); order.append(t)
        for t in order:
            f.write(f"\n## Table {t}\n\n")
            f.write("| quantity | rg_er | BLEU-4 | METEOR | n | source |\n")
            f.write("|---|---:|---:|---:|---:|---|\n")
            for tt, name, (rg, bl, me, n, src), kind in ROWS:
                if tt != t:
                    continue
                s = f"`{src}`" if src != "index" else "index-paired"
                f.write(f"| {name} | {fmt(rg,kind)} | {fmt(bl,kind)} | {fmt(me,kind)} "
                        f"| {n} | {s} |\n")
        f.write("\n## Not covered here\n\n")
        f.write("- **Reversal probe** (Table 4's right-hand column) is a signed-flip rate, "
                "not a report-quality metric; it lives in `PAPER_CIS.md`.\n")
        f.write("- **Change-decodability AUROCs** (Table 8) are probe outputs, not "
                "generations.\n")
        f.write("- **Metric floors** are reference-vs-reference scores and carry no BLEU or "
                "METEOR.\n")
    print(f"wrote {OUT}  ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
