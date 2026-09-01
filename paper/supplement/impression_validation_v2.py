"""Real-text Impression validation, v2. Scores each system's generated Impression against the
real radiologist Impression from report2, held out from every model.

Three changes from impression_validation.py, all of which the v1 numbers depend on:

1. Production checkpoint, not the delta-ON ablation. v1 read logs/rger_AB/A (change-map
   ep5), the arm that fills one row of Table 4 and whose own CI straddles zero, while every
   other result uses nodelta_10 at logs/recompute/pp. That arm also has 10x the
   repetition-loop rate (61 vs 6 reports with no Impression at all).
2. No study is dropped for a BrainDiff failure. v1 intersected on all three texts having an
   Impression, removing studies where BrainDiff looped and was truncated before writing the
   section -- excluding its own failures from its own mean while NeuroVFM contributed all
   1,444. The universe here is fixed by the reference alone, with a last-complete-sentence
   fallback so a generation failure scores badly instead of vanishing.
3. Bounded, longest-candidate extraction on the generated side. A looped report re-opens
   "Impression:" several times and the final one is a truncated fragment, so v1's
   last-header rule returned fragments on 57 BrainDiff reports. Reference extraction is
   deliberately unchanged (last header to end of text): real reports are concatenated
   multi-study documents where the last Impression is the operative one.

GPU RadGraph -- run in radgraph_env.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
_TEST_CSV_DEFAULT = _os.path.join(_REPO, "paper", "cache", "s4_test.csv")
if not _os.path.exists(_TEST_CSV_DEFAULT):
    _TEST_CSV_DEFAULT = "/home/data/BRAIN_DIFF_S4/splits_extended/s4_test.csv"
import csv, json, os, re, sys
import numpy as np
csv.field_size_limit(10 ** 7)
from radgraph import F1RadGraph

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_CSV = _TEST_CSV_DEFAULT
BD_SHARDS = [f"{REPO}/logs/recompute/pp/rger_real.rank{k}.json" for k in (0, 1)]
SIDECAR = f"{REPO}/logs/rger_AB/A/rger_real.uids.json"
NV_CSV = f"{REPO}/paper/cache/outputs/neurovfm_base/fulltest_longitudinal_meta_reports.csv"

_SECT = (r"(?:findings|lesions|structural\s+effects|background\s+findings|impression|"
         r"conclusion|assessment|sonu[çc]|technique|comparison|history|indication|"
         r"clinical\s+information|protocol)")
IMP_HDR = re.compile(r"\b(impression|conclusion|assessment|sonu[çc])\b\s*[:\-]", re.I)
NEXT_HDR = re.compile(r"\b" + _SECT + r"\b\s*[:\-]", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")


def ref_impression(text):
    """v1's rule, unchanged -- the reference must stay what the paper validated."""
    text = text or ""
    ms = list(IMP_HDR.finditer(text))
    if not ms:
        return ""
    return re.sub(r"^\s*[:\-]\s*", "", text[ms[-1].end():].strip()).strip()


def last_sentence(text):
    t = (text or "").strip()
    parts = [p.strip() for p in _SENT.split(t) if p.strip()] if t else []
    if parts and not re.search(r"[.!?]\s*$", parts[-1]):
        parts = parts[:-1]                      # drop the truncation fragment
    for p in reversed(parts):
        if len(p.split()) >= 3:
            return p
    return ""


def hyp_impression(text):
    """Generated side: bounded segments, longest candidate, last-sentence fallback."""
    t = (text or "").strip()
    if not t:
        return ""
    cands = []
    for m in IMP_HDR.finditer(t):
        nxt = NEXT_HDR.search(t, m.end())
        seg = re.sub(r"^\s*[:\-]\s*", "",
                     t[m.end(): nxt.start() if nxt else len(t)]).strip()
        if seg and len(seg.split()) >= 3:
            cands.append(seg)
    return max(cands, key=len) if cands else last_sentence(t)


def main():
    rows = list(csv.DictReader(open(TEST_CSV)))
    real = {r["study_uid2"]: ref_impression(r.get("report2", "")) for r in rows}
    real = {u: v for u, v in real.items() if v}

    hyps = []
    for p in BD_SHARDS:
        hyps += json.load(open(p))["hyps"]
    side = json.load(open(SIDECAR))
    refs_on_disk = []
    for p in BD_SHARDS:
        refs_on_disk += json.load(open(p))["refs"]
    assert side["refs"] == refs_on_disk, "sidecar order does not match the dump"
    bd = {u: hyp_impression(h) for h, u in zip(hyps, side["uids"])}

    nv = {r["study_uid2"]: hyp_impression(r.get("generated_report", ""))
          for r in csv.DictReader(open(NV_CSV))}

    # Universe fixed by the REFERENCE only. A system that failed to generate is
    # kept and scores 0 -- it is not allowed to disappear.
    uids = sorted(u for u in real if u in bd and u in nv)
    bd_empty = sum(1 for u in uids if not bd[u])
    nv_empty = sum(1 for u in uids if not nv[u])
    print(f"n (real report2 has an Impression, both systems present): {len(uids)}")
    print(f"  BrainDiff empty after fallback: {bd_empty}   NeuroVFM empty: {nv_empty}")

    f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl",
                    cuda=int(os.environ.get("RG_CUDA", "0")))

    def per(H, R):
        keep = [i for i, (h, r) in enumerate(zip(H, R)) if h.strip() and r.strip()]
        a = np.zeros(len(H))                    # empty hypothesis -> 0.0, not dropped
        if keep:
            _, rl, *_ = f1(hyps=[H[i] for i in keep], refs=[R[i] for i in keep])
            for j, i in enumerate(keep):
                a[i] = rl[1][j]
        return a

    R = [real[u] for u in uids]
    rb = per([bd[u] for u in uids], R)
    rn = per([nv[u] for u in uids], R)
    d = rb - rn
    rng = np.random.default_rng(0)
    ix = np.arange(len(d))
    s = np.array([d[rng.choice(ix, len(ix), True)].mean() for _ in range(10000)])
    lo, hi = np.percentile(s, [2.5, 97.5])
    line = ("IMPRESSION-vs-REAL rg_er (v2, production nodelta_10, no BrainDiff-based "
            "exclusion): BrainDiff %.4f  NeuroVFM %.4f  BrainDiff-NeuroVFM %+.4f "
            "CI[%+.4f,%+.4f]  n=%d" % (rb.mean(), rn.mean(), d.mean(), lo, hi, len(d)))
    print(line)
    with open(f"{REPO}/benchmarks/mrrate_proprietary/impression_validation_v2_result.txt", "w") as fh:
        fh.write(line + "\n")
    json.dump({"uids": uids, "bd": rb.tolist(), "nv": rn.tolist()},
              open(f"{REPO}/benchmarks/mrrate_proprietary/impression_validation_v2_perreport.json", "w"))
    print("IMPRESSION_VALIDATION_V2_DONE")


if __name__ == "__main__":
    main()
