#!/usr/bin/env python3
"""
Stage-1 tables. CPU except `label` (one GPU per shard).

    python build_stage1.py dupes                            (env BrainDiff)      -> stage1/dupes_brats_met_yale{.json,_volumes.csv}
    python build_stage1.py probe_pairs                      (env BrainDiff)      -> probes/pairs_t1ce.csv (+ phase0_pairs.csv)
    python build_stage1.py coverage                         (env bind_proc)      -> stage1/coverage.csv
    CUDA_VISIBLE_DEVICES=0 python build_stage1.py label --src bind --shard 0 --nshard 6   (scoring env)
    python build_stage1.py combine --src bind               (s4gen and bind)     -> findings/{src}_{units,reports}.parquet
    STAGE1_BIND_SNAPSHOT=$BRAINDIFF_BUILD/stage1/bind_ready_2026-10-22T195800.csv python build_stage1.py pairs
                                                            (env BrainDiff)      -> stage1/pairs.parquet, triplets.parquet, ...
    python build_stage1.py entity_text                                           -> stage1/entity_text.parquet (status targets)
    python build_stage1.py tasks                                                 -> stage1/tasks.parquet (single-study tasks)

Rows of pairs.parquet are (prior, current) in time order. The BIND split used in training comes from bind_qc_split.py,
not the `split` column. Without STAGE1_BIND_SNAPSHOT, `pairs` re-scans $BIND_OUT and writes a new snapshot; the
trained tables use bind_ready_2026-10-22T195800.csv.

Env: BRAINDIFF_DATA (BRAIN_DIFF_S1, BRAIN_DIFF_S3, BRAIN_DIFF_S4, MR-RATE), BRAINDIFF_BUILD (output root),
BIND_* (bind_preprocess.py), RADDELTA_CKPT (Rad-Δ student checkpoint).
"""
import argparse
import json
import os
import re
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bind_preprocess import (MODALITIES, OUTPUT_ROOT, PAIRED_CSV, TABLES, findings_onward, is_mr_brain,
                             canonical_patient_series, canonical_session_series, load_dates, load_patient_merge_map,
                             load_reports, session_dir, split_sections, strip_comparison, BIND_ROOT)

DATA = Path(os.environ.get("BRAINDIFF_DATA", "data"))
BUILD = Path(os.environ.get("BRAINDIFF_BUILD", "build"))
S1, S3, S4, MRRATE = DATA / "BRAIN_DIFF_S1", DATA / "BRAIN_DIFF_S3", DATA / "BRAIN_DIFF_S4", DATA / "MR-RATE"
OUT, F = BUILD / "stage1", BUILD / "findings"
COVERAGE = OUT / "coverage.csv"
MODS = ["T1w", "T1ce", "T2w", "FLAIR"]
CHANGE = ["NEW", "WORSENED", "IMPROVED", "STABLE", "RESOLVED"]
REVERSE = [4, 2, 1, 3, 0]          # L_ent reversed pair: NEW<->RESOLVED, WORSENED<->IMPROVED, STABLE fixed
MIN_LENT_WEIGHT = 0.5
MONITOR_FRAC, SEED = 0.05, 0
STATS = {}


# ================================================================== coverage
# Per BIND report pair: change-language class, whether / with what modality the COMPARISON section names a prior, and
# sessions between the pair dates. An intervening exam or a CT-only comparison drops the BIND L_ent label in `pairs`.
NON_CLINICAL = {"HISTORY", "CLINICAL HISTORY", "ADDITIONAL HISTORY", "INDICATION", "INDICATIONS", "CLINICAL INDICATION",
                "TECHNIQUE", "PROCEDURE COMMENTS", "ACCESSION NUMBER", "ATTESTATION", "NARRATIVE", "EXAM"}
# direction-free references to a prior exam ("unchanged", "new" are scored by the lexicon)
REFERENCE = re.compile(
    r"\bcompared (?:to|with)\b|\bin comparison\b|\binterval(?:ly)?\b|\bagain (?:seen|noted|"
    r"demonstrated|identified|visualized|present)\b|\bsince\b|\b(?:prior|previous) "
    r"(?:exam|study|studies|mri|mr|ct|imaging|examination|scan)\b|\bno (?:significant )?change\b", re.I)
# signal descriptors that reuse change words ("increased T2 signal"); masked in the strict variant only
DESCRIPTOR = re.compile(
    r"\b(increased|decreased|reduced|increase|decrease)\s+(?:\w+\s+){0,2}?"
    r"(signal|intensity|diffusion|diffusivity|susceptibility|perfusion|blood flow|cbf|cbv|"
    r"t2|flair|t1|enhancement|attenuation|uptake)\b", re.I)
ATTEST = re.compile(r"teaching physician|reviewed the images|attestation|electronically signed", re.I)
BLOCK_GAP = re.compile(r"\n[ \t\xa0]*\n[ \t\xa0]*\n")
MASKED_DATE = re.compile(r"\*{5}[/.-]\*{5}(?:[/.-]\*{5})?|\*{5} \*{5},? \*{5}")
NUM_DATE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{4}\b", re.I)
YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
NONE_RE = re.compile(r"^\W*(none|no prior|no previous|no comparison|not available|n/?a\b|no relevant|"
                     r"no prior imaging|no outside|unavailable)", re.I)
NONE_ANY = re.compile(r"no prior (?:imaging|studies|study|exam)|none available|no comparison", re.I)
MR_RE = re.compile(r"\bMR\b|\bMRI|\bMRA\b|\bMRs\b|magnetic resonance", re.I)
CT_RE = re.compile(r"\bCT\b|\bCTA\b|\bCTs\b|computed tomograph|\bhead CT", re.I)
OTHER_RE = re.compile(r"\bPET\b|ultrasound|angiogra|x-?ray|radiograph", re.I)
SAME_DAY = re.compile(r"same[- ]day|earlier today|same date|earlier the same|today", re.I)


def polarities(text, strict=False):
    from braindiff.eval import temporal_score as ts
    text = (text or "").lower()
    if strict:
        text = DESCRIPTOR.sub(lambda m: "x " + m.group(0)[len(m.group(1)):], text)
    found = set()
    for clause in ts.CLAUSE_SPLIT.split(text):
        found |= ts._polarity(clause)
    return found


def clinical_body(report):
    return "\n".join(b for h, b in split_sections(report) if h not in NON_CLINICAL)


def conclusion(report, site):
    """IMPRESSION/SUMMARY body; for I0001 (no header) the last non-attestation block after a double blank line."""
    sections = split_sections(report)
    imp = [b for h, b in sections if h in {"IMPRESSION", "SYNOPSIS FOR CLINICAL MANAGEMENT"}]
    if imp:
        return imp[-1], "header"
    if site != "I0001":
        return "", "none"
    body = "\n".join(b for h, b in sections if h not in NON_CLINICAL - {"ATTESTATION"})
    blocks = [b.strip() for b in BLOCK_GAP.split(body) if b.strip()]
    blocks = [b for b in blocks if not ATTEST.search(b[:200])]
    return (blocks[-1], "gap") if len(blocks) >= 2 else ("", "none")


def cue_class(text, strict=False):
    from braindiff.eval import temporal_score as ts
    p = polarities(text, strict)
    ref = bool(REFERENCE.search(text or ""))
    return {"any": bool(p) or ref, "stable": ts.STABLE in p, "prog": ts.PROGRESSED in p, "impr": ts.IMPROVED in p, "ref": ref}


def category(c):
    if c["prog"] and c["impr"]:
        return "mixed"
    if c["prog"]:
        return "progression"
    if c["impr"]:
        return "improvement"
    if c["stable"]:
        return "stability_only"
    if c["ref"]:
        return "reference_only"
    return "none"


def parse_comparison(raw):
    comp = [b for h, b in split_sections(raw) if h in {"COMPARISON", "COMPARISON STUDIES"}]
    if not comp:
        return dict(comp_section=False)
    text = " ".join(comp)[:600]
    n_masked, n_num = len(MASKED_DATE.findall(text)), len(NUM_DATE.findall(text))
    mr, ct, other = bool(MR_RE.search(text)), bool(CT_RE.search(text)), bool(OTHER_RE.search(text))
    first = min(((m.start(), k) for k, rx in (("MR", MR_RE), ("CT", CT_RE)) if (m := rx.search(text))), default=(None, ""))[1]
    is_none = bool(NONE_RE.match(text)) or (bool(NONE_ANY.search(text)) and n_masked + n_num == 0)
    return dict(comp_section=True, comp_none=is_none, comp_n_masked_dates=n_masked, comp_n_unmasked_dates=n_num,
                comp_years=",".join(YEAR.findall(text)), comp_mr=mr, comp_ct=ct, comp_other=other,
                comp_first_modality=first, comp_same_day=bool(SAME_DAY.search(text)),
                comp_text=text[:200].replace("\n", " "))


def site_raw(site, pairs):
    """Raw report of each study_uid2, matched on exact stripped text."""
    reports = load_reports(site)
    reports = reports[reports["Type"].fillna("").str.upper().str.strip().map(is_mr_brain)]
    reports = reports[reports["session_uid"].isin(set(pairs["study_uid2"]))]
    reports["stripped"] = reports["Report_txt"].map(lambda t: strip_comparison(t)[0])
    want = pairs[["study_uid2", "report2"]].drop_duplicates()
    merged = want.merge(reports[["session_uid", "Report_txt", "stripped"]], left_on="study_uid2", right_on="session_uid", how="left")
    exact = merged[merged["stripped"] == merged["report2"]].drop_duplicates("study_uid2")
    fallback = merged.drop_duplicates("study_uid2")
    raw = dict(zip(fallback["study_uid2"], fallback["Report_txt"]))
    raw.update(dict(zip(exact["study_uid2"], exact["Report_txt"])))
    print(f"  {site}: raw text matched exactly for {len(exact):,}/{len(want):,} current sessions")
    return raw


def coverage_sessions(site):
    """Every dated session of every patient with its Type (incl. 'not available')."""
    demo = pd.read_csv(f"{BIND_ROOT}/Imaging/{site}/Clinical/{site}_demographics.csv", dtype=str,
                       usecols=["bdsp_patient_id", "Session_id"])
    reports = pd.read_csv(f"{BIND_ROOT}/Imaging/{site}/Clinical/{site}_reports.csv", dtype=str, usecols=["session_id", "Type"])
    demo["session_uid"] = canonical_session_series(demo["Session_id"], site)
    reports["session_uid"] = canonical_session_series(reports["session_id"], site)
    demo["patient_uid"] = canonical_patient_series(demo["bdsp_patient_id"], load_patient_merge_map(site))
    ses = demo.drop_duplicates("session_uid").merge(load_dates(site), on="session_uid", how="left")
    types = reports.groupby("session_uid")["Type"].agg(lambda s: " | ".join(sorted(set(s.fillna("")))))
    ses["Type"] = ses["session_uid"].map(types).fillna("")
    ses["kind"] = ses["Type"].str.lower().map(
        lambda t: "mr_brain" if any(is_mr_brain(x) for x in t.upper().split(" | "))
        else "not_available" if "not available" in t else "other")
    return ses


def intervening(pairs, ses):
    """Sessions strictly between the pair dates, by kind. Undated sessions are unplaceable."""
    d = dict(zip(ses["session_uid"], ses["study_date"]))
    by_pat = {p: g for p, g in ses.dropna(subset=["study_date"]).groupby("patient_uid")}
    undated = ses[ses["study_date"].isna()].groupby("patient_uid").size()
    rows = []
    for p, s1, s2 in zip(pairs["patient_uid"], pairs["study_uid1"], pairs["study_uid2"]):
        g = by_pat.get(p)
        d1, d2 = d.get(s1), d.get(s2)
        if g is None or pd.isna(d1) or pd.isna(d2):
            rows.append((None, None, None, None, int(undated.get(p, 0))))
            continue
        mid = g[(g["study_date"] > d1) & (g["study_date"] < d2)]
        same2 = g[(g["study_date"] == d2) & (g["session_uid"] != s2)]
        k = mid["kind"].value_counts()
        rows.append((int(k.get("mr_brain", 0)), int(k.get("not_available", 0)), int(k.get("other", 0)), len(same2),
                     int(undated.get(p, 0))))
    return pd.DataFrame(rows, columns=["between_mr_brain", "between_not_available", "between_other",
                                       "same_day_as_current", "patient_undated_sessions"], index=pairs.index)


def coverage(a):
    pairs = pd.read_csv(PAIRED_CSV, dtype=str)
    pairs["duration"] = pairs["duration"].astype(int)
    print(f"{len(pairs):,} pairs")
    out = pairs[["patient_uid", "site", "study_uid1", "study_uid2", "duration", "had_comparison1", "had_comparison2"]].copy()
    out["dur_bucket"] = pd.cut(out["duration"], [0, 29, 180, 365, 10**6], labels=["<30", "30-180", "180-365", ">365"])
    for strict in (False, True):
        tag = "_strict" if strict else ""
        out[f"body_cat{tag}"] = [category(cue_class(clinical_body(r), strict)) for r in pairs["report2"]]
    out["whole_text_cat"] = [category(cue_class(r)) for r in pairs["report2"]]
    concl = [conclusion(r, s) for r, s in zip(pairs["report2"], pairs["site"])]
    out["concl_source"] = [src for _, src in concl]
    out["concl_cat"] = [category(cue_class(t)) if t else "no_conclusion" for t, _ in concl]
    out["concl_cat_strict"] = [category(cue_class(t, True)) if t else "no_conclusion" for t, _ in concl]
    raws, comp_rows, inter = {}, [], []
    for site, g in pairs.groupby("site"):
        raws.update(site_raw(site, g))
        inter.append(intervening(g, coverage_sessions(site)))
        dates = load_dates(site)
        raws.update({f"date::{k}": v for k, v in zip(dates["session_uid"], dates["study_date"])})
    for s1, s2 in zip(pairs["study_uid1"], pairs["study_uid2"]):
        row = parse_comparison(raws.get(s2) or "")
        d1 = raws.get(f"date::{s1}")
        row["study1_year"] = d1.year if d1 is not None else None
        row["year_match_study1"] = (str(d1.year) in row.get("comp_years", "").split(",")
                                    if d1 is not None and row.get("comp_years") else None)
        comp_rows.append(row)
    out = pd.concat([out, pd.DataFrame(comp_rows, index=pairs.index), pd.concat(inter).loc[pairs.index]], axis=1)

    def status(r):
        if not r["comp_section"]:
            return "no_section"
        if r["comp_none"]:
            return "none_or_no_prior"
        if r["comp_n_masked_dates"] + r["comp_n_unmasked_dates"] == 0:
            return "no_date_given"
        return "dates_all_masked" if r["comp_n_unmasked_dates"] == 0 else "unmasked_date"
    out["comp_status"] = out.apply(status, axis=1)

    def modality(r):
        if r["comp_status"] in ("no_section", "none_or_no_prior"):
            return ""
        if r["comp_mr"] and r["comp_ct"]:
            return "MR+CT"
        if r["comp_ct"]:
            return "CT_only"
        if r["comp_mr"]:
            return "MR_only"
        return "other_only" if r["comp_other"] else "unstated"
    out["comp_modality"] = out.apply(modality, axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(COVERAGE, index=False)
    print(f"saved {COVERAGE}")


# ================================================================== label / combine
# Rad-Δ per-unit change labels = the L_ent / status targets. Text: s4gen = generated_report; bind = findings_onward(report2)
# (raw report2 without a FINDINGS header). Reports > MAX_WORDS are cut at sentence boundaries (longer ones OOM in the
# span-pair relation layer). Stored s4gen / bind units predate the `presence` field.
SRC = {"s4gen": lambda: S4 / "main_extended.csv", "bind": lambda: PAIRED_CSV}
MAX_WORDS = 600


def load(src):
    df = pd.read_csv(SRC[src]())
    if src == "s4gen":
        df["text"] = df.generated_report.fillna("")
    else:
        df["text"] = [findings_onward(r) or r for r in df.report2.fillna("")]
    return df


def chunks(text):
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    out, cur = [], []
    for s in re.split(r"(?<=[.!?])\s+", text):
        n = len(s.split())
        if cur and len(cur) + n > MAX_WORDS:
            out.append(" ".join(cur))
            cur = []
        cur += s.split()
        while len(cur) > MAX_WORDS:      # one run-on "sentence"
            out.append(" ".join(cur[:MAX_WORDS]))
            cur = cur[MAX_WORDS:]
    return out + ([" ".join(cur)] if cur else [])


def units(m, text):
    """Per Observation unit: root/member/anatomy tokens, change label, root RadGraph label (presence), mean member
    change probs (ATTRIBUTES order), margin (top-1 minus top-2)."""
    if not text.strip():
        return []
    out = []
    for c in chunks(text):
        ents = m.annotate(c)["entities"]
        for k, e in ents.items():
            if not e["head"]:
                continue
            mem = [x for x in ents.values() if x.get("unit") == k]
            out.append({"root": e["tokens"], "members": [x["tokens"] for x in mem],
                        "anatomy": [ents[t]["tokens"] for x in mem for r, t in x["relations"] if r == "located_at"],
                        "label": e["head"], "presence": e["label"],
                        "probs": np.mean([x["probs"] for x in mem], axis=0).round(4).tolist(), "margin": round(e["margin"], 4)})
    return out


def label(a):
    import torch
    from raddelta.metric import RadDelta
    df = load(a.src).iloc[a.shard::a.nshard]
    m = RadDelta(str(a.ckpt), 0) if a.ckpt else RadDelta(cuda=0)
    (F / "out").mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    for i, (ix, r) in enumerate(df.iterrows()):
        t = time.time()
        try:
            hf, err = units(m, r.text), ""
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            hf, err = None, "oom"
        recs.append({"row": ix, "units": hf, "sec": time.time() - t, "err": err, "n_words": len(r.text.split())})
        if i % 100 == 0:
            print(f"[{a.src} {a.shard}] {i}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    path = F / "out" / f"rgd_{a.src}_{a.shard}of{a.nshard}.json"
    json.dump({"wall": time.time() - t0, "recs": recs}, open(path, "w"))
    print(f"[{a.src} {a.shard}] done {len(recs)} in {time.time() - t0:.0f}s -> {path}")


def combine(a):
    """Shard JSONs -> {src}_units.parquet (one row per Observation unit) and {src}_reports.parquet (one row per pair)."""
    src = a.src
    df = load(src)
    keys = ["site", "study_uid1", "study_uid2"] if src == "bind" else ["study_uid1", "study_uid2"]
    assert not df.duplicated(keys).any(), "keys not unique"
    recs, wall = [], []
    for f in sorted((F / "out").glob(f"rgd_{src}_*.json")):
        d = json.load(open(f))
        recs += d["recs"]
        wall.append(d["wall"])
    r = pd.DataFrame(recs).set_index("row")
    assert len(r) == len(df) and r.index.isin(df.index).all(), f"{len(r)} rows for {len(df)} pairs"
    df = df.join(r)
    rep = df[keys].assign(n_units=df.units.map(lambda u: len(u) if isinstance(u, list) else None),
                          n_words=df.n_words, err=df.err, sec=df.sec)
    u = df[keys + ["units"]].explode("units").dropna(subset=["units"])
    u = pd.concat([u[keys].reset_index(drop=True), pd.json_normalize(u.units.tolist())], axis=1)
    u["unit_ix"] = u.groupby(keys).cumcount()
    rep.to_parquet(F / f"{src}_reports.parquet", index=False)
    u.to_parquet(F / f"{src}_units.parquet", index=False)
    print(f"{src}: {len(rep)} reports, {len(u)} units, oom {(rep.err == 'oom').sum()}, {len(wall)} shards")
    grp = rep.groupby("site") if src == "bind" else [("S4", rep)]
    for s, g in grp:
        uu = u[(u.study_uid2.isin(g.study_uid2)) & (u.site == s)] if src == "bind" else u
        ch = uu[uu.label != "NONE"]
        has = ch.groupby(keys).size().reindex(pd.MultiIndex.from_frame(g[keys])).fillna(0) > 0
        print(f"{s}: n={len(g)} units/report mean {g.n_units.mean():.1f}; NONE share {(uu.label == 'NONE').mean():.3f}; "
              f"reports with >=1 non-NONE unit {has.mean():.3f}")
        print("  non-NONE label distribution", ch.label.value_counts(normalize=True).reindex(CHANGE).round(3).to_dict())


# ================================================================== BIND snapshot
# Pairs = report pairs ∪ image-only pairs. Ready = every selected modality exists, is non-empty and is older than SETTLE_S
# (ants.image_write is not atomic).
SEL = ["bind_longitudinal_data_3d.csv", "bind_imageonly_data.csv", "bind_longitudinal_data.csv"]
OUT_MOD = dict(zip(MODALITIES, MODS))
SETTLE_S = 600
KEY = ["site", "study_uid1", "study_uid2"]


def _stat(p):
    try:
        s = os.stat(p)
        return s.st_size > 0, s.st_mtime
    except FileNotFoundError:
        return False, None


def snapshot(out_dir):
    t0 = time.time()
    stamp = time.strftime("%Y-%m-%dT%H%M%S", time.localtime(t0))
    cols = ["patient_uid", "site", "study_uid1", "study_uid2", "duration"]
    rep = pd.read_csv(PAIRED_CSV, dtype=str, usecols=cols)
    img = pd.read_csv(TABLES / "bind_imageonly_pairs.csv", dtype=str, usecols=cols)
    rep["has_report"], img["has_report"] = True, False
    pairs = pd.concat([rep, img]).sort_values("has_report", ascending=False)
    n_rep, n_img = len(rep), len(img)
    pairs = pairs.drop_duplicates(KEY).reset_index(drop=True)
    pairs["duration"] = pairs.duration.astype(int)
    sel = pd.concat([pd.read_csv(TABLES / f, dtype=str).assign(sel_file=f) for f in SEL])
    sel = sel.drop_duplicates(["site", "session_uid"])          # priority = order of SEL
    sess = set(zip(pairs.site, pairs.study_uid1)) | set(zip(pairs.site, pairs.study_uid2))
    sel = sel[[k in sess for k in zip(sel.site, sel.session_uid)]].reset_index(drop=True)
    files = [(i, m, str(session_dir(OUTPUT_ROOT, r.site, r.session_uid) / f"{m}.nii.gz"))
             for i, r in enumerate(sel.itertuples()) for m in MODALITIES if isinstance(getattr(r, m), str)]
    with ThreadPoolExecutor(64) as ex:
        st = list(ex.map(_stat, [f for _, _, f in files]))
    for m in MODALITIES:
        sel[OUT_MOD[m]] = None
    sel["n_sel"], sel["n_ok"], sel["n_fresh"] = 0, 0, 0
    for (i, m, f), (ok, mt) in zip(files, st):
        sel.at[i, "n_sel"] += 1
        if ok and mt <= t0 - SETTLE_S:
            sel.at[i, "n_ok"] += 1
            sel.at[i, OUT_MOD[m]] = f
        elif ok:
            sel.at[i, "n_fresh"] += 1
    sel["ready"] = (sel.n_sel > 0) & (sel.n_ok == sel.n_sel)
    ready = {k: r for k, r in zip(zip(sel.site, sel.session_uid), sel.itertuples()) if r.ready}
    ok = [(s, a) in ready and (s, b) in ready for s, a, b in zip(pairs.site, pairs.study_uid1, pairs.study_uid2)]
    out = pairs[ok].reset_index(drop=True)
    for t, col in ((1, "study_uid1"), (2, "study_uid2")):
        for m in OUT_MOD.values():
            out[f"{m}_{t}"] = [getattr(ready[(s, u)], m) for s, u in zip(out.site, out[col])]
    out.to_csv(out_dir / f"bind_ready_{stamp}.csv", index=False)
    stats = dict(stamp=stamp, n_report_pairs=n_rep, n_imageonly_pairs=n_img, n_union=len(pairs),
                 n_sessions_in_pairs=len(sess), n_sessions_with_selection=len(sel),
                 n_sel_files=dict(sel.sel_file.value_counts()), n_sessions_ready=len(ready),
                 n_sessions_blocked_by_fresh_file=int(((sel.n_fresh > 0) & (sel.n_ok + sel.n_fresh == sel.n_sel)).sum()),
                 n_pairs_ready=len(out), n_pairs_ready_with_report=int(out.has_report.sum()),
                 n_patients_ready=out[["site", "patient_uid"]].drop_duplicates().shape[0],
                 scan_seconds=round(time.time() - t0))
    return out, stats


# ================================================================== dupes
# BraTS-MET subjects duplicated in Yale (different pipelines, so compare images, not hashes). Feature: 4 mm block mean,
# intensity-normalised, brain-masked, z-scored; r = dot product. Duplicate when the best Yale match is as close as a
# typical Yale same-subject match (>= q25) and as distinct (gap to next subject >= q10). `pairs` drops the copies.
def dupe_feat(path):
    import nibabel as nib
    v = np.asarray(nib.load(path).dataobj, dtype=np.float32)[:192, :228, :192]
    v = v.reshape(48, 4, 57, 4, 48, 4).mean((1, 3, 5)).ravel()
    fg = v[v > 0]
    return (v / (np.percentile(fg, 99) if fg.size else 1.0)).astype(np.float16)


def subject_max(R, subj):
    """[n, m] volume r -> [n, n_subjects] max over each column subject's volumes."""
    return pd.DataFrame(R.T).groupby(pd.Series(subj).values).max().T


def dupes(a):
    import torch
    torch.set_num_threads(16)
    img = pd.read_csv(S3 / "image.csv", low_memory=False)
    img = img[img.T1ce.notna()]
    src = img.T1ce.str.split("/").str[4]           # path component 4 = S3 source
    img = img[src.isin(["BraTS-MET", "YALE"])]
    paths = img.T1ce.reset_index(drop=True)
    with ProcessPoolExecutor(16) as ex:
        X = np.stack(list(ex.map(dupe_feat, paths, chunksize=16)))
    X = X.astype(np.float32)
    name = paths.str.split("/").str[-1]
    subj, src = name.str.split("_").str[0].to_numpy(), paths.str.split("/").str[4].to_numpy()
    mu = X.mean(0)
    Fz = torch.from_numpy((X - mu)[:, mu > 0.15 * mu.max()])
    Fz = (Fz - Fz.mean(1, keepdim=True)) / Fz.std(1, keepdim=True) / Fz.shape[1] ** 0.5
    met, yale = src == "BraTS-MET", src == "YALE"
    Ry = (Fz[yale] @ Fz[yale].T).numpy()
    np.fill_diagonal(Ry, -9)
    ys = subj[yale]
    same = ys[:, None] == ys[None, :]
    bs, bo = np.where(same, Ry, -9).max(1), np.where(same, -9, Ry).max(1)
    ok = bs > -9
    r_thr, gap_thr = np.quantile(bs[ok], 0.25), np.quantile((bs - bo)[ok], 0.10)
    top1 = float((bs > bo)[ok].mean())
    M = subject_max((Fz[met] @ Fz[yale].T).numpy(), ys).to_numpy()
    cols = np.array(sorted(set(ys)))
    o = np.argsort(-M, 1)
    r1, r2 = M[np.arange(len(M)), o[:, 0]], M[np.arange(len(M)), o[:, 1]]
    d = pd.DataFrame({"met_volume": paths[met].values, "met_subject": subj[met], "yale_subject": cols[o[:, 0]],
                      "r_top": r1.round(4), "gap": (r1 - r2).round(4)})
    d["duplicate"] = (d.r_top >= r_thr) & (d.gap >= gap_thr)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "dupes_brats_met_yale_volumes.csv", index=False)
    hub = d.yale_subject.value_counts()
    res = {"method": "4 mm residual T1ce correlation, Yale same-subject calibration",
           "n_met_volumes": int(met.sum()), "n_met_subjects": int(len(set(subj[met]))),
           "n_yale_volumes": int(yale.sum()), "n_yale_subjects": int(len(set(ys))),
           "yale_same_subject_retrieval_top1": round(top1, 3),
           "yale_same_subject_r_q05_q25_q50": np.quantile(bs[ok], [.05, .25, .5]).round(3).tolist(),
           "yale_same_subject_gap_q10_q50": np.quantile((bs - bo)[ok], [.1, .5]).round(3).tolist(),
           "threshold_r": round(float(r_thr), 3), "threshold_gap": round(float(gap_thr), 3),
           "met_to_yale_r_top_q50_q99_max": d.r_top.quantile([.5, .99, 1]).round(3).tolist(),
           "met_to_yale_gap_q50_q99_max": d.gap.quantile([.5, .99, 1]).round(3).tolist(),
           "met_top_match_distinct_yale_subjects": int(hub.size), "met_top_match_hubs": hub.head(3).to_dict(),
           "duplicate_volumes": int(d.duplicate.sum()),
           "duplicate_met_subjects": sorted(set(d.met_subject[d.duplicate]))}
    json.dump(res, open(OUT / "dupes_brats_met_yale.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k not in ("duplicate_met_subjects", "met_top_match_hubs")})


# ================================================================== probe pairs
# BraTS pairs of the encoder change probe (held out of Stage 1): first modality with a lesion mask at both timepoints,
# >= 3 change (XOR) and >= 3 stable (AND) enhancing-tumour tokens; first 500 qualify. Re-targeted to T1ce with T1ce masks.
PROBE_LABELS, PROBE_N, PROBE_MIN = [4, 3], 500, 3


def lesion_tokens(mask_path, labels, thresh=0.05):
    """Label mask -> [2016] bool: binarise the selected labels, NeuroVFM grid, token = occupancy > thresh."""
    import nibabel as nib
    from braindiff.data.transforms import PATCH, to_neurovfm_grid
    GD, GH, GW = TOKEN_GRID
    img = nib.load(str(mask_path))
    sel = np.isin(np.rint(img.get_fdata(dtype=np.float32)), list(labels)).astype(np.float32)
    arr = (np.asarray(to_neurovfm_grid(sel, nib.aff2axcodes(img.affine))) > 1e-6).astype(np.float32)
    assert arr.shape == (GD * PATCH[0], GH * PATCH[1], GW * PATCH[2]), arr.shape
    return (arr.reshape(GD, PATCH[0], GH, PATCH[1], GW, PATCH[2]).mean(axis=(1, 3, 5)) > thresh).reshape(-1)


def probe_pairs(a):
    li = pd.read_csv(S3 / "lesion_index.csv")
    mask_map, sub, ds = (dict(zip(li.volume_path, li[c])) for c in ("mask_path", "subject", "dataset"))
    img = pd.read_csv(S3 / "image.csv", low_memory=False).set_index("UID")
    rows = []
    for r in pd.read_csv(S3 / "main.csv", low_memory=False).itertuples():
        if len(rows) >= PROBE_N:
            break
        ua, ub = r.UID_1, r.UID_2
        if ua not in img.index or ub not in img.index:
            continue
        mods = [m for m in MODS if isinstance(img.loc[ua].get(m), str) and isinstance(img.loc[ub].get(m), str)]
        if not mods:
            continue
        pa, pb = img.loc[ua, mods[0]], img.loc[ub, mods[0]]
        ma, mb = mask_map.get(pa), mask_map.get(pb)
        if ma is None or mb is None:
            continue
        try:
            ta, tb = lesion_tokens(ma, PROBE_LABELS), lesion_tokens(mb, PROBE_LABELS)
        except Exception:
            continue
        ch, st = ta ^ tb, ta & tb
        if int(ch.sum()) < PROBE_MIN or int(st.sum()) < PROBE_MIN:
            continue
        rows.append(dict(pair=len(rows), uid_a=ua, uid_b=ub, mod=mods[0], subject=sub[pa], dataset=ds[pa],
                         n_change=int(ch.sum()), n_stable=int(st.sum()), n_a=int(ta.sum()), n_b=int(tb.sum()),
                         grew=int((tb & ~ta).sum()), shrank=int((ta & ~tb).sum()), path_a=pa, path_b=pb))
    t1w = pd.DataFrame(rows)
    out, drop = [], {"no_volume": 0, "no_mask": 0, "too_few": 0}
    for r in t1w.to_dict("records"):
        pa, pb = img.loc[r["uid_a"]].get("T1ce"), img.loc[r["uid_b"]].get("T1ce")
        if not isinstance(pa, str) or not isinstance(pb, str):
            drop["no_volume"] += 1
            continue
        if pa not in mask_map or pb not in mask_map:
            drop["no_mask"] += 1
            continue
        ta, tb = lesion_tokens(mask_map[pa], PROBE_LABELS), lesion_tokens(mask_map[pb], PROBE_LABELS)
        ch, st = ta ^ tb, ta & tb
        if int(ch.sum()) < PROBE_MIN or int(st.sum()) < PROBE_MIN:
            drop["too_few"] += 1
            continue
        out.append(r | dict(mod="T1ce", path_a=pa, path_b=pb, n_change_t1w=r["n_change"], n_stable_t1w=r["n_stable"],
                            n_change=int(ch.sum()), n_stable=int(st.sum())))
    (BUILD / "probes").mkdir(parents=True, exist_ok=True)
    t1w.to_csv(BUILD / "probes" / "phase0_pairs.csv", index=False)
    pd.DataFrame(out).to_csv(BUILD / "probes" / "pairs_t1ce.csv", index=False)
    print(f"T1w list {len(t1w)} pairs; T1ce kept {len(out)}, dropped {drop}; subjects {pd.DataFrame(out).subject.nunique()}")


# ================================================================== L_mag
# Whole-tumour voxels per BraTS mask, globally and per NeuroVFM token (to_neurovfm_grid, 4x16x16 patches, row-major).
# Empty T1ce mask at either timepoint -> has_lmag=False. Not used by Stage 1-D.
TOKEN_GRID = (12, 14, 12)          # 193x229x193 @ 1 mm -> (48, 224, 192) -> 2016 tokens
NOTE = ("pairs with an empty T1ce mask are excluded from L_mag (has_lmag=False, pair kept); "
        "per-token targets per sequence from that sequence's own mask (was T1ce for all)")


def token_counts(mask_path):
    """-> (global voxel count, int32 [2016] voxel count per token). 1 voxel = 1 mm^3 (asserted)."""
    import nibabel as nib
    from braindiff.data.transforms import PATCH, to_neurovfm_grid
    GD, GH, GW = TOKEN_GRID
    img = nib.load(str(mask_path))
    assert np.allclose(img.header.get_zooms()[:3], 1.0), f"{mask_path}: not 1 mm"
    sel = (np.asarray(img.dataobj) != 0).astype(np.float32)
    v = to_neurovfm_grid(sel, nib.aff2axcodes(img.affine)).numpy() * PATCH[0]
    assert v.shape == (GD * PATCH[0], GH * PATCH[1], GW * PATCH[2]), v.shape
    tok = v.reshape(GD, PATCH[0], GH, PATCH[1], GW, PATCH[2]).sum((1, 3, 5)).reshape(-1)
    return int(sel.sum()), np.rint(tok).astype(np.int32)


def same_across_modalities(paths):
    """Masks [T1w, T1ce, T2w, FLAIR] -> (binary identical, labels identical, Dice of T1ce vs each other)."""
    import nibabel as nib
    arr = [np.asarray(nib.load(p).dataobj) for p in paths]
    t = arr[1] != 0
    dice = [2 * (t & (a != 0)).sum() / (t.sum() + (a != 0).sum()) if t.any() or a.any() else np.nan
            for i, a in enumerate(arr) if i != 1]                    # NaN = both empty
    return (all(np.array_equal(a != 0, t) for a in arr), all(np.array_equal(a, arr[1]) for a in arr),
            [float(x) for x in dice])


def sequence_targets(df, li, workers=16):
    """BraTS rows -> (has_lmag bool, lmag_tokens.npz dict, stats); NaN where a volume or mask is missing."""
    GD, GH, GW = TOKEN_GRID
    mask = dict(zip(li.volume_path, li.mask_path))
    b = df[df.source.isin(["BraTS-GLI", "BraTS-MET"])]
    mp = {f"{m}_{k}": b[f"{m}_{k}"].map(mask) for m in MODS for k in "12"}
    uniq = sorted({x for v in mp.values() for x in v.dropna()})
    with ProcessPoolExecutor(workers) as ex:
        res = dict(zip(uniq, ex.map(token_counts, uniq, chunksize=8)))
    nvox = dict(zip(li.mask_path, li.n_lesion_voxels))
    assert all(res[x][0] == nvox[x] for x in uniq), "global count != lesion_index n_lesion_voxels"
    t1ce_empty = (mp["T1ce_1"].map(lambda m: res[m][0] == 0) | mp["T1ce_2"].map(lambda m: res[m][0] == 0)).to_numpy()
    keep = b[~t1ce_empty]
    n = len(keep)
    va, vb = (np.full((n, len(MODS), GD * GH * GW), np.nan, np.float32) for _ in "ab")
    va_g, vb_g = (np.full((n, len(MODS)), np.nan, np.float32) for _ in "ab")
    for j, m in enumerate(MODS):
        a, c = mp[f"{m}_1"][keep.index], mp[f"{m}_2"][keep.index]
        for i in np.flatnonzero((a.notna() & c.notna()).to_numpy()):
            va_g[i, j], va[i, j] = res[a.iloc[i]]
            vb_g[i, j], vb[i, j] = res[c.iloc[i]]
    valid = ~np.isnan(va_g)
    lost = [res[x][0] - res[x][1].sum() for x in uniq]
    st = {"note": NOTE, "lmag_brats_pairs": len(b), "lmag_excluded_empty_t1ce_pairs": int(t1ce_empty.sum()),
          "lmag_pairs_kept": n, "lmag_masks_all_sequences": len(uniq),
          "lmag_voxels_outside_token_grid_max": int(max(lost)),
          "lmag_voxels_outside_token_grid_masks_nonzero": int(sum(x > 0 for x in lost)),
          "lmag_valid_pairs_per_sequence": dict(zip(MODS, valid.sum(0).tolist())),
          "lmag_pairs_all_4_sequences_valid": int(valid.all(1).sum()),
          "lmag_kept_pairs_with_empty_mask_per_sequence": dict(zip(MODS, ((va_g == 0) | (vb_g == 0)).sum(0).tolist()))}
    out = dict(pair_id=keep.pair_id.to_numpy(), mods=np.array(MODS), va=va, vb=vb, va_g=va_g, vb_g=vb_g)
    return df.index.isin(keep.index), out, st


# ================================================================== pairs
def flat(s):
    return {"/".join(map(str, k)) if isinstance(k, tuple) else str(k): int(v) for k, v in s.items()}


def stat(key, val):
    STATS[key] = val
    print(f"  {key}: {val}")
    return val


def holdouts():
    held = set()
    for s in ("val", "test"):
        held |= set(pd.read_csv(S4 / f"splits_extended/s4_{s}.csv", usecols=["patient_uid"]).patient_uid.astype(str))
    img = pd.read_csv(S3 / "image.csv", low_memory=False)
    probe = pd.read_csv(BUILD / "probes" / "pairs_t1ce.csv")
    uid2st = dict(zip(img.UID, img.StudyUID))
    probe_st = set(probe.uid_a.map(uid2st))
    assert len(set(probe.subject)) == len(probe_st), "probe StudyUID != subject"
    stat("holdout_s4_valtest_patients", len(held))
    stat("holdout_brats_probe_subjects", len(probe_st))
    return held, probe_st, sorted(set(probe.subject))


def mrrate_dates():
    out = []
    for f in sorted((MRRATE / "metadata").glob("batch*_metadata.csv")):
        out.append(pd.read_csv(f, usecols=["study_uid", "anon_study_date"]).drop_duplicates("study_uid"))
    d = pd.concat(out).drop_duplicates("study_uid")
    return dict(zip(d.study_uid, pd.to_datetime(d.anon_study_date)))


def s3_rows(st2pt, dates):
    img = pd.read_csv(S3 / "image.csv", low_memory=False).set_index("UID")
    p = img[MODS].bfill(axis=1).iloc[:, 0].fillna("").astype(str)
    stat("s3_image_rows_without_any_path", int((p == "").sum()))
    part = p.str.split("/")
    src = part.str[4].fillna("none")         # S3 paths are <root>/<a>/<b>/BRAIN_DIFF_S3/<source>/...: component 4 = source
    img["src"] = src.where(~src.str.startswith("batch"), "MR-RATE")
    mr = img.src == "MR-RATE"
    img["session"] = np.where(mr, part.str[5], img.index)
    # Patient = real subject id from the file name (StudyUIDs are minted per build run).
    img["subject"] = np.where(mr, img.session.map(st2pt).astype(str), part.str[-1].str.split("_").str[0])
    img = img[img.src != "none"]
    mr = img.src == "MR-RATE"
    stat("s3_mrrate_sessions_without_patient", int((mr & img.session.map(st2pt).isna()).sum()))
    chk = img[~mr].groupby("StudyUID").subject.nunique().max(), img[~mr].groupby("subject").StudyUID.nunique().max()
    assert chk == (1, 1), f"StudyUID vs subject not 1:1 {chk}"
    img["t"] = np.where(mr, img.session.map(dates).map(lambda d: d.toordinal() if pd.notna(d) else np.nan),
                        np.where(img.src == "OASIS-3", img.days, np.nan))
    m = pd.read_csv(S3 / "main.csv")
    stat("s3_main_pairs", len(m))
    nopath = ~(m.UID_1.isin(img.index) & m.UID_2.isin(img.index))
    stat("s3_pairs_dropped_session_without_path", int(nopath.sum()))
    m = m[~nopath]
    # Time order: MR-RATE by metadata date, else Timepoint.
    key = img.t.where(img.src == "MR-RATE", img.Timepoint)
    k1, k2 = m.UID_1.map(key), m.UID_2.map(key)
    sw = k1 > k2
    stat("s3_pairs_reversed_order", flat(m[sw].UID_1.map(img.src).value_counts()))
    m.loc[sw, ["UID_1", "UID_2"]] = m.loc[sw, ["UID_2", "UID_1"]].values
    n0 = len(m)
    m = m.drop_duplicates(["UID_1", "UID_2"])
    stat("s3_pairs_dropped_as_reverse_duplicates", n0 - len(m))
    r = pd.DataFrame({"source": m.UID_1.map(img.src).values, "origin": "S3",
                      "patient": m.UID_1.map(img.subject).values,
                      "study1": m.UID_1.map(img.session).values, "study2": m.UID_2.map(img.session).values,
                      "s3_studyuid": m.UID_1.map(img.StudyUID).values,
                      "t1": m.UID_1.map(img.t).values, "t2": m.UID_2.map(img.t).values,
                      "tp1": m.UID_1.map(img.Timepoint).values, "tp2": m.UID_2.map(img.Timepoint).values})
    for k, u in (("1", m.UID_1), ("2", m.UID_2)):
        for mod in MODS:
            r[f"{mod}_{k}"] = u.map(img[mod]).values
    r["duration"] = r.t2 - r.t1
    stat("s3_pairs_by_source", r.source.value_counts().to_dict())
    return r


def s4_rows(dates):
    m = pd.read_csv(S4 / "main_extended.csv", low_memory=False)
    img = pd.read_csv(S4 / "image_extended.csv", low_memory=False)
    assert img.study_uid.is_unique
    img = img.set_index("study_uid")
    stat("s4_extended_pairs", len(m))
    r = pd.DataFrame({"source": "MR-RATE", "origin": "S4", "patient": m.patient_uid.astype(str),
                      "study1": m.study_uid1, "study2": m.study_uid2, "duration": m.duration.astype(float),
                      "classification": m.classification})
    for k, c in (("1", m.study_uid1), ("2", m.study_uid2)):
        for mod in MODS:
            r[f"{mod}_{k}"] = c.map(img[mod]).values
    dd = (r.study2.map(dates) - r.study1.map(dates)).dt.days
    stat("s4_duration_equals_metadata_date_diff", f"{int((dd == r.duration).sum())}/{len(r)}")
    r["t1"] = r.study1.map(dates).map(lambda d: d.toordinal() if pd.notna(d) else np.nan)
    r["t2"] = r.study2.map(dates).map(lambda d: d.toordinal() if pd.notna(d) else np.nan)
    return r


def bind_rows():
    snap = os.environ.get("STAGE1_BIND_SNAPSHOT")   # reuse a stored snapshot
    if snap:
        b = pd.read_csv(snap, dtype=str)
        b["duration"], b["has_report"] = b.duration.astype(int), b.has_report == "True"
        s = dict(stamp=Path(snap).stem.removeprefix("bind_ready_"), from_file=snap, n_pairs_ready=len(b),
                 n_pairs_ready_with_report=int(b.has_report.sum()))
    else:
        b, s = snapshot(OUT)
    STATS["bind_snapshot"] = s
    print("  bind_snapshot:", s)
    d = {}
    for site in b.site.unique():
        x = load_dates(site)
        d.update({(site, u): t.toordinal() for u, t in zip(x.session_uid, x.study_date)})
    r = pd.DataFrame({"source": "BIND", "origin": "BIND", "site": b.site,
                      "patient": b.site + ":" + b.patient_uid, "study1": b.study_uid1, "study2": b.study_uid2,
                      "duration": b.duration.astype(float), "has_report": b.has_report,
                      "t1": [d.get(k, np.nan) for k in zip(b.site, b.study_uid1)],
                      "t2": [d.get(k, np.nan) for k in zip(b.site, b.study_uid2)]})
    for k in "12":
        for mod in MODS:
            r[f"{mod}_{k}"] = b[f"{mod}_{k}"].values
    stat("bind_duration_equals_date_diff", f"{int(((r.t2 - r.t1) == r.duration).sum())}/{len(r)}")
    return r, s["stamp"]


def lent_table(units, keys):
    p = np.stack(units.probs.values).astype(np.float64)
    assert np.allclose(p.sum(1), 1, atol=1e-2), "probs do not sum to 1"
    u = units[keys].copy()
    u[CHANGE] = p[:, :5]                       # = w * normalised probs[:5], w = 1 - p_NONE
    u["w"] = 1 - p[:, 5]
    u["nn"] = (units.label != "NONE").astype(int)
    g = u.groupby(keys)[CHANGE + ["w", "nn"]].sum()
    v = g[CHANGE].to_numpy()
    g["lent"] = list(np.where(g.w.to_numpy()[:, None] > 0, v / np.maximum(v.sum(1, keepdims=True), 1e-12), np.nan))
    return g.rename(columns={"w": "lent_weight", "nn": "lent_n_units"})[["lent", "lent_weight", "lent_n_units"]]


def _join(df, mask, cols, tab):
    t = tab.reindex(pd.MultiIndex.from_frame(df.loc[mask, cols]))
    idx = df.index[mask]
    df.loc[idx, "lent_weight"] = t.lent_weight.fillna(0).to_numpy()
    df.loc[idx, "lent_n_units"] = t.lent_n_units.fillna(0).astype(int).to_numpy()
    lent = df.lent.tolist()
    for pos, v in zip(df.index.get_indexer(idx), t.lent):
        lent[pos] = v if isinstance(v, np.ndarray) else None
    df["lent"] = lent


def attach_lent(df):
    df["lent"], df["lent_weight"], df["lent_n_units"] = [None] * len(df), 0.0, 0
    df["lent_drop"] = ""
    # S4 / MR-RATE: generated_report (s4gen); S3-only MR-RATE pairs get none.
    s4 = lent_table(pd.read_parquet(F / "s4gen_units.parquet"), ["study_uid1", "study_uid2"])
    rep = pd.read_parquet(F / "s4gen_reports.parquet")
    stat("s4gen_reports", len(rep)), stat("s4gen_oom", int((rep.err == "oom").sum()))
    _join(df, df.origin.str.contains("S4"), ["study1", "study2"], s4)
    # BIND: current report (report pairs only).
    bu = lent_table(pd.read_parquet(F / "bind_units.parquet"), ["site", "study_uid1", "study_uid2"])
    j = df.source == "BIND"
    _join(df, j, ["site", "study1", "study2"], bu)
    cand = df.lent_weight >= MIN_LENT_WEIGHT
    stat("lent_weight_ge_0.5_by_source_origin", flat(df[cand].groupby(["source", "origin"]).size()))
    stat("lent_weight_lt_0.5_with_any_units", int(((df.lent_weight > 0) & ~cand).sum()))
    stat("lent_weight_zero_among_labelled_sources",
         int(((df.lent_weight == 0) & (df.origin.str.contains("S4") | (j & df.has_report.fillna(False).astype(bool)))).sum()))
    # BIND label filters (keep the pair, drop only the label).
    c = pd.read_csv(COVERAGE, dtype=str)
    btw = c[["between_mr_brain", "between_not_available", "between_other"]].astype(float)
    c["f_between"] = np.where(c.site == "I0004", btw.sum(1) > 0, btw.between_not_available > 0)
    c["f_ct_only"] = c.comp_modality == "CT_only"
    c = c.set_index(["site", "study_uid1", "study_uid2"])
    t = c.reindex(pd.MultiIndex.from_arrays([df.loc[j, "site"], df.loc[j, "study1"], df.loc[j, "study2"]]))
    fb = pd.Series(t.f_between.fillna(False).astype(bool).values, index=df.index[j])
    fc = pd.Series(t.f_ct_only.fillna(False).astype(bool).values, index=df.index[j])
    cj = cand[j]
    stat("bind_lent_removed_by_intervening_exam", int((fb & cj).sum()))
    stat("bind_lent_removed_by_intervening_exam_by_site", df.loc[j][(fb & cj).values].site.value_counts().to_dict())
    stat("bind_lent_removed_by_ct_only_comparison", int((fc & cj).sum()))
    stat("bind_lent_removed_by_both", int((fb & fc & cj).sum()))
    stat("bind_lent_removed_by_either", int(((fb | fc) & cj).sum()))
    df.loc[fb[fb].index, "lent_drop"] += "intervening_exam;"
    df.loc[fc[fc].index, "lent_drop"] += "ct_only_comparison;"
    df["has_lent"] = cand & (df.lent_drop == "")
    df["lent"] = [v if h else None for v, h in zip(df.lent, df.has_lent)]
    return df


def lmag(df):
    li = pd.read_csv(S3 / "lesion_index.csv")
    mask = dict(zip(li.volume_path, li.mask_path))
    nvox = dict(zip(li.mask_path, li.n_lesion_voxels))
    b = df[df.source.isin(["BraTS-GLI", "BraTS-MET"])]
    stat("lmag_pairs", len(b))
    ma, mb = b.T1ce_1.map(mask), b.T1ce_2.map(mask)
    assert ma.notna().all() and mb.notna().all(), "BraTS T1ce without mask"
    uniq = sorted(set(ma) | set(mb))
    with ProcessPoolExecutor(16) as ex:
        res = dict(zip(uniq, ex.map(token_counts, uniq, chunksize=8)))
    assert all(res[m][0] == nvox[m] for m in uniq), "global count != lesion_index n_lesion_voxels"
    lost = [res[m][0] - res[m][1].sum() for m in uniq]
    stat("lmag_masks", len(uniq))
    stat("lmag_voxels_outside_token_grid_max", int(max(lost)))
    stat("lmag_voxels_outside_token_grid_masks_nonzero", int(sum(x > 0 for x in lost)))
    empty = {m for m in uniq if res[m][0] == 0}
    emp_all = li[(li.modality == "T1ce") & (li.n_lesion_voxels == 0)]
    stat("empty_t1ce_masks_all_brats", emp_all.dataset.value_counts().to_dict())
    stat("empty_t1ce_masks_in_lmag_pairs", len(empty))
    # Modality agreement on 20 random pairs (40 timepoints).
    rng = np.random.default_rng(SEED)
    sel = b.iloc[rng.choice(len(b), 20, replace=False)]
    ok_bin, ok_lab, dice = 0, 0, []
    for _, r in sel.iterrows():
        for k in "12":
            a, l, d = same_across_modalities([mask[r[f"{m}_{k}"]] for m in MODS])
            ok_bin += a
            ok_lab += l
            dice += [(r.source, x) for x in d]
    stat("mask_modality_identical_20pairs_binary", f"{ok_bin}/40")
    stat("mask_modality_identical_20pairs_labels", f"{ok_lab}/40")
    dd = pd.DataFrame(dice, columns=["src", "dice"]).dropna()
    stat("mask_timepoints_empty_in_all_modalities_20pairs", int(40 - len(dd) / 3))
    stat("mask_dice_t1ce_vs_other_20pairs", {s: dict(n=len(g), q0=round(g.dice.min(), 3), q50=round(g.dice.median(), 3),
                                                     mean=round(g.dice.mean(), 3)) for s, g in dd.groupby("src")})
    has, out, st = sequence_targets(df, li)
    for k, v in st.items():
        stat(f"lmag_per_sequence/{k}", v)
    df["has_lmag"] = has
    df["lmag_V1"] = np.nan
    df["lmag_V2"] = np.nan
    df.loc[b.index, "lmag_V1"] = [res[m][0] for m in ma]
    df.loc[b.index, "lmag_V2"] = [res[m][0] for m in mb]
    df["lmag_empty_mask"] = False
    df.loc[b.index, "lmag_empty_mask"] = (ma.isin(empty) | mb.isin(empty)).values
    stat("lmag_pairs_with_empty_mask", int(df.lmag_empty_mask.sum()))
    np.savez_compressed(OUT / "lmag_tokens.npz", **out)
    return df


def triplets(df):
    # Per patient: sessions of its rows in time order; consecutive windows of 3.
    # Order key: Timepoint for YALE/BraTS (no dates), date ordinal / OASIS days otherwise.
    tp = df.source.isin(["YALE", "BraTS-GLI", "BraTS-MET"])
    sess = pd.concat([
        df.assign(session=df.study1, t=df.tp1.where(tp, df.t1), **{m: df[f"{m}_1"] for m in MODS}),
        df.assign(session=df.study2, t=df.tp2.where(tp, df.t2), **{m: df[f"{m}_2"] for m in MODS})])
    sess = sess[["source", "patient", "split", "session", "t"] + MODS].drop_duplicates(["source", "patient", "session"])
    stat("triplet_sessions_without_time", int(sess.t.isna().sum()))
    pairs = set(zip(df.source, df.study1, df.study2))
    rows, ties = [], 0
    for (src, pt), g in sess.dropna(subset=["t"]).sort_values("t").groupby(["source", "patient"], sort=False):
        g = g.to_dict("records")
        for a, b_, c in zip(g, g[1:], g[2:]):
            if not a["t"] < b_["t"] < c["t"]:
                ties += 1
                continue
            r = {"source": src, "patient": pt, "split": a["split"],
                 "studyA": a["session"], "studyB": b_["session"], "studyC": c["session"],
                 "tA": a["t"], "tB": b_["t"], "tC": c["t"],
                 "has_AB": (src, a["session"], b_["session"]) in pairs,
                 "has_BC": (src, b_["session"], c["session"]) in pairs,
                 "has_AC": (src, a["session"], c["session"]) in pairs}
            for s, x in (("A", a), ("B", b_), ("C", c)):
                r.update({f"{m}_{s}": x[m] for m in MODS})
            rows.append(r)
    tr = pd.DataFrame(rows)
    stat("triplets_skipped_tied_time", ties)
    stat("triplets_by_source", tr.source.value_counts().to_dict())
    stat("triplets_by_split", tr.split.value_counts().to_dict())
    stat("triplets_with_AB_and_BC_pairs", int((tr.has_AB & tr.has_BC).sum()))
    stat("triplets_with_all_three_pairs", int((tr.has_AB & tr.has_BC & tr.has_AC).sum()))
    return tr


def check_files(df):
    cols = [f"{m}_{k}" for k in "12" for m in MODS]
    uniq = [p for p in pd.unique(df[cols].to_numpy().ravel()) if isinstance(p, str)]
    with ThreadPoolExecutor(64) as ex:
        ok = dict(zip(uniq, ex.map(os.path.isfile, uniq)))
    stat("files_referenced", len(uniq)), stat("files_missing", int(sum(not v for v in ok.values())))
    miss = pd.Series([p.split("/")[3] + "/" + p.split("/")[4] for p, v in ok.items() if not v])
    stat("files_missing_by_dir", miss.str.replace(r"/batch\d+", "/batch*", regex=True).value_counts().to_dict())
    for c in cols:
        df[c] = df[c].where(df[c].map(lambda p: isinstance(p, str) and ok[p]))
    for k in "12":
        df[f"n_mod_{k}"] = df[[f"{m}_{k}" for m in MODS]].notna().sum(1)
    df["n_common"] = sum((df[f"{m}_1"].notna() & df[f"{m}_2"].notna()).astype(int) for m in MODS)
    return df


def build_pairs(a):
    OUT.mkdir(parents=True, exist_ok=True)
    print("holdouts"); held, probe_st, probe_subj = holdouts()
    sp = pd.read_csv(MRRATE / "splits.csv")
    st2pt = dict(zip(sp.study_uid, sp.patient_uid.astype(str)))
    print("dates"); dates = mrrate_dates()
    print("S3"); s3 = s3_rows(st2pt, dates)
    print("S4"); s4 = s4_rows(dates)
    stat("s4_patient_differs_from_mrrate_splits", int((s4.study1.map(st2pt) != s4.patient).sum()))
    # MR-RATE dedupe: keep the S4 row (it carries the generated_report label).
    m3 = s3[s3.source == "MR-RATE"]
    k4 = set(zip(s4.study1, s4.study2))
    both = [k in k4 for k in zip(m3.study1, m3.study2)]
    stat("mrrate_s3_pairs", len(m3)), stat("mrrate_s3_in_s4", int(sum(both)))
    stat("mrrate_s3_reversed_in_s4", int(sum((b, a) in k4 for a, b in zip(m3.study1, m3.study2))))
    s4.loc[[k in set(zip(m3.study1, m3.study2)) for k in zip(s4.study1, s4.study2)], "origin"] = "S3+S4"
    s3 = s3.drop(m3.index[both])
    print("BIND"); bind, stamp = bind_rows()
    df = pd.concat([s3, s4, bind], ignore_index=True)
    stat("pairs_before_exclusions_by_source", df.source.value_counts().to_dict())
    # Duplicate BraTS-MET subjects (copy of a Yale subject) -> drop the BraTS-MET copy.
    dup = json.load(open(OUT / "dupes_brats_met_yale.json"))
    df["drop"] = ""
    df.loc[(df.source == "BraTS-MET") & df.patient.isin(dup["duplicate_met_subjects"]), "drop"] = "brats_met_yale_duplicate"
    df.loc[(df.source == "MR-RATE") & df.patient.isin(held), "drop"] = "s4_valtest_patient"
    df.loc[df.source.str.startswith("BraTS") & df.s3_studyuid.isin(probe_st), "drop"] = "brats_probe_subject"
    stat("dropped", flat(df[df["drop"] != ""].groupby(["drop", "origin"]).size()))
    stat("dropped_patients", df[df["drop"] != ""].groupby("drop").patient.nunique().to_dict())
    df = df[df["drop"] == ""].drop(columns="drop").reset_index(drop=True)
    df = check_files(df)
    bad = (df.n_mod_1 == 0) | (df.n_mod_2 == 0)
    stat("pairs_dropped_no_existing_modality_at_a_timepoint", int(bad.sum()))
    df = df[~bad].reset_index(drop=True)
    stat("pairs_with_no_common_modality", int((df.n_common == 0).sum()))
    # Monitor split: 5% of patients per source, seed 0, fresh generator per source.
    mon = set()
    for src in sorted(df.source.unique()):
        p = np.sort(df.loc[df.source == src, "patient"].unique())
        mon |= {(src, x) for x in np.random.default_rng(SEED).permutation(p)[:int(round(MONITOR_FRAC * len(p)))]}
    df["split"] = ["monitor" if k in mon else "train" for k in zip(df.source, df.patient)]
    df["pair_id"] = np.arange(len(df))
    print("L_ent"); df = attach_lent(df)
    print("L_mag"); df = lmag(df)
    print("triplets"); tr = triplets(df)
    df["duration_days"] = df.duration
    cols = (["pair_id", "source", "origin", "site", "patient", "study1", "study2", "split", "duration_days",
             "has_report", "classification"] + [f"{m}_{k}" for k in "12" for m in MODS] +
            ["n_mod_1", "n_mod_2", "n_common", "has_lent", "lent", "lent_weight", "lent_n_units", "lent_drop",
             "has_lmag", "lmag_V1", "lmag_V2", "lmag_empty_mask", "tp1", "tp2", "t1", "t2"])
    df = df[cols]
    df["lent"] = [None if v is None else [float(x) for x in v] for v in df.lent]
    df.to_parquet(OUT / "pairs.parquet", index=False)
    tr.to_parquet(OUT / "triplets.parquet", index=False)
    g = df.groupby(["source", "split"])
    stat("final_pairs", flat(g.size()))
    stat("final_patients", flat(g.patient.nunique()))
    stat("final_pairs_total", len(df)), stat("final_patients_total", int(df[["source", "patient"]].drop_duplicates().shape[0]))
    stat("final_has_lent", df.groupby("source").has_lent.sum().astype(int).to_dict())
    stat("final_has_lmag", int(df.has_lmag.sum()))
    stat("final_duration_known", df.groupby("source").duration_days.apply(lambda s: int(s.notna().sum())).to_dict())
    stat("final_mrrate_origin", df[df.source == "MR-RATE"].origin.value_counts().to_dict())
    lv = np.stack(df.loc[df.has_lent, "lent"].values)
    stat("lent_mean_vector", dict(zip(CHANGE, lv.mean(0).round(3).tolist())))
    stat("lent_argmax_share", dict(zip(CHANGE, (np.bincount(lv.argmax(1), minlength=5) / len(lv)).round(3).tolist())))
    stat("lent_by_classification_argmax",
         pd.crosstab(df.loc[df.has_lent, "classification"].fillna("none"),
                     pd.Series(np.array(CHANGE)[lv.argmax(1)], index=df.index[df.has_lent])).to_dict())
    json.dump({"s4_valtest_patient_uid": sorted(held, key=int), "brats_probe_studyuid": sorted(probe_st),
               "brats_probe_subject": probe_subj, "brats_met_yale_duplicate_subject": dup["duplicate_met_subjects"],
               "monitor": {s: sorted(p for ss, p in mon if ss == s) for s in sorted({s for s, _ in mon})},
               "bind_snapshot": stamp}, open(OUT / "holdouts.json", "w"), indent=1)
    json.dump(STATS, open(OUT / "stats.json", "w"), indent=1, default=str)


# ================================================================== entity_text
# Finding-level change text from the Rad-Δ units: each NEW / WORSENED / IMPROVED / RESOLVED unit with p >= 0.5 becomes
# "<finding> (<anatomy>)", grouped by status, then a closing line, e.g.
#     NEW: gliotic foci (subcortical). WORSENED: radiation necrosis (cerebellar, parietal). No other interval change.
# Finding = member tokens minus change / degree words and measurements. STABLE is left out. ent_ok = False when a
# dropped unit carried a status the text lacks. text_statuses is the Stage 1-D status target.
STATUS = ["NEW", "WORSENED", "IMPROVED", "STABLE", "RESOLVED", "NONE"]      # units.probs order
ECHANGE = ["NEW", "WORSENED", "IMPROVED", "RESOLVED"]
EREVERSE = {"NEW": "RESOLVED", "RESOLVED": "NEW", "WORSENED": "IMPROVED", "IMPROVED": "WORSENED"}
P_MIN = 0.5
STOP = set("""new newly increased increase increasing increasingly decreased decrease decreasing enlarging
improved improvement improving worsened worsening worse progressed progression progressive progressing regressed
regression regressing resolved resolution resolving reduced reduction reducing unchanged stable similar
persistent persists persist persisting more less larger smaller greater lesser change changes changed development
developed developing size extent conspicuous conspicuity slight slightly mild mildly minimal minimally marked
markedly moderate moderately subtle partial partially complete completely near nearly overall now previously
prior baseline added interval interim from to in of than since approximately""".split())
MEASURE = re.compile(r"^(\d+([.,]\d+)?(mm|cm|ml|cc)?|mm|cm|ml|cc|x|×)$", re.I)
WEAK = set("""multiple several few numerous two three small large tiny focal diffuse scattered additional signal
mapping sequence imaging""".split())
# A finding needs a head noun: a word that ends >= HEAD_MIN of the multi-word member strings it occurs in.
# Words seen < HEAD_N times count as heads.
HEAD_MIN, HEAD_N, HEAD_ALWAYS = 0.3, 20, {"mass"}
CLOSE, NONE_TEXT = "No other interval change.", "No interval change."


def tokens(s):
    """RadGraph tokens -> words: rejoin 'non - enhancing', 'T2 / FLAIR', '5 . 5'; keep acronyms and T2-style case."""
    s = re.sub(r"\s+([-/])\s+", r"\1", s)
    s = re.sub(r"(\d)\s+\.\s+(\d)", r"\1.\2", s)
    s = re.sub(r"\s+'\s+s\b", "'s", s)
    return [w if (len(w) > 1 and w.isupper()) or any(c.isdigit() for c in w) else w.lower() for w in s.split()]


def head_words(units):
    """-> words that cannot head a finding (see HEAD_MIN)."""
    fin, n = Counter(), Counter()
    for m in units.members:
        for s in m:
            w = [x.lower() for x in tokens(s)]
            if len(w) > 1:
                fin[w[-1]] += 1
                n.update(w)
    return {w for w in n if n[w] >= HEAD_N and fin[w] / n[w] < HEAD_MIN} - HEAD_ALWAYS


def phrase(members, anatomy, nonhead=frozenset()):
    """-> '<finding> (<anatomy>)', or None when no finding word is left."""
    words = []
    for m in members:
        for w in tokens(m):
            if w.lower() not in STOP and not MEASURE.match(w) and w not in words:
                words.append(w)
    if not any(re.search(r"[a-z]", w, re.I) and w.lower() not in WEAK | nonhead for w in words):
        return None
    anat = []
    for a in anatomy:
        a = " ".join(w for w in tokens(a) if w.lower() not in STOP)
        if a and a not in anat and not set(a.split()) <= set(words):
            anat.append(a)
    return " ".join(words) + (f" ({', '.join(anat)})" if anat else "")


def target(lines):
    """lines: [(status, phrase)] in report order -> grouped text."""
    by = {s: [] for s in ECHANGE}
    for s, p in lines:
        if p not in by[s]:
            by[s].append(p)
    # drop a finding contained in another of the same status ('edema' next to 'surrounding edema (cerebellar)')
    bag = lambda x: set(re.findall(r"[\w'-]+", x.lower()))
    by = {s: [p for p in ps if not any(q != p and bag(p) < bag(q) for q in ps)] for s, ps in by.items()}
    parts = [f"{s}: {'; '.join(by[s])}." for s in ECHANGE if by[s]]
    return " ".join(parts + [CLOSE]) if parts else NONE_TEXT


def reverse_target(text):
    """Target for the prior<->current swapped pair."""
    if text == NONE_TEXT:
        return text
    segs = dict(re.findall(r"(NEW|WORSENED|IMPROVED|RESOLVED): (.*?\.)(?= [A-Z]+:| No other)", text))
    return " ".join([f"{s}: {segs[EREVERSE[s]]}" for s in ECHANGE if EREVERSE[s] in segs] + [CLOSE])


def build_text(units, keys, nonhead):
    p = np.stack(units.probs.values).astype(np.float64)
    st = np.array(STATUS)[p.argmax(1)]
    keep = np.isin(st, ECHANGE) & (p.max(1) >= P_MIN)
    u = units.assign(status=st, change=keep)
    u["phrase"] = [phrase(m, a, nonhead) if k else None for m, a, k in zip(u.members, u.anatomy, keep)]
    rows = []
    for key, d in u.sort_values(keys + ["unit_ix"]).groupby(keys, sort=False):
        c = d[d.change]
        lines = [(s, ph) for s, ph in zip(c.status, c.phrase) if ph is not None]
        rows.append(dict(zip(keys, key)) | {
            "ent_text": target(lines), "n_change_units": len(c), "n_dropped": int(c.phrase.isna().sum()),
            "unit_statuses": ",".join(s for s in ECHANGE if s in set(c.status)),
            "text_statuses": ",".join(s for s in ECHANGE if s in {s for s, _ in lines})})
    return pd.DataFrame(rows)


def entity_text(a):
    us, ub = pd.read_parquet(F / "s4gen_units.parquet"), pd.read_parquet(F / "bind_units.parquet")
    nonhead = head_words(pd.concat([us, ub]))
    s4 = build_text(us, ["study_uid1", "study_uid2"], nonhead).assign(source="MR-RATE", site="MR-RATE")
    bd = build_text(ub, ["site", "study_uid1", "study_uid2"], nonhead).assign(source="BIND")
    df = pd.concat([s4, bd], ignore_index=True).rename(columns={"study_uid1": "study1", "study_uid2": "study2"})
    df["ent_ok"] = df.unit_statuses == df.text_statuses
    df = df[["source", "site", "study1", "study2", "ent_text", "ent_ok", "n_change_units", "n_dropped",
             "unit_statuses", "text_statuses"]]
    assert not df.duplicated(["source", "site", "study1", "study2"]).any()
    df.to_parquet(OUT / "entity_text.parquet", index=False)
    pairs = pd.read_parquet(OUT / "pairs.parquet")
    pairs["site"] = pairs.site.where(pairs.source != "MR-RATE", "MR-RATE")       # NaN there
    j = pairs[pairs.has_lent].merge(df, on=["source", "site", "study1", "study2"], how="left")
    stats = {"pairs_with_units": df.groupby("source").size().to_dict(), "n_nonhead_words": len(nonhead),
             "has_lent_pairs_joined": f"{j.ent_text.notna().sum()} / {len(j)}"}
    for src, d in df.groupby("source"):
        nl = d.ent_text.str.count(r"; ") + d.ent_text.str.count(r"\b(NEW|WORSENED|IMPROVED|RESOLVED):")
        w = d.ent_text.str.split().map(len)
        stats[src] = {"no_change_share": round(float((d.ent_text == NONE_TEXT).mean()), 3),
                      "findings_per_pair_q50_q90_max": [int(nl.quantile(.5)), int(nl.quantile(.9)), int(nl.max())],
                      "words_q50_q90_q99_max": [int(w.quantile(q)) for q in (.5, .9, .99)] + [int(w.max())],
                      "change_units": int(d.n_change_units.sum()), "dropped_contentfree": int(d.n_dropped.sum()),
                      "ent_ok_share": round(float(d.ent_ok.mean()), 3),
                      "status_presence": {s: round(float(d.text_statuses.str.contains(s).mean()), 3) for s in ECHANGE}}
    phr = Counter(p for t in df[df.source == "MR-RATE"].ent_text for p in re.split(r"[;.]\s*|: ", t)
                  if p and p not in ECHANGE and not p.startswith("No "))
    stats["top_mrrate_findings"] = phr.most_common(25)
    json.dump(stats, open(OUT / "entity_text_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "top_mrrate_findings"}, indent=1))
    s4r = pd.read_csv(S4 / "main_extended.csv", usecols=["study_uid1", "study_uid2", "generated_report"])
    ex = df[df.source == "MR-RATE"].sample(20, random_state=0).merge(
        s4r, left_on=["study1", "study2"], right_on=["study_uid1", "study_uid2"])
    with open(OUT / "entity_text_review.md", "w") as f:          # MR-RATE examples only; BIND report text stays local
        f.write("# Stage-1 entity-change targets: 20 random MR-RATE pairs (seed 0)\n\nSource = generated_report "
                "(the Rad-Delta labelling input). Built by build_stage1.py entity_text.\n\n")
        for i, r in enumerate(ex.itertuples(), 1):
            f.write(f"## {i}. {r.study1[-8:]} / {r.study2[-8:]}\n\n**Target:** {r.ent_text}"
                    f"{'' if r.ent_ok else ' *(ent_ok False: excluded)*'}\n\n"
                    f"<details><summary>source report</summary>\n\n{r.generated_report}\n\n</details>\n\n")


# ================================================================== tasks
# Single-study tasks for Stage 1-D: boxes (aref / gcap / caref on ATLAS, BraTS-MEN, BraTS-MET, ISLES-22) and pathology
# (pcls, 29 labels, MR-RATE). Boxes = 6 integers 0-100, percent of each NeuroVFM grid axis.
# val = crc32(subject) % 10 == 0 (subject-grouped).
PATHOLOGY_PROMPT = (
    "Your job is to state which pathologies out of the following list are present in the image. "
    "Pathology List: Arachnoid cyst, Cavernous hemangioma, Cerebellar degeneration, "
    "Cerebral atrophy, Cerebral edema, Cerebral hemorrhage, Cerebral infarction, "
    "Chiari malformation, Choroid plexus cyst, Cyst of pineal gland, "
    "Demyelinating disease of central nervous system, Empty sella syndrome, Encephalomalacia, "
    "Glioma, Gliosis, Intracranial aneurysm, Intracranial meningioma, Lacunar infarct, "
    "Lipoma of brain, Mega cisterna magna, Metastatic malignant neoplasm to brain, "
    "Pituitary adenoma, Rathke's pouch cyst, Schwannoma, Silent micro-hemorrhage of brain, "
    "Structure of cave of septum pellucidum, Subdural intracranial hemorrhage, "
    "Ventriculomegaly, Watershed infarct. "
    "Output only pathologies from this list, in alphabetical order, separated by a comma and a "
    "space. If more than one is present, list them all. No additional words or statements. "
    "Which pathologies from the list do you observe in the scans?"
)


def subject(r):
    p = next(r[m] for m in MODS if isinstance(r[m], str))
    corpus, b = p.split("/")[-3], os.path.basename(p)
    m = re.match(r"(BraTS-[A-Z]+-\d+)", b)
    return corpus, f"{corpus}/{m.group(1) if m else b.split('_')[0].split('.')[0]}"


def tasks(a):
    d = pd.read_csv(S1 / "unified_dataframe_norm.csv")
    d = d[d.task.isin(["aref", "gcap", "caref", "pcls"])].reset_index(drop=True)
    box = d.task != "pcls"
    fix = lambda s: s.str.replace("{pathology} infarct in", "{finding} in", regex=False).str.replace(
        "There is a metastases in", "There is a metastasis in", regex=False)
    d.loc[box, "prompt"] = fix(d.prompt[box])
    d.loc[box, "caption"] = fix(d.caption[box])
    d.loc[~box, "prompt"] = PATHOLOGY_PROMPT
    cs = [subject(r) for _, r in d.iterrows()]
    d["corpus"] = [c if b else "MR-RATE" for (c, _), b in zip(cs, box)]
    d["subject"] = [s for _, s in cs]
    pid = pd.read_csv(MRRATE / "splits.csv").set_index("study_uid").patient_uid
    sid = d.T1w[~box].str.split("/").str[-2]
    assert sid.isin(pid.index).all(), f"{(~sid.isin(pid.index)).sum()} pcls studies without a patient_uid"
    d.loc[~box, "subject"] = "MR-RATE/" + sid.map(pid).astype(str)
    d["split"] = ["val" if zlib.crc32(s.encode()) % 10 == 0 else "train" for s in d.subject]
    out = d.rename(columns={"caption": "target", "BoundingBox": "box"})[
        ["task", "prompt", "target", *MODS, "subject", "corpus", "with_lesion", "box", "split"]]
    assert not out.prompt.str.contains("infarct in the {anatomical", regex=False).any()
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "tasks.parquet", index=False)
    print(out.groupby(["task", "split"]).size().unstack(), "\n")
    print(out.groupby("task").subject.nunique(), "\n")
    print(out.groupby(["corpus", "split"]).subject.nunique().unstack())


# ================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("coverage")
    sub.add_parser("dupes")
    sub.add_parser("probe_pairs")
    p = sub.add_parser("label")
    p.add_argument("--src", choices=list(SRC), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    p.add_argument("--ckpt", default=os.environ.get("RADDELTA_CKPT"), help="Rad-Δ student (paper: distill_joint_seed0.pt)")
    sub.add_parser("combine").add_argument("--src", choices=list(SRC), required=True)
    sub.add_parser("pairs")
    sub.add_parser("entity_text")
    sub.add_parser("tasks")
    a = ap.parse_args()
    {"coverage": coverage, "dupes": dupes, "probe_pairs": probe_pairs, "label": label, "combine": combine, "pairs": build_pairs, "entity_text": entity_text,
     "tasks": tasks}[a.cmd](a)


if __name__ == "__main__":
    main()
