#!/usr/bin/env python3
"""
Stage-2 targets and pair table. CPU only; `rewrite --submit` / `--resume` are PAID Anthropic API calls.

Prompts in report_rewrite_prompt.py (privacy-edited release copy, docs/PROMPT_CHANGES.md: running it will not
reproduce the shipped targets exactly).

    python build_stage2.py rewrite --set bind   --model claude-opus-5-5   --kinds prior   --tag _full   # dry run (scoring env)
    python build_stage2.py rewrite --set bind   --model claude-sonnet-5-5 --kinds current --tag _full
    python build_stage2.py rewrite --set mrrate --model claude-opus-5-5   --kinds prior   --tag _full
    python build_stage2.py rewrite --set mrrate --model claude-sonnet-5-5 --kinds current --tag _full
        (add --submit to send, --resume <batch id> to collect; a request whose reply misses a case is flagged)
    python build_stage2.py rewrite --set bind --model claude-sonnet-5-5 --kinds current --batching 10 --tag _retry --keys K
    python build_stage2.py merge --into rewrite/batch_bind_claude-sonnet-5-5_nothink_full.jsonl --retry rewrite/..._retry.jsonl
    python build_stage2.py pairs                                    (env BrainDiff)  -> stage2/pairs.parquet

rewrite: per report pair, prior -> Baseline edit (change language removed; the model's prior-report input), BIND
current -> Followup edit (target), MR-RATE generated_report -> Restyle into the BIND follow-up style (one target
style). BIND reports are sent from FINDINGS on. Pairs with a session lacking an ok / partial registered modality
(bind_qc_split.py qc) are not sent.

pairs: per rewritten pair, prior_edit / target / originals; st_NEW .. st_RESOLVED = Rad-Δ presence (noisy-OR >= 0.5)
of the ORIGINAL current report; group = bit pattern of those statuses (the counterfactual partner must differ in it);
T1w_1 .. FLAIR_2 = ok / partial series only; split = BIND by bind_patient_split.csv, MR-RATE all train. Pairs without
a series present at both timepoints are dropped.

Env: BRAINDIFF_DATA (BRAIN_DIFF_S4), BRAINDIFF_BUILD (rewrite/, qc/, splits/, findings/, stage2/), BIND_TABLES.
ANTHROPIC_API_KEY is read by the anthropic client only when sending.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bind_preprocess import PAIRED_CSV, findings_onward
from report_rewrite_prompt import ReportEditPrompt

BUILD = Path(os.environ.get("BRAINDIFF_BUILD", "build"))
S4 = Path(os.environ.get("BRAINDIFF_DATA", "data")) / "BRAIN_DIFF_S4" / "main_extended.csv"
RW = BUILD / "rewrite"
QC = BUILD / "qc" / "session_qc.csv"


# ================================================================== rewrite
PROMPT = {("BIND", "prior"): "baseline", ("BIND", "current"): "followup",
          ("MR-RATE", "prior"): "baseline", ("MR-RATE", "current"): "restyle"}
# 16 cases per request can exceed 8,192 output tokens.
MAX_TOKENS = 16384
# Neither model accepts disabled thinking: adaptive + low effort for both.
LOW_THINK = {"claude-sonnet-5-5": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
             "claude-opus-5-5": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}}


def cases(which):
    """-> DataFrame: key (site/study1/study2/kind), source, site, study_uid1, study_uid2, kind, original."""
    q = pd.read_csv(QC, dtype=str)
    q = q[q.status.isin(["ok", "partial"])]
    if which == "bind":
        df = pd.read_csv(PAIRED_CSV, usecols=["site", "study_uid1", "study_uid2", "report1", "report2"], dtype=str)
        ok = set(q[q.source == "BIND"][["site", "session"]].itertuples(index=False, name=None))
        df = df[[(s, a) in ok and (s, b) in ok for s, a, b in zip(df.site, df.study_uid1, df.study_uid2)]]
        for c in ("report1", "report2"):
            df[c] = df[c].map(lambda r: findings_onward(r) or r if isinstance(r, str) else r)
        df = df.assign(source="BIND")
    else:
        df = pd.read_csv(S4, usecols=["study_uid1", "study_uid2", "report1", "generated_report"], dtype=str)
        df = df.rename(columns={"generated_report": "report2"}).assign(site="MR-RATE", source="MR-RATE")
        ok = set(q[q.source != "BIND"].session)
        df = df[[a in ok and b in ok for a, b in zip(df.study_uid1, df.study_uid2)]]
    rows = [{"source": r.source, "site": r.site, "study_uid1": r.study_uid1, "study_uid2": r.study_uid2, "kind": k,
             "original": (r.report1 if k == "prior" else r.report2) or ""}
            for r in df.itertuples() for k in ("prior", "current")]
    out = pd.DataFrame(rows)
    out["key"] = out.site + "/" + out.study_uid1 + "/" + out.study_uid2 + "/" + out.kind
    return out


def build_requests(df, model, batching, kinds=("prior", "current"), ttl="5m"):
    reqs, index = [], {}
    for (source, kind), pk in PROMPT.items():
        if kind not in kinds:
            continue
        sub = df[(df.source == source) & (df.kind == kind)].reset_index(drop=True)
        prompt = ReportEditPrompt(pk)
        for j in range(0, len(sub), batching):
            chunk = sub.iloc[j:j + batching]
            ids = list(range(1, len(chunk) + 1))
            system, user = prompt.generate(chunk.original.tolist(), ids)
            cid = f"{source}-{pk}-{j}"
            index[cid] = dict(zip(ids, chunk.key))
            reqs.append({"custom_id": cid, "params": {
                "model": model, "max_tokens": MAX_TOKENS, **LOW_THINK.get(model, {"thinking": {"type": "disabled"}}),
                "system": [{"type": "text", "text": system,
                            "cache_control": {"type": "ephemeral"} | ({"ttl": ttl} if ttl == "1h" else {})}],
                "messages": [{"role": "user", "content": user}]}})
    return reqs, index


def token_report(reqs, df):
    """o200k_base as a proxy for Claude's tokenizer (Claude's counts run somewhat higher)."""
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    sys_tok = sum(len(enc.encode(r["params"]["system"][0]["text"])) for r in reqs)
    usr_tok = sum(len(enc.encode(r["params"]["messages"][0]["content"])) for r in reqs)
    out_tok = int(0.6 * sum(len(enc.encode(t)) for t in df.original))     # edits keep ~55% of the words
    print(f"[tokens] {len(reqs):,} requests x {len(df):,} reports")
    print(f"[tokens] system {sys_tok / 1e6:.2f}M (cacheable), user {usr_tok / 1e6:.2f}M, output ~{out_tok / 1e6:.2f}M")


def parse(text, ids):
    got = {}
    for m in re.finditer(r"^\s*Report\s*(\d+)\s*:\s*(.*)$", text, re.M):
        n = int(m.group(1))
        if n in ids and n not in got:
            got[n] = m.group(2).strip()
    return got


def rewrite(a):
    RW.mkdir(parents=True, exist_ok=True)
    tag = f"batch_{a.set}_{a.model}_nothink{a.tag}"
    kinds = tuple(a.kinds.split(","))
    df = cases(a.set)
    df = df[df.kind.isin(kinds)]
    if a.keys:
        df = df[df.key.isin(open(a.keys).read().split())]
    reqs, index = build_requests(df, a.model, a.batching, kinds, a.ttl)
    with open(RW / f"{tag}_requests.jsonl", "w") as f:
        for r in reqs:
            f.write(json.dumps(r) + "\n")
    json.dump(index, open(RW / f"{tag}_index.json", "w"))
    print(f"[dry] wrote {RW / f'{tag}_requests.jsonl'}")
    token_report(reqs, df)
    if not (a.submit or a.resume):
        print("[dry] nothing sent")
        return
    import anthropic
    client = anthropic.Anthropic()
    batch_id = a.resume
    if a.submit:
        batch_id = client.messages.batches.create(requests=reqs).id
        print(f"[submit] batch {batch_id}", flush=True)
        (RW / f"{tag}_batch_id.txt").write_text(batch_id)
    while True:
        b = client.messages.batches.retrieve(batch_id)
        print(f"[poll] {b.processing_status} {b.request_counts}", flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(60)
    by_key, usage, bad, raw = dict(zip(df.key, df.to_dict("records"))), [], [], []
    with open(RW / f"{tag}.jsonl", "w") as f:
        for res in client.messages.batches.results(batch_id):
            ids = {int(k): v for k, v in index[res.custom_id].items()}
            if res.result.type != "succeeded":
                bad += [(res.custom_id, res.result.type)]
                continue
            msg = res.result.message
            text = "".join(c.text for c in msg.content if c.type == "text")
            usage.append(msg.usage.model_dump() | {"prompt": res.custom_id.rsplit("-", 1)[0]})
            raw.append({"custom_id": res.custom_id, "stop_reason": msg.stop_reason, "text": text})
            got = parse(text, ids)
            for n, key in ids.items():
                r = by_key[key]
                if n not in got:
                    bad.append((res.custom_id, f"missing Report {n}"))
                f.write(json.dumps({k: r[k] for k in ("source", "site", "study_uid1", "study_uid2", "kind", "original")}
                                   | {"rewrite": got.get(n, ""), "flag": "" if n in got else "missing"}) + "\n")
    json.dump(raw, open(RW / f"{tag}_raw.json", "w"))
    tot = pd.DataFrame(usage).sum(numeric_only=True).to_dict() if usage else {}
    print(f"[done] {len(usage)} requests ok, {len(bad)} problems {bad[:5]}; usage {tot}")
    if usage:
        u = pd.DataFrame(usage)
        print("[cache] requests that read the system prompt from the cache:",
              (u.cache_read_input_tokens > 0).groupby(u.prompt).agg(["sum", "count"]).to_dict("index"))


def merge(a):
    """Replace the flagged rows of a run with the rows of its --keys resend; keeps the pre-merge file as
    *_preretry.jsonl."""
    key = lambda r: (r["site"], r["study_uid1"], r["study_uid2"], r["kind"])
    rows = [json.loads(l) for l in open(a.into)]
    new = {key(r): r for r in map(json.loads, open(a.retry))}
    assert all(r["flag"] == "" for r in new.values()), "the resend itself has missing reports"
    pre = a.into.with_name(a.into.name.replace(".jsonl", "_preretry.jsonl"))
    os.replace(a.into, pre)
    merged = [new[key(r)] if r["flag"] and key(r) in new else r for r in rows]
    with open(a.into, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    print(f"merged {sum(1 for r in rows if r['flag'] and key(r) in new)} rows; still missing {sum(r['flag'] != '' for r in merged)}")


# ================================================================== pairs
MODS = {"t1w": "T1w", "t1ce": "T1ce", "t2w": "T2w", "flair": "FLAIR"}
STATUS = ["NEW", "WORSENED", "IMPROVED", "RESOLVED"]
COLS = [0, 1, 2, 4]                 # Rad-Δ probs order: NEW, WORSENED, IMPROVED, STABLE, RESOLVED


def rewrites(src, model, kind):
    rows = [json.loads(l) for l in open(RW / f"batch_{src}_claude-{model}-5-5_nothink_full.jsonl")]
    d = pd.DataFrame([r for r in rows if r["kind"] == kind])
    assert (d.flag == "").all(), f"{src} {kind}: {(d.flag != '').sum()} rewrites missing"
    return d.rename(columns={"study_uid1": "study1", "study_uid2": "study2"})


def statuses(units, keys):
    """units parquet -> {(study1, study2[, site]): presence [4]} by noisy-OR over the report's units."""
    p = np.clip(np.stack(units.probs.values)[:, COLS], 0, 1 - 1e-12)
    u = pd.DataFrame(np.log1p(-p), columns=STATUS)
    for k in keys:
        u[k] = units[k].values
    return (1 - np.exp(u.groupby(keys)[STATUS].sum())) >= 0.5


def pairs(a):
    parts = []
    for src, sname in (("bind", "BIND"), ("mrrate", "MR-RATE")):
        pr, cu = rewrites(src, "opus", "prior"), rewrites(src, "sonnet", "current")
        k = ["site", "study1", "study2"]
        d = pr[k + ["original", "rewrite"]].rename(columns={"original": "prior_orig", "rewrite": "prior_edit"}).merge(
            cu[k + ["original", "rewrite"]].rename(columns={"original": "current_orig", "rewrite": "target"}),
            on=k, validate="1:1")
        assert len(d) == len(pr) == len(cu), (src, len(d), len(pr), len(cu))
        parts.append(d.assign(source=sname))
    p = pd.concat(parts, ignore_index=True)
    b = pd.read_csv(PAIRED_CSV, usecols=["site", "patient_uid", "study_uid1", "study_uid2"], dtype=str).rename(
        columns={"study_uid1": "study1", "study_uid2": "study2"})
    m = pd.read_csv(S4, usecols=["patient_uid", "study_uid1", "study_uid2"], dtype=str).rename(
        columns={"study_uid1": "study1", "study_uid2": "study2"}).assign(site="MR-RATE")
    pat = pd.concat([b.assign(patient=b.site + ":" + b.patient_uid), m.assign(patient="MR-RATE:" + m.patient_uid)])
    p = p.merge(pat[["site", "study1", "study2", "patient"]].drop_duplicates(["site", "study1", "study2"]),
                on=["site", "study1", "study2"], how="left", validate="m:1")
    assert p.patient.notna().all(), f"{p.patient.isna().sum()} pairs without a patient"
    sp = pd.read_csv(BUILD / "splits" / "bind_patient_split.csv").set_index("patient").split
    p["split"] = np.where(p.source == "BIND", p.patient.map(sp), "train")
    assert p.split.notna().all(), f"{p.split.isna().sum()} BIND pairs without a split"
    bs = statuses(pd.read_parquet(BUILD / "findings" / "bind_units.parquet"), ["site", "study_uid1", "study_uid2"])
    ms = statuses(pd.read_parquet(BUILD / "findings" / "s4gen_units.parquet"), ["study_uid1", "study_uid2"])
    lab = np.zeros((len(p), 4), bool)          # a report with no units has no status
    for i, r in enumerate(p.itertuples()):
        key = (r.site, r.study1, r.study2) if r.source == "BIND" else (r.study1, r.study2)
        s = bs if r.source == "BIND" else ms
        if key in s.index:
            lab[i] = s.loc[key].values
    for j, s in enumerate(STATUS):
        p[f"st_{s}"] = lab[:, j]
    p["any_change"] = lab.any(1)
    p["group"] = (lab * (1 << np.arange(4))).sum(1)
    q = pd.read_csv(QC, dtype=str)
    q = q[q.status.isin(["ok", "partial"])]
    path = {(r.site, r.session, r.modality): r.path for r in q.itertuples()}
    for mod, M in MODS.items():
        for t in (1, 2):
            p[f"{M}_{t}"] = [path.get((s, x if t == 1 else y, mod)) for s, x, y in zip(p.site, p.study1, p.study2)]
    p["n_common"] = sum((p[f"{M}_1"].notna() & p[f"{M}_2"].notna()).astype(int) for M in MODS.values())
    print(f"{(p.n_common == 0).sum()} of {len(p)} pairs share no usable series -> dropped")
    p = p[p.n_common > 0].reset_index(drop=True)
    p["pair_id"] = np.arange(len(p))
    out = BUILD / "stage2" / "pairs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(out)
    print(p.groupby(["source", "split"]).agg(pairs=("pair_id", "size"), patients=("patient", "nunique"),
                                             any_change=("any_change", "mean"), n_common=("n_common", "mean")).round(3))
    print("->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("rewrite")
    p.add_argument("--set", choices=["bind", "mrrate"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--batching", type=int, default=16)
    p.add_argument("--kinds", default="prior,current", help="report kinds to send (paper: prior with Opus, current with Sonnet)")
    p.add_argument("--tag", default="", help="suffix for output files (paper: _full; resend: _retry)")
    p.add_argument("--keys", default=None, help="file of case keys (site/study1/study2/kind) to send only those")
    p.add_argument("--ttl", choices=["5m", "1h"], default="5m", help="system-prompt cache TTL")
    p.add_argument("--submit", action="store_true", help="PAID: create the batch")
    p.add_argument("--resume", default=None, help="batch id to poll and parse")
    p = sub.add_parser("merge")
    p.add_argument("--into", type=Path, required=True)
    p.add_argument("--retry", type=Path, required=True)
    sub.add_parser("pairs")
    a = ap.parse_args()
    {"rewrite": rewrite, "merge": merge, "pairs": pairs}[a.cmd](a)


if __name__ == "__main__":
    main()
