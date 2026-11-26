#!/usr/bin/env python3
"""
Paper Table 2, chest X-ray rows: MIMIC-CXR prior -> current pairs, MAIRA-2 and Libra greedy generation.
Runs in a transformers 4.51.3 env; Libra code from $LIBRA_ROOT (commit 0d6c56a).

sample         current = each study of --split with a frontal image and FINDINGS or IMPRESSION; prior = same patient's most
               recent earlier study with a frontal image. -> out/cxr_arm/sample_mimic<tag>.parquet + the image list
findings_sets  FINDINGS-only sets (both generators write FINDINGS only) -> sample_mimic_f.parquet, sample_mimic_valf.parquet
libra_context  Libra's official prompt (arXiv 2411.19378, App. C) + Indication / History / Comparison / Technique lines; a
               report with none gets the first sentence alone [assumed]. -> libra_context.parquet; existing rows kept
generate       maira2 (microsoft/maira-2, --maira_lora = a maira2_finetune.py epoch dir) or libra (X-iZhang/libra-v1.0-7b)
               -> out/cxr_arm/gen_<model><tag>_shard<k>.jsonl {"patient", "report"}; resumable per patient

    python baselines/cxr/cxr_generate.py sample --root <MIMIC-CXR-JPG 2.1.0> --reports <mimic-cxr-reports.zip> [--split validate --tag _valall]
    python baselines/cxr/cxr_generate.py findings_sets
    python baselines/cxr/cxr_generate.py libra_context sample_mimic_f.parquet sample_mimic_valf.parquet --reports <zip>
    CUDA_VISIBLE_DEVICES=0 python baselines/cxr/cxr_generate.py generate --model libra --images <root> --sample sample_mimic_f.parquet --tag _mimicoff --official_prompt
    CUDA_VISIBLE_DEVICES=0 python baselines/cxr/cxr_generate.py generate --model maira2 --images <root> --sample sample_mimic_f.parquet --tag _sfM0_test --no_lateral --max_new 340
"""
import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

WORK = Path(os.environ.get("CXR_WORK", "."))
OUT = WORK / "out" / "cxr_arm"
LIBRA_ROOT = Path(os.environ.get("LIBRA_ROOT", "baselines/Libra"))
SMOKE_URL = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
LIBRA_PROMPT = "Provide a detailed description of the findings in the radiology image. Following clinical context: {}"
SECTIONS = ("FINDINGS", "IMPRESSION", "INDICATION", "HISTORY", "TECHNIQUE", "COMPARISON")
HEAD = "Provide a detailed description of the findings in the radiology image."
FIELDS = (("indication", "Indication"), ("history", "History"), ("comparison", "Comparison"), ("technique", "Technique"))


def sections(text):
    """Report text -> {SECTION: text} for the usual MIMIC-CXR headers (upper case, colon)."""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*([A-Z][A-Z /]+):\s*(.*)", line)
        if m and m.group(1).strip() in SECTIONS:
            cur = m.group(1).strip(); out[cur] = m.group(2).strip()
        elif cur:
            out[cur] = (out[cur] + " " + line.strip()).strip()
    return {k: " ".join(v.split()) for k, v in out.items()}


def fi(c):
    return " ".join(x for x in (f"FINDINGS: {c['FINDINGS']}" if c.get("FINDINGS") else "",
                                f"IMPRESSION: {c['IMPRESSION']}" if c.get("IMPRESSION") else "") if x)


# ---- pairs ----
def sample(a):
    R = Path(a.root)
    meta = pd.read_csv(R / "mimic-cxr-2.0.0-metadata.csv.gz").merge(pd.read_csv(R / "mimic-cxr-2.0.0-split.csv.gz")[["dicom_id", "split"]], on="dicom_id")
    meta["when"] = meta.StudyDate.astype(str) + meta.StudyTime.astype(float).map(lambda t: f"{t:010.3f}")
    meta["path"] = meta.apply(lambda r: f"files/p{str(r.subject_id)[:2]}/p{r.subject_id}/s{r.study_id}/{r.dicom_id}.jpg", axis=1)
    rank = {"PA": 0, "AP": 1}
    fr = meta[meta.ViewPosition.isin(rank)].assign(r=lambda d: d.ViewPosition.map(rank)).sort_values(["study_id", "r"])
    frontal = fr.groupby("study_id").first()
    lateral = meta[meta.ViewPosition.isin(["LATERAL", "LL"])].groupby("study_id").path.first()
    studies = frontal.reset_index()[["subject_id", "study_id", "when", "split", "path"]].sort_values(["subject_id", "when"])
    studies["prior_study"] = studies.groupby("subject_id").study_id.shift(1)
    studies["prior_path"] = studies.groupby("subject_id").path.shift(1)
    cur = studies[(studies.split == a.split) & studies.prior_study.notna()].copy()
    zf = zipfile.ZipFile(a.reports)
    names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".txt")}
    rep = lambda sid: sections(zf.read(names[f"s{int(sid)}"]).decode("utf-8", "replace")) if f"s{int(sid)}" in names else {}  # noqa: E731
    rows = []
    for k, r in enumerate(cur.itertuples()):
        c, p = rep(r.study_id), rep(r.prior_study)
        ref = fi(c)
        if not ref:
            continue
        rows.append({"draw_order": k, "patient": f"p{r.subject_id}_s{r.study_id}", "prior_frontal": r.prior_path, "current_frontal": r.path,
                     "current_lateral": lateral.get(r.study_id), "prior_report": fi(p),
                     "indication": c.get("INDICATION") or c.get("HISTORY") or "", "technique": c.get("TECHNIQUE", ""),
                     "comparison": c.get("COMPARISON", ""), "ref": ref})
    D = pd.DataFrame(rows).reset_index(drop=True)
    if a.n:
        D = D.sample(n=min(a.n, len(D)), random_state=0).reset_index(drop=True)
    D["draw_order"] = range(len(D))
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT / f"sample_mimic{a.tag}.parquet")
    need = [(r.draw_order, r.patient, role, getattr(r, col)) for r in D.itertuples()
            for role, col in (("prior", "prior_frontal"), ("current", "current_frontal"), ("current_lateral", "current_lateral"))
            if isinstance(getattr(r, col), str)]
    pd.DataFrame(need, columns=["pair", "study", "role", "path"]).drop_duplicates("path").to_csv(OUT / f"mimic_cxr_needed_images{a.tag}.csv", index=False)
    print(f"{a.split} studies with a frontal prior and a report: {len(D)}; images to fetch: {len(set(p for *_, p in need))}")


def findings_sets(a):
    F = re.compile(r"^FINDINGS: (.*?)(?: IMPRESSION: .*)?$", re.S)
    for src, dst in (("sample_mimic.parquet", "sample_mimic_f.parquet"), ("sample_mimic_valall.parquet", "sample_mimic_valf.parquet")):
        S = pd.read_parquet(OUT / src)
        S = S[S.ref.str.startswith("FINDINGS: ")].copy()
        S["ref"] = S.ref.map(lambda t: F.match(t).group(1).strip())
        S = S[S.ref != ""].reset_index(drop=True)
        S.to_parquet(OUT / dst)
        print(f"{dst}: {len(S)} pairs, {S.patient.str.split('_').str[0].nunique()} patients")


def libra_query(text):
    """Report text -> Libra's official prompt (module docstring)."""
    sys.path.insert(0, str(LIBRA_ROOT / "scripts" / "mimic-cxr"))
    import section_parser as sp
    secs, names, _ = sp.section_text(text)
    c = {}
    for s, n in zip(secs, sp.normalize_section_names(names)):
        s = " ".join(s.split()).strip(" :")
        if n in dict(FIELDS) and s and n not in c:
            c[n] = s
    lines = [f"{lab}: {c[k]}" for k, lab in FIELDS if c.get(k)]
    return HEAD + (" Following clinical context:\n" + "\n".join(lines) if lines else "")


def libra_context(a):
    zf = zipfile.ZipFile(a.reports)
    names = {Path(n).stem: n for n in zf.namelist() if n.endswith(".txt")}
    f = OUT / "libra_context.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["patient", "query"])
    rows = []
    for s in a.samples:
        for p in pd.read_parquet(OUT / s).patient:
            if p in set(have.patient) or any(r["patient"] == p for r in rows[-1:]):
                continue
            sid = re.search(r"_s(\d+)$", p).group(1)
            rows.append({"patient": p, "query": libra_query(zf.read(names[f"s{sid}"]).decode("utf-8", "replace"))})
    D = pd.concat([have, pd.DataFrame(rows)]).drop_duplicates("patient").reset_index(drop=True)
    D.to_parquet(f)
    print(f"{len(D)} patients ({len(rows)} new); with a context block {D['query'].str.contains('Following').mean():.0%}")


# ---- generation ----
def load(model, maira_lora=None):
    import torch
    if model == "maira2":
        from transformers import AutoModelForCausalLM, AutoProcessor
        m = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True, torch_dtype=torch.bfloat16)
        if maira_lora:   # fine-tuned arm: LLM LoRA merged + projector
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, maira_lora).merge_and_unload()
            if (Path(maira_lora) / "projector.pt").exists():
                m.multi_modal_projector.load_state_dict({k: v.to(torch.bfloat16) for k, v in torch.load(Path(maira_lora) / "projector.pt", map_location="cpu").items()})
        return m.eval().cuda(), AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)
    sys.path.insert(0, str(LIBRA_ROOT))
    from libra.mm_utils import get_model_name_from_path
    from libra.model.builder import load_pretrained_model
    path = "X-iZhang/libra-v1.0-7b"
    return load_pretrained_model(path, None, get_model_name_from_path(path)), None


def generate(model, M, P, cur, prior, row, max_new=300):
    import torch
    if model == "maira2":
        x = P.format_and_preprocess_reporting_input(
            current_frontal=cur, current_lateral=row.get("lateral_img"), prior_frontal=prior,
            indication=row["indication"] or None, technique=row["technique"] or None, comparison=row["comparison"] or None,
            prior_report=row["prior_report"] or None, return_tensors="pt", get_grounding=False).to("cuda")
        x["pixel_values"] = x["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            out = M.generate(**x, max_new_tokens=max_new, use_cache=True, do_sample=False)
        text = P.decode(out[0][x["input_ids"].shape[-1]:], skip_special_tokens=True).lstrip()
        # a stray <obj> / <box> tag makes MAIRA-2's parser assert: drop tags and box coordinates
        if re.search(r"</?(obj|box)>", text):
            print(f"[maira2] {row['patient']}: grounding tags stripped", file=sys.stderr, flush=True)
            text = re.sub(r"\s+", " ", re.sub(r"<box>.*?</box>|</?(obj|box)>|<[xy]\d+>", " ", text))
        return P.convert_output_to_plaintext_or_grounded_sequence(text)
    from libra.eval import libra_eval
    tmp = OUT / "_libra_tmp" / str(os.getpid()); tmp.mkdir(parents=True, exist_ok=True)   # per process: parallel jobs race
    files = []
    for name, im in (("cur", cur), ("prior", prior)):
        f = tmp / f"{row['patient']}_{name}.png"; im.save(f); files.append(str(f))
    q = row.get("libra_query") or LIBRA_PROMPT.format(row["indication"] or "None.")
    return libra_eval(libra_model=M, image_file=files, query=q, conv_mode="libra_v1", temperature=0.0, num_beams=1, max_new_tokens=512)


def run_generate(a):
    from PIL import Image
    M, P = load(a.model, a.maira_lora)
    if a.smoke:      # one public IU X-ray image as current and prior: code path only
        import requests
        im = Image.open(requests.get(SMOKE_URL, headers={"User-Agent": "MAIRA-2"}, stream=True).raw).convert("RGB")
        row = {"patient": "smoke", "indication": "Dyspnea.", "technique": "PA view of the chest.", "comparison": "Prior radiograph.",
               "prior_report": "FINDINGS: The lungs are clear. No pleural effusion. IMPRESSION: No acute cardiopulmonary process."}
        print(f"[smoke {a.model}]", generate(a.model, M, P, im, im, row))
        return
    D = pd.read_parquet(OUT / a.sample).iloc[a.shard::a.n_shards]
    if a.official_prompt:
        D = D.merge(pd.read_parquet(OUT / "libra_context.parquet").rename(columns={"query": "libra_query"}), on="patient", how="left")
        assert D.libra_query.notna().all(), "patients missing from libra_context.parquet"
    out = OUT / f"gen_{a.model}{a.tag}_shard{a.shard}.jsonl"
    done = {json.loads(l)["patient"] for l in open(out)} if out.exists() else set()
    root = Path(a.images)

    def img(rel):   # .jpg listed; a PNG download keeps the same paths otherwise
        q = root / rel
        return Image.open(q if q.exists() else q.with_suffix(".png")).convert("RGB")
    with open(out, "a") as f:
        for r in D.itertuples():
            if r.patient in done:
                continue
            row = r._asdict()
            cur, prior = img(r.current_frontal), img(r.prior_frontal)
            if r.current_lateral and a.model == "maira2" and not a.no_lateral:   # Libra never reads the lateral
                row["lateral_img"] = img(r.current_lateral)
            rep = generate(a.model, M, P, cur, prior, row, a.max_new)
            f.write(json.dumps({"patient": r.patient, "report": rep if isinstance(rep, str) else str(rep)}) + "\n"); f.flush()
    print(f"[{a.model}] shard {a.shard}: done")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--root", required=True, help="MIMIC-CXR-JPG 2.1.0 folder with the split / metadata csv.gz")
    s.add_argument("--reports", required=True, help="mimic-cxr-reports.zip (MIMIC-CXR 2.1.0)")
    s.add_argument("--split", default="test", help="MIMIC split of the CURRENT study")
    s.add_argument("--n", type=int, default=0, help="seeded random subset of pairs (0 = all)")
    s.add_argument("--tag", default="")
    sub.add_parser("findings_sets")
    c = sub.add_parser("libra_context")
    c.add_argument("samples", nargs="+"); c.add_argument("--reports", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", choices=["maira2", "libra"], required=True)
    g.add_argument("--images", help="MIMIC-CXR-JPG root (paths in the sample file are relative to it)")
    g.add_argument("--sample", default="sample_mimic_f.parquet", help="sample file in out/cxr_arm")
    g.add_argument("--shard", type=int, default=0); g.add_argument("--n_shards", type=int, default=1)
    g.add_argument("--tag", default="", help="output suffix: gen_<model><tag>_shard<k>.jsonl")
    g.add_argument("--maira_lora", help="maira2_finetune.py epoch dir (adapter + projector.pt)")
    g.add_argument("--no_lateral", action="store_true", help="MAIRA-2: never pass the current lateral (fine-tuned arms train without it)")
    g.add_argument("--max_new", type=int, default=300, help="MAIRA-2 max_new_tokens (model card 300; Table 2 / fine-tuned arms 340)")
    g.add_argument("--official_prompt", action="store_true", help="Libra: the paper's prompt (libra_context.parquet)")
    g.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    {"sample": sample, "findings_sets": findings_sets, "libra_context": libra_context, "generate": run_generate}[a.cmd](a)


if __name__ == "__main__":
    main()
