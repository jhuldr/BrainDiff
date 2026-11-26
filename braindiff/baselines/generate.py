#!/usr/bin/env python3
"""Table 2 baselines, run LOCALLY with vLLM: general VLMs write the follow-up report from the edited prior report + both
scans as images; NeuroVFM writes a current-scan report that its own Qwen3-14B merges with the prior report. Greedy, max
1,024 tokens, output-only repetition penalty 1.1 (out_rep_penalty.py; server flag
--logits-processors out_rep_penalty:OutputRepetitionPenalty).
  rows (one per Table 2 system; output dir = checkpoints/table2_baselines/<dir>):
    qwen3vl8b_8      Qwen3-VL-8B, 8 labelled slices per volume, image instruction        --model qwen3vl8b --img_instr --sl8_images    -> qwen3vl8b_imginstr_sl8img_rp110
    qwen3vl8b_48     Qwen3-VL-8B, 48 slices as three 16-slice montages             --model qwen3vl8b --img_instr --slices 48   -> qwen3vl8b_imginstr_s48_rp110
    medgemma27b_8    MedGemma-27B, 8 labelled slices per volume                          --model medgemma27b --sl8_images             -> medgemma27b_sl8img_rp110
    medgemma27b_48   MedGemma-27B, 48-slice montages                               --model medgemma27b --slices 48             -> medgemma27b_s48_rp110
    medgemma4b_vol   MedGemma-1.5-4B volume input, at most 85 slices per query     --model medgemma --vol 85                   -> medgemma_vol85_rp110
    qwen3vl8b_ft_sf  status-first fine-tuned Qwen3-VL-8B (finetune.py), served LoRA --model qwen3vl8b --img_instr --sl8_images --lora sf<k> -> qwen3vl8b_ft-sf<k>_imginstr_sl8img_rp110
    neurovfm14b      NeuroVFM -> its Qwen3-14B: step 1 `neurovfm_current`, step 2 (this row) = neurovfm_compare --model neurovfm-qwen3-14b --out neurovfm14b_rp110
  tools: render (slice / montage PNGs), vlm (any flag combination above), neurovfm_current, neurovfm_compare, sf_strip
    python generate.py render --split test external_test --sl8            (also --slices 48, --vol 85)
    python generate.py qwen3vl8b_8 --split test --ports 8221,8222,8223 --threads 120 [--limit 16]
    python generate.py neurovfm_current --split test --shard 0 --nshards 6 --gpu 0
    python generate.py neurovfm14b --split test --ports 8232,8233
    python generate.py sf_strip <gen_<split>.jsonl> [...]
Paths: BRAINDIFF_PAIRS (data/stage2/pairs.parquet), BASELINE_DIR (checkpoints/table2_baselines; montages in <BASELINE_DIR>/montages).
Rows add --repetition_penalty 1.1 (vlm / neurovfm_compare default 1.0).
Outputs <BASELINE_DIR>/<dir>/gen_<split>[_speed][_sub].jsonl {pair_id, n_images, generated}; resumable (pairs already written are skipped)."""
import argparse, base64, hashlib, json, os, sys, threading, time

PAIRS = os.environ.get("BRAINDIFF_PAIRS", "data/stage2/pairs.parquet"); OUT = os.environ.get("BASELINE_DIR", "checkpoints/table2_baselines")
MON = f"{OUT}/montages"; M = ["T1ce", "FLAIR", "T2w", "T1w"]; MR = ["T1w", "T1ce", "T2w", "FLAIR"]   # vlm order / render + NeuroVFM order
NAMES = {"medgemma": "google/medgemma-1.5-4b-it", "qwen3vl8b": "Qwen/Qwen3-VL-8B-Instruct", "medgemma27b": "google/medgemma-27b-it"}
RP = ["--repetition_penalty", "1.1"]
ROWS = {"qwen3vl8b_8": ["--model", "qwen3vl8b", "--img_instr", "--sl8_images"] + RP, "qwen3vl8b_48": ["--model", "qwen3vl8b", "--img_instr", "--slices", "48"] + RP,
        "medgemma27b_8": ["--model", "medgemma27b", "--sl8_images"] + RP, "medgemma27b_48": ["--model", "medgemma27b", "--slices", "48"] + RP,
        "medgemma4b_vol": ["--model", "medgemma", "--vol", "85"] + RP, "qwen3vl8b_ft_sf": ["--model", "qwen3vl8b", "--img_instr", "--sl8_images"] + RP,   # + --lora sf<k>
        "neurovfm14b": ["--model", "neurovfm-qwen3-14b", "--out", "neurovfm14b_rp110"] + RP}
SYSTEM = "You are an expert neuroradiologist. You compare a follow-up brain MRI with the prior examination and write the follow-up report."
INSTR = ("Write the FINDINGS section of the report for the CURRENT examination. Describe the intracranial findings on the current scans and, "
         "for each relevant finding, state whether it is new, increased/worsened, decreased/improved, resolved, or unchanged compared with "
         "the prior examination. Use the prior report as the baseline. Write plain prose without headings or recommendations.")
INSTR_IMG = ("Write the FINDINGS section of the report for the CURRENT examination. Base every comparison on the images: for each finding, "
             "look at the PRIOR and CURRENT images of the same sequence and judge from them whether it is new, increased/worsened, "
             "decreased/improved, resolved, or unchanged. The prior report tells you what was described on the prior examination; it does not "
             "tell you what the current scans show. Do not assume a finding is unchanged, or that it has followed its usual course, unless the "
             "current images support it, and report new findings you see on the current images even if the prior report does not mention them. "
             "Write plain prose without headings or recommendations.")


def b64(p): return base64.b64encode(open(p, "rb").read()).decode()
def matched(r): return [m for m in M if isinstance(r[f"{m}_1"], str) and r[f"{m}_1"] and isinstance(r[f"{m}_2"], str) and r[f"{m}_2"]]
def tail(r, a): return {"type": "text", "text": f"Prior report:\n{(r['prior_edit'] or 'Not available.').strip()}\n\n{INSTR_IMG if a.img_instr else INSTR}"}


def build_sl8(r, a):
    """8-slice images: labelled single slices, current study first, priority T1ce > FLAIR > T2w > T1w, matched series only,
    cap 96 (render --sl8, 8 per volume); then prior report + instruction."""
    labelled = []
    for t, tag in ((2, "Current"), (1, "Prior")):
        for m in matched(r):
            h = hashlib.md5(r[f"{m}_{t}"].encode()).hexdigest(); nf = f"{MON}/{h}_sl8_n.txt"
            n = int(open(nf).read()) if os.path.exists(nf) else 0
            for k in range(n):
                if len(labelled) < 96: labelled.append((f"{tag} study - {m}, slice {k + 1}/{n}", f"{MON}/{h}_sl8_{k}.png"))
    imgs = []
    for lab, f in labelled:
        imgs += [{"type": "text", "text": lab}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(f)}"}}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": imgs + [tail(r, a)]}], len(labelled)


def build_vol(r, a):
    """MedGemma 1.5 volume input [render --vol]: per matched series and timepoint a label, then single slices each followed by
    "SLICE <j>", Q // n_volumes slices per volume; then prior report + instruction."""
    ms = matched(r); content, n = [], 0
    for t, lab in (((1, "PRIOR"), (2, "CURRENT")) if ms else ()):
        for m in ms:
            k = a.vol // (2 * len(ms)); h = hashlib.md5(r[f'{m}_{t}'].encode()).hexdigest(); fs = [f"{MON}/{h}_vol{k}_{j}.png" for j in range(k)]
            if not all(os.path.exists(f) for f in fs): continue
            content.append({"type": "text", "text": f"{lab} examination, {m}: {k} axial slices, inferior to superior."})
            for j, f in enumerate(fs, 1):
                content += [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(f)}"}}, {"type": "text", "text": f"SLICE {j}"}]; n += 1
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content + [tail(r, a)]}], n


def build(r, a):
    """N-slice input as N/16 montages per matched series and timepoint (render --slices N): PRIOR first, then CURRENT."""
    content, n, SL = [], 0, a.slices
    for t, lab in ((1, "PRIOR"), (2, "CURRENT")):
        for m in matched(r):
            h = hashlib.md5(r[f'{m}_{t}'].encode()).hexdigest(); fs = [f"{MON}/{h}_s{SL}_{b}.png" for b in range(SL // 16)]
            if not all(os.path.exists(f) for f in fs): continue
            for b, f in enumerate(fs):
                txt = f"{lab} examination, {m}: axial slices {16 * b + 1}-{min(16 * b + 16, SL)} of {SL} (whole brain, inferior to superior), read left-to-right, top-to-bottom."
                content += [{"type": "text", "text": txt}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64(f)}"}}]; n += 1
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content + [tail(r, a)]}], n


def vlm_request(r, a):
    """-> (chat.completions.create kwargs, n_images) for one pair."""
    msgs, n = (build_sl8 if a.sl8_images else build_vol if a.vol else build)(r, a)
    extra = {"vllm_xargs": {"out_rep_penalty": a.repetition_penalty}} if a.repetition_penalty != 1 else {}
    return dict(model=a.lora or NAMES[a.model], messages=msgs, temperature=0, max_tokens=1024, extra_body=extra), n


def vlm_out(a):
    return (f"{OUT}/{a.model}{f'_ft-{a.lora}' if a.lora else ''}{'_imginstr' if a.img_instr else ''}{f'_s{a.slices}' if a.slices != 16 else ''}"
            f"{f'_vol{a.vol}' if a.vol else ''}{'_sl8img' if a.sl8_images else ''}{f'_rp{round(100 * a.repetition_penalty)}' if a.repetition_penalty != 1 else ''}"
            f"/gen_{a.split}{'_speed' if a.limit else ''}{'_sub' if a.pairs else ''}.jsonl")


def vlm(argv):
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    from openai import OpenAI
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=list(NAMES)); ap.add_argument("--split"); ap.add_argument("--ports")
    ap.add_argument("--limit", type=int); ap.add_argument("--threads", type=int, default=48); ap.add_argument("--host", default="localhost")
    ap.add_argument("--img_instr", action="store_true", help="instruct the model to base comparisons on the images -> <model>_imginstr/")
    ap.add_argument("--slices", type=int, default=16, help="axial slices per volume as N/16 montages (render --slices N) -> suffix _s<N>")
    ap.add_argument("--sl8_images", action="store_true", help="8 labelled slices per volume -> _sl8img")
    ap.add_argument("--vol", type=int, default=0, help="MedGemma 1.5 volume input, max slices per query (render --vol) -> suffix _vol<Q>")
    ap.add_argument("--pairs", help="file of pair_ids to run (val subset); output gets suffix _sub")
    ap.add_argument("--repetition_penalty", type=float, default=1.0, help="HF-semantics output-only penalty (out_rep_penalty.py) -> suffix _rp<100p>")
    ap.add_argument("--lora", help="served LoRA adapter name (vllm --lora-modules NAME=dir) requested instead of the base model -> <model>_ft-<NAME>")
    a = ap.parse_args(argv); assert a.sl8_images or a.vol or (a.slices % 16 == 0 and a.slices > 16), "montage input needs --slices 32, 48, ..."
    p = pd.read_parquet(PAIRS); p = p[p.split == a.split].sort_values("pair_id")
    if a.limit: p = p.iloc[:a.limit]
    if a.pairs: p = p[p.pair_id.isin({int(x) for x in open(a.pairs).read().split()})]
    out = vlm_out(a); os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["pair_id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [r for _, r in p.iterrows() if int(r.pair_id) not in done]
    clients = [OpenAI(base_url=f"http://{q if ':' in q else f'{a.host}:{q}'}/v1", api_key="local", timeout=900) for q in a.ports.split(",")]  # port or host:port
    lock = threading.Lock(); t0 = time.time(); cnt = [0]

    def run(i_r):
        i, r = i_r; kw, n = vlm_request(r, a)
        for k in range(3):
            try:
                c = clients[i % len(clients)].chat.completions.create(**kw); txt = c.choices[0].message.content or ""; break
            except Exception as e:
                txt = None; err = f"{type(e).__name__}: {str(e)[:150]}"; time.sleep(5)
        with lock:
            if txt is None: print("ERR", r.pair_id, err, flush=True); return
            with open(out, "a") as f: f.write(json.dumps({"pair_id": int(r.pair_id), "n_images": n, "generated": txt}) + "\n")
            cnt[0] += 1
            if cnt[0] % 50 == 0 or cnt[0] == len(todo): print(f"{cnt[0]}/{len(todo)} {(time.time() - t0) / cnt[0]:.2f} s/pair wall", flush=True)

    with ThreadPoolExecutor(a.threads) as ex: list(ex.map(run, enumerate(todo)))
    print(f"FINISHED {cnt[0]} in {time.time() - t0:.0f} s", flush=True)


# ---- render: <md5 of the volume path>[...].png in MON; matched series only; canonical RAS, axial, radiological display
def volume_to_slices(path, slices_per_seq=8, size=224, axis=2):
    """Per-volume min-max to [0, 1], drop > 90%-black slices, evenly subsample to slices_per_seq, 224 x 224 bilinear, uint8.
    No reorientation (array axes as stored)."""
    import cv2, nibabel as nib, numpy as np
    vol = nib.load(path).get_fdata().astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-6:
        return []
    vol = np.moveaxis((vol - vmin) / (vmax - vmin), axis, 0)
    keep = [i for i in range(vol.shape[0]) if (vol[i] < 1e-3).mean() <= 0.90]
    if not keep:
        return []
    if len(keep) > slices_per_seq:
        keep = [keep[j] for j in np.linspace(0, len(keep) - 1, slices_per_seq).round().astype(int)]
    return [(np.clip(cv2.resize(vol[i], (size, size), interpolation=cv2.INTER_LINEAR), 0, 1) * 255).astype(np.uint8) for i in keep]


def make_sl8(p):   # <md5>_sl8_<k>.png (k < n <= 8) + <md5>_sl8_n.txt
    from PIL import Image
    o = f"{MON}/{hashlib.md5(p.encode()).hexdigest()}_sl8_n.txt"
    if os.path.exists(o): return 1
    try:
        sl = volume_to_slices(p, slices_per_seq=8)
        for k, x in enumerate(sl): Image.fromarray(x, "L").save(o.replace("_sl8_n.txt", f"_sl8_{k}.png"))
        open(o, "w").write(str(len(sl))); return 1
    except Exception as e:
        print("ERR", p, type(e).__name__, flush=True); return 0


def make_vol(pk):
    """MedGemma 1.5 volume input: k single axial slices per volume, evenly spaced over the non-mostly-black slices, per-volume
    min-max (no percentile clip), native in-plane size; <md5>_vol<k>_<j>.png, j inferior to superior."""
    import nibabel as nib, numpy as np
    from PIL import Image
    p, k = pk; outs = [f"{MON}/{hashlib.md5(p.encode()).hexdigest()}_vol{k}_{j}.png" for j in range(k)]
    if all(os.path.exists(o) for o in outs): return 1
    try:
        v = nib.as_closest_canonical(nib.load(p)).get_fdata(dtype=np.float32)
        v = (v - v.min()) / max(float(v.max() - v.min()), 1e-6)
        keep = [z for z in range(v.shape[2]) if (v[:, :, z] < 1e-3).mean() <= 0.9]
        if not keep: return 0
        sel = [keep[j] for j in np.linspace(0, len(keep) - 1, k).round().astype(int)]
        for o, z in zip(outs, sel): Image.fromarray((np.fliplr(np.flipud(v[:, :, z].T)) * 255).round().astype(np.uint8), "L").save(o)
        return 1
    except Exception as e:
        print("ERR", p, type(e).__name__, flush=True); return 0


def make(pn):
    """N slices as N/16 896 x 896 montages (4 x 4 tiles of 224 px, INTER_AREA): intensity clipped to the volume's 0.5-99.5
    percentiles, evenly spaced over the non-mostly-black slices; <md5>_s<N>_<b>.png, block b = slices 16b+1..16b+16 inferior to superior."""
    import cv2, nibabel as nib, numpy as np
    from PIL import Image
    p, N = pn; outs = [f"{MON}/{hashlib.md5(p.encode()).hexdigest()}_s{N}_{b}.png" for b in range(N // 16)]
    if all(os.path.exists(o) for o in outs): return 1
    try:
        v = nib.as_closest_canonical(nib.load(p)).get_fdata(dtype=np.float32)
        lo, hi = np.percentile(v[v > 0], [0.5, 99.5]) if (v > 0).any() else (0, 1)
        v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
        keep = [k for k in range(v.shape[2]) if (v[:, :, k] < 1e-3).mean() <= 0.9]
        if not keep: return 0
        sel = [keep[j] for j in np.linspace(0, len(keep) - 1, N).round().astype(int)]
        tiles = [cv2.resize(np.fliplr(np.flipud(v[:, :, k].T)), (224, 224), interpolation=cv2.INTER_AREA) for k in sel]
        for b, o in enumerate(outs):
            t = tiles[16 * b:16 * (b + 1)]
            grid = np.vstack([np.hstack(t[r * 4:(r + 1) * 4]) for r in range(len(t) // 4)])
            Image.fromarray((grid * 255).astype(np.uint8), "L").save(o)
        return 1
    except Exception as e:
        print("ERR", p, type(e).__name__, flush=True); return 0


def render(argv):
    from multiprocessing import Pool
    import pandas as pd
    ap = argparse.ArgumentParser(); ap.add_argument("--split", nargs="+"); ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--slices", type=int, default=0, help="N-slice montages, N a multiple of 16")
    ap.add_argument("--sl8", action="store_true", help="8 labelled slices"); ap.add_argument("--vol", type=int, default=0, help="MedGemma 1.5 volume input, max slices per query")
    a = ap.parse_args(argv); os.makedirs(MON, exist_ok=True)
    p = pd.read_parquet(PAIRS); p = p[p.split.isin(a.split)]
    ms = lambda r: [m for m in MR if isinstance(r[f"{m}_1"], str) and r[f"{m}_1"] and isinstance(r[f"{m}_2"], str) and r[f"{m}_2"]]
    if a.vol:
        jobs, f = sorted({(r[f"{m}_{t}"], a.vol // (2 * len(ms(r)))) for _, r in p.iterrows() for m in ms(r) for t in (1, 2)}), make_vol
    else:
        paths = sorted({r[f"{m}_{t}"] for _, r in p.iterrows() for m in ms(r) for t in (1, 2)})
        assert a.sl8 or (a.slices and a.slices % 16 == 0), "--sl8, --vol Q or --slices N (multiple of 16)"
        jobs, f = (paths, make_sl8) if a.sl8 else ([(x, a.slices) for x in paths], make)
    print(len(jobs), "render jobs", flush=True)
    with Pool(a.workers) as pool: ok = sum(pool.imap_unordered(f, jobs, chunksize=16))
    print("done", ok, "/", len(jobs))


# ---- NeuroVFM -> Qwen3-14B
def neurovfm_current(argv):
    """Step 1: released NeuroVFM (mlinslab/neurovfm-llm, not fine-tuned) writes a single-study report of the CURRENT scan with its
    own pipeline (study_cu_seqlens fixed). -> <BASELINE_DIR>/neurovfm/current_<split>_<shard>.jsonl {pair_id, study2, report};
    resumable. Needs the neurovfm package (flash-attn, CUDA)."""
    import pandas as pd, torch
    from neurovfm.pipelines import load_vlm
    ap = argparse.ArgumentParser(); ap.add_argument("--split"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--limit", type=int); ap.add_argument("--pairs", help="file of pair_ids; filtered before sharding")
    a = ap.parse_args(argv)
    p = pd.read_parquet(PAIRS); p = p[p.split == a.split].sort_values("pair_id")
    if a.pairs: p = p[p.pair_id.isin({int(x) for x in open(a.pairs).read().split()})]
    p = p.iloc[a.shard::a.nshards]
    if a.limit: p = p.iloc[:a.limit]
    out = f"{OUT}/neurovfm/current_{a.split}_{a.shard}.jsonl"; os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["pair_id"] for l in open(out)} if os.path.exists(out) else set()
    gen, pre = load_vlm("mlinslab/neurovfm-llm", device=f"cuda:{a.gpu}"); t0 = time.time(); n = 0
    with open(out, "a") as f:
        for _, r in p.iterrows():
            if int(r.pair_id) in done: continue
            paths = [r[f"{m}_2"] for m in MR if isinstance(r[f"{m}_1"], str) and r[f"{m}_1"] and isinstance(r[f"{m}_2"], str) and r[f"{m}_2"]]
            try:
                b = pre.load_study(paths, modality="mri")
                b["study_cu_seqlens"] = torch.tensor([0, int(b["series_cu_seqlens"][-1])], dtype=torch.int32)
                rep = gen(b, clinical_context=None)
            except Exception as e:
                print("ERR", r.pair_id, type(e).__name__, str(e)[:120], flush=True); continue
            f.write(json.dumps({"pair_id": int(r.pair_id), "study2": r.study2, "report": rep}) + "\n"); f.flush(); n += 1
            if n % 20 == 0: print(f"{n} done, {(time.time() - t0) / n:.1f} s/study", flush=True)
    print(f"FINISHED {n} in {time.time() - t0:.0f} s ({(time.time() - t0) / max(n, 1):.1f} s/study)", flush=True)


def compare_request(r, cur, model, rp):
    """Step 2: text-only merge of the edited prior report and NeuroVFM's current-scan report, thinking off."""
    u = (f"Prior report:\n{(r['prior_edit'] or 'Not available.').strip()}\n\nReport of the CURRENT examination written by an AI model from the "
         f"current scans alone (it did not see the prior):\n{cur.strip()}\n\n{INSTR}")
    return dict(model=model, temperature=0, max_tokens=1024, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}} | ({"vllm_xargs": {"out_rep_penalty": rp}} if rp != 1 else {}))


def neurovfm_compare(argv):
    import glob
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    from openai import OpenAI
    ap = argparse.ArgumentParser(); ap.add_argument("--split"); ap.add_argument("--ports"); ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--model", default="neurovfm-qwen3-14b", help="served merger name"); ap.add_argument("--out", default="neurovfm14b_rp110")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--repetition_penalty", type=float, default=1.0, help="output-only penalty on the merge step")
    a = ap.parse_args(argv)
    cur = {}
    for f in glob.glob(f"{OUT}/neurovfm/current_{a.split}_*.jsonl"):
        for l in open(f): r = json.loads(l); cur[r["pair_id"]] = r["report"]
    p = pd.read_parquet(PAIRS); p = p[p.split == a.split]
    os.makedirs(f"{OUT}/{a.out}", exist_ok=True); out = f"{OUT}/{a.out}/gen_{a.split}.jsonl"; done = {json.loads(l)["pair_id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [r for _, r in p.iterrows() if int(r.pair_id) in cur and int(r.pair_id) not in done][:a.limit or None]
    print(f"{len(cur)} NeuroVFM current reports / {len(p)} pairs; {len(todo)} to do", flush=True)
    cl = [OpenAI(base_url=f"http://localhost:{q}/v1", api_key="local", timeout=900) for q in a.ports.split(",")]; lock = threading.Lock(); n = [0]; t0 = time.time()

    def run(i_r):
        i, r = i_r; kw = compare_request(r, cur[int(r.pair_id)], a.model, a.repetition_penalty)
        for k in range(3):
            try:
                c = cl[i % len(cl)].chat.completions.create(**kw); txt = c.choices[0].message.content or ""; break
            except Exception as e: txt = None; err = f"{type(e).__name__}: {str(e)[:150]}"; time.sleep(5)
        with lock:
            if txt is None: print("ERR", r.pair_id, err, flush=True); return
            open(out, "a").write(json.dumps({"pair_id": int(r.pair_id), "generated": txt}) + "\n"); n[0] += 1
            if n[0] % 200 == 0: print(f"{n[0]}/{len(todo)} {(time.time() - t0) / n[0]:.2f} s/pair", flush=True)

    with ThreadPoolExecutor(a.threads) as ex: list(ex.map(run, enumerate(todo)))
    print(f"FINISHED {n[0]}", flush=True)


# ---- sf_strip
def sf_strip(files):
    """Cut the leading status line exactly as Stage2.STATUS_RE, in place: generated = report, status_line = the cut line
    ("" if none), generated_raw = the model output. Idempotent."""
    import re
    STATUS_RE = re.compile(r"^((?:(?:NEW|WORSENED|IMPROVED|RESOLVED)\.\s*)*No (?:other )?interval change\.)\s*\n?", re.I)
    for fn in files:
        rows = [json.loads(l) for l in open(fn) if l.strip()]; ok = 0
        for r in rows:
            raw = r.setdefault("generated_raw", r["generated"]); m = STATUS_RE.match(raw or ""); ok += bool(m)
            r["status_line"], r["generated"] = (m.group(1), raw[m.end():].strip()) if m else ("", raw)
        with open(fn, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"{fn}: {len(rows)} rows, status line well formed {ok / max(len(rows), 1):.3f}", flush=True)


if __name__ == "__main__":
    TOOLS = {"vlm": vlm, "render": render, "neurovfm_current": neurovfm_current, "neurovfm_compare": neurovfm_compare, "sf_strip": sf_strip}
    if len(sys.argv) < 2 or sys.argv[1] not in {**ROWS, **TOOLS}:
        print(__doc__); sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd in ROWS: (neurovfm_compare if cmd == "neurovfm14b" else vlm)(ROWS[cmd] + rest)
    else: TOOLS[cmd](rest)
