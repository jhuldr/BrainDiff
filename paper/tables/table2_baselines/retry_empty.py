"""Re-run ONLY the empty-response studies and merge back in. Two stages so the API call
(CLIP env, has keys, no nibabel) is decoupled from payload building (BrainDiff env, has nibabel):

  # stage 1 (BrainDiff env): build image payloads for the empties into a json
  python retry_empty.py --build --dump outputs/openai/reports.jsonl --payloads _retry_openai.json
  # stage 2 (CLIP env): call the API from the prebuilt payloads, merge into the dump
  python retry_empty.py --provider openai --dump outputs/openai/reports.jsonl \
       --payloads _retry_openai.json --model <id> [--use-responses] --max-tokens 4000
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM, user_text

def empties_of(dump):
    return [json.loads(l)["study_uid2"] for l in open(dump)
            if not (json.loads(l).get("hyp") or "").strip()]

def build(a):
    from common import rows_with_payload            # nibabel needed here only
    want = set(empties_of(a.dump)); out = []
    for r in rows_with_payload(a.slices_per_seq, bool(a.both_timepoints), a.max_images, None):
        if r["study_uid2"] in want:
            out.append({"study_uid2": r["study_uid2"], "prior_report": r["prior_report"],
                        "ref": r["ref"], "n_current": r["n_current"], "n_prior": r["n_prior"],
                        "images": r["images"]})
    json.dump(out, open(a.payloads, "w"))
    print(f"built {len(out)} payloads for empties -> {a.payloads} "
          f"(images/study: {[len(r['images']) for r in out]})")

def claude_msgs(r):
    c = [{"type": "text", "text": user_text(r["prior_report"], r["n_current"], r["n_prior"])}]
    for label, b64 in r["images"]:
        c += [{"type": "text", "text": label},
              {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]
    return [{"role": "user", "content": c}]

def openai_msgs(r, detail="high"):
    c = [{"type": "text", "text": user_text(r["prior_report"], r["n_current"], r["n_prior"])}]
    for label, b64 in r["images"]:
        c += [{"type": "text", "text": label},
              {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": detail}}]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": c}]

def call(a):
    pay = {r["study_uid2"]: r for r in json.load(open(a.payloads))}
    rows = [json.loads(l) for l in open(a.dump)]
    empties = [d["study_uid2"] for d in rows if not (d.get("hyp") or "").strip()]
    print(f"[{a.provider}] retrying {len(empties)}: {empties}")
    fixed = {}
    if a.provider == "claude":
        import anthropic; client = anthropic.Anthropic()
        for uid in empties:
            resp = client.messages.create(model=a.model, system=SYSTEM, max_tokens=a.max_tokens, messages=claude_msgs(pay[uid]))
            hyp = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            fixed[uid] = hyp; print(f"  {uid}: {len(hyp)} chars stop={resp.stop_reason}")
    else:
        from openai import OpenAI; client = OpenAI()
        for uid in empties:
            msgs = openai_msgs(pay[uid], a.detail)
            if a.use_responses:
                resp = client.responses.create(model=a.model, input=msgs, max_output_tokens=a.max_tokens)
                hyp = resp.output_text or ""; fr = getattr(resp, "status", "?")
            else:
                resp = client.chat.completions.create(model=a.model, messages=msgs, max_completion_tokens=a.max_tokens)
                hyp = resp.choices[0].message.content or ""; fr = resp.choices[0].finish_reason
            fixed[uid] = hyp; print(f"  {uid}: {len(hyp)} chars finish={fr}")
    for d in rows:
        if fixed.get(d["study_uid2"], "").strip(): d["hyp"] = fixed[d["study_uid2"]]
    with open(a.dump, "w") as f:
        for d in rows: f.write(json.dumps(d) + "\n")
    still = sum(1 for d in rows if not (d.get("hyp") or "").strip())
    print(f"merged; remaining empties: {still} -> {a.dump}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--provider", choices=["claude", "openai"])
    ap.add_argument("--dump", required=True); ap.add_argument("--payloads", required=True)
    ap.add_argument("--model", default=""); ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--slices-per-seq", type=int, default=8); ap.add_argument("--both-timepoints", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=96); ap.add_argument("--detail", default="high")
    ap.add_argument("--use-responses", action="store_true")
    a = ap.parse_args()
    build(a) if a.build else call(a)

if __name__ == "__main__":
    main()
