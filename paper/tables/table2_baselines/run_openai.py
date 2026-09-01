"""Generate longitudinal comparison reports with an OpenAI model (default GPT-5.6 Sol) on
the 64-subset. Same prompt content as our S4 prompt. --dry-run builds+validates payloads
without calling the API (no key). Live run needs OPENAI_API_KEY.

NOTE: set --model to the exact OpenAI id for 'GPT-5.6 Sol'. Newer reasoning models may
require the Responses API and `max_completion_tokens`; this uses chat.completions by default
and falls back to the Responses API if --use-responses is set.

Output: outputs/openai/reports.jsonl."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM, user_text
from common import rows_with_payload

def build_messages(row, detail="high"):
    content = [{"type": "text", "text": user_text(row["prior_report"], row["n_current"], row["n_prior"])}]
    for label, b64 in row["images"]:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": detail}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

def est_tokens(row, detail):
    per = 255 if detail == "high" else 85            # 224x224 = one 512 tile
    img = per * len(row["images"])
    txt = (len(user_text(row["prior_report"], row["n_current"], row["n_prior"]))
           + sum(len(l) for l, _ in row["images"])) // 4
    return img + txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-sol")   # set to the exact id at run time
    ap.add_argument("--detail", default="high", choices=["high", "low"])
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--slices-per-seq", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=96)
    ap.add_argument("--both-timepoints", type=int, default=1)
    ap.add_argument("--use-responses", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs/openai/reports.jsonl"))
    a = ap.parse_args()

    rows = list(rows_with_payload(a.slices_per_seq, bool(a.both_timepoints), a.max_images, a.limit))
    tot_img = sum(len(r["images"]) for r in rows)
    tot_tok = sum(est_tokens(r, a.detail) for r in rows)
    print(f"[openai] {len(rows)} studies | images total {tot_img} (avg {tot_img/len(rows):.0f}) "
          f"| est input tok total {tot_tok} (avg {tot_tok/len(rows):.0f}) | detail {a.detail} | model {a.model}")

    if a.dry_run:
        msgs = build_messages(rows[0], a.detail)
        n_img = sum(1 for b in msgs[1]["content"] if b["type"] == "image_url")
        print(f"  DRY-RUN ok: row0 built, {n_img} image_url blocks, valid data URIs. "
              f"est cost ~${round(tot_tok/1e6*1.5 + len(rows)*250/1e6*10, 2)} (at ~$1.5/$10 per M)")
        return

    from openai import OpenAI
    client = OpenAI()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            msgs = build_messages(r, a.detail)
            if a.use_responses:
                resp = client.responses.create(model=a.model, input=msgs, max_output_tokens=a.max_tokens)
                hyp = resp.output_text
            else:
                resp = client.chat.completions.create(model=a.model, messages=msgs,
                                                      max_completion_tokens=a.max_tokens)
                hyp = resp.choices[0].message.content
            f.write(json.dumps({"study_uid2": r["study_uid2"], "hyp": hyp, "ref": r["ref"]}) + "\n")
            f.flush()
            print(f"  [{i+1}/{len(rows)}] {r['study_uid2']}  ({len(hyp or '')} chars)")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
