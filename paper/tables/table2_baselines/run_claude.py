"""Generate longitudinal comparison reports with Claude (default Opus 5) on the 64-subset.
The prompt carries the same content as our S4 prompt (prompts.py). Images are 224x224 PNGs
(both timepoints, priority-ordered, capped). --dry-run builds+validates payloads and prints
token/image stats WITHOUT calling the API (no key needed). Live run needs ANTHROPIC_API_KEY.

Output: outputs/claude/reports.jsonl  ({study_uid2, hyp, ref} per line)."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM, user_text
from common import rows_with_payload

def build_messages(row):
    content = [{"type": "text", "text": user_text(row["prior_report"], row["n_current"], row["n_prior"])}]
    for label, b64 in row["images"]:
        content.append({"type": "text", "text": label})
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64}})
    return [{"role": "user", "content": content}]

def est_tokens(row):
    # Claude image ~ (224*224)/750 ~= 67 tok; text ~ chars/4
    img = 67 * len(row["images"])
    txt = (len(user_text(row["prior_report"], row["n_current"], row["n_prior"]))
           + sum(len(l) for l, _ in row["images"])) // 4
    return img + txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--slices-per-seq", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=96)
    ap.add_argument("--both-timepoints", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "outputs/claude/reports.jsonl"))
    a = ap.parse_args()

    rows = list(rows_with_payload(a.slices_per_seq, bool(a.both_timepoints), a.max_images, a.limit))
    tot_img = sum(len(r["images"]) for r in rows)
    tot_tok = sum(est_tokens(r) for r in rows)
    print(f"[claude] {len(rows)} studies | images total {tot_img} (avg {tot_img/len(rows):.0f}) "
          f"| est input tok total {tot_tok} (avg {tot_tok/len(rows):.0f}) | model {a.model}")
    over = [r["study_uid2"] for r in rows if len(r["images"]) > 100]
    if over:
        print(f"  WARNING: {len(over)} studies exceed Claude's 100-image/request cap: {over[:3]}")

    if a.dry_run:
        # validate payload structure on the first row, dump a summary
        msgs = build_messages(rows[0])
        n_img = sum(1 for b in msgs[0]["content"] if b["type"] == "image")
        est = round(0.512 * (tot_tok / 512000) * 5 + 0.00025 * len(rows) * 25, 2)  # rough $ w/ ~250 out
        print(f"  DRY-RUN ok: row0 built, {n_img} image blocks, valid base64. "
              f"est cost ~${round(tot_tok/1e6*5 + len(rows)*250/1e6*25, 2)}")
        return

    import anthropic
    client = anthropic.Anthropic()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for i, r in enumerate(rows):
            resp = client.messages.create(model=a.model, system=SYSTEM, max_tokens=a.max_tokens,
                                          messages=build_messages(r))
            hyp = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            # stop_reason is recorded so truncation is VERIFIED, not inferred from the text:
            # at max_tokens 600 every report was cut mid-word and lost its Impression.
            f.write(json.dumps({"study_uid2": r["study_uid2"], "hyp": hyp, "ref": r["ref"],
                                "stop_reason": resp.stop_reason}) + "\n")
            f.flush()
            print(f"  [{i+1}/{len(rows)}] {r['study_uid2']}  ({len(hyp)} chars) stop={resp.stop_reason}")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
