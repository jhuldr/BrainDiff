"""Merge select_ckpts.py shards per checkpoint tag, compute in-env metrics
(triple_f1 / BLEU-4 / METEOR), and write one sel_<tag>.json per checkpoint for the
batched RadGraph pass. Prints a table sorted by triple_f1 as a quick in-env preview;
rg_er ranking comes from `-m braindiff.eval.radgraph_score --hyp_ref_dir <out> --model_name sel`.
"""
import argparse, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from braindiff.eval import nlg_score
from braindiff.eval.temporal_score import score as temporal_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    merged_dir = os.path.join(args.out_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    shards = glob.glob(os.path.join(args.out_dir, "sel_*.rank*.json"))
    tags = {}
    for s in shards:
        tag = re.sub(r"\.rank\d+\.json$", "", os.path.basename(s))[4:]  # strip 'sel_' and rank
        tags.setdefault(tag, []).append(s)

    rows = []
    for tag, files in sorted(tags.items()):
        hyps, refs = [], []
        for s in sorted(files):
            d = json.load(open(s))
            hyps.extend(d["hyps"]); refs.extend(d["refs"])
        merged = os.path.join(merged_dir, f"sel_{tag}.json")
        with open(merged, "w") as f:
            json.dump({"hyps": hyps, "refs": refs}, f)
        nlg = nlg_score.score(hyps, refs)
        ts = temporal_score(hyps, refs)
        rows.append((tag, len(hyps), ts["triple_f1"], nlg["bleu4"], nlg["meteor"], len(files)))

    def ep(t):
        m = re.search(r"_(\d+)$", t); return int(m.group(1)) if m else 0
    print(f"\n  {'checkpoint':40s} {'n':>4s} {'triple_f1':>9s} {'bleu4':>7s} {'meteor':>7s} {'shards':>6s}")
    for tag, n, tf, b4, mt, ns in sorted(rows, key=lambda r: ep(r[0])):
        print(f"  {tag:40s} {n:>4d} {tf:>9.4f} {b4:>7.4f} {mt:>7.4f} {ns:>6d}")
    best = max(rows, key=lambda r: r[2])
    print(f"\n  best by in-env triple_f1: {best[0]} ({best[2]:.4f}) "
          f"-- confirm with rg_er batch scoring", flush=True)


if __name__ == "__main__":
    main()
