"""Merge sharded rg_er dumps from probe_delta_rger.py (--num-shards > 1).

Each rank writes logs/rger_dump/rger_<mode>.rank<i>.json holding its slice of the
fixed 60 val batches. This concatenates them per mode into rger_<mode>.json,
computes the in-env metrics (triple_f1 / BLEU-4 / METEOR) over the merged set, and
prints the radgraph-env commands to score rg_er. Run in the BrainDiff env after all
ranks finish.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from braindiff.eval import nlg_score
from braindiff.eval.temporal_score import score as temporal_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=_REPO + "/logs/rger_dump")
    ap.add_argument("--modes", default="real,delta_roll")
    args = ap.parse_args()

    rows = {}
    for mode in args.modes.split(","):
        shards = sorted(glob.glob(os.path.join(args.out_dir, f"rger_{mode}.rank*.json")))
        if not shards:
            print(f"  {mode}: no shards found in {args.out_dir} (looked for rger_{mode}.rank*.json)")
            continue
        hyps, refs = [], []
        for s in shards:
            d = json.load(open(s))
            hyps.extend(d["hyps"]); refs.extend(d["refs"])
        merged = os.path.join(args.out_dir, f"rger_{mode}.json")
        with open(merged, "w") as f:
            json.dump({"hyps": hyps, "refs": refs}, f)
        nlg = nlg_score.score(hyps, refs)
        ts = temporal_score(hyps, refs)
        rows[mode] = dict(triple_f1=ts["triple_f1"], bleu4=nlg["bleu4"],
                          meteor=nlg["meteor"], n=nlg["n"], path=merged, shards=len(shards))
        m = rows[mode]
        print(f"  {mode:11s}  {m['shards']} shards -> n={m['n']:4d}  "
              f"triple_f1={m['triple_f1']:.4f}  bleu4={m['bleu4']:.4f}  "
              f"meteor={m['meteor']:.4f}   -> {merged}", flush=True)

    if "real" in rows and len(rows) > 1:
        print("\n  --- delta effect on the IN-ENV metrics (real - ablated); + => delta HELPS ---")
        for mode in rows:
            if mode == "real":
                continue
            print(f"  real - {mode:9s}  d_triple_f1={rows['real']['triple_f1']-rows[mode]['triple_f1']:+.4f}  "
                  f"d_meteor={rows['real']['meteor']-rows[mode]['meteor']:+.4f}")

    print("\n  Now score rg_er in the RADGRAPH env (per mode):")
    for mode in rows:
        print(f"    python trainer/radgraph_score.py < {rows[mode]['path']}   # -> RG_RESULT for {mode}")
    print(flush=True)


if __name__ == "__main__":
    main()
