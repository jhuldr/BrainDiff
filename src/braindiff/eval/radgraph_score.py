"""RadGraph-XL scorer.

radgraph pins transformers <5, which conflicts with the decoder's transformers 5.x, so it
cannot share the training interpreter -- this runs in the isolated `radgraph_env`.

Two modes:
- One-shot (default): reads {"hyps": [...], "refs": [...]} JSON from stdin, prints one
  "RG_RESULT {...}" line to stdout.
- Batch (--hyp_ref_dir): scores every <checkpoint>.json in a directory in one process, so
  F1RadGraph loads once. Prints a table sorted by rg_er and writes rg_er_results.json
  naming the best checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

from radgraph import F1RadGraph


def score(hyps, refs, f1radgraph):
    """Drop empty pairs RadGraph can't parse, return rg_er (entity+relation F1)."""
    pairs = [(h, r) for h, r in zip(hyps, refs) if h and r]
    if not pairs:
        return 0.0, 0
    h_list, r_list = map(list, zip(*pairs))
    mean_reward, *_ = f1radgraph(hyps=h_list, refs=r_list)
    return float(mean_reward[1]), len(pairs)   # (rg_e, rg_er, rg_bar_er)


def run_one_shot():
    payload = json.load(sys.stdin)
    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
    rg_er, n = score(payload["hyps"], payload["refs"], f1radgraph)
    # Sentinel-prefixed so the caller ignores any library chatter on stdout.
    print(f"RG_RESULT {json.dumps({'rg_er': rg_er, 'n': n})}", flush=True)


def run_batch(hyp_ref_dir, run_name):
    hyp_ref_dir = Path(hyp_ref_dir)
    paths = sorted(hyp_ref_dir.glob("*.json"))
    paths = [p for p in paths if p.name != "rg_er_results.json" and run_name.split(".")[0] in p.name]
    if not paths:
        raise SystemExit(f"No *.json files found in {hyp_ref_dir}")

    print(f"Loading RadGraph-XL once for {len(paths)} checkpoints...")
    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")

    results = {}
    for path in paths:
        payload = json.loads(path.read_text())
        rg_er, n = score(payload["hyps"], payload["refs"], f1radgraph)
        results[path.stem] = {"rg_er": rg_er, "n": n}
        print(f"  {path.stem}: rg_er={rg_er:.4f}  (n={n})")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["rg_er"], reverse=True)
    best_name, best = ranked[0]

    print("\n=== Results (sorted by rg_er) ===")
    for name, r in ranked:
        marker = "  *" if name == best_name else ""
        print(f"  {name}: rg_er={r['rg_er']:.4f}{marker}")
    print(f"\nBest checkpoint: {best_name}  rg_er={best['rg_er']:.4f}")

    out_path = hyp_ref_dir / "rg_er_results.json"
    out_path.write_text(json.dumps({"results": results, "best": best_name}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hyp_ref_dir", default=None,
                    help="Score every *.json in this dir (evaluate_checkpoints.py's "
                         "rg_er_cache); omit to read a single {hyps,refs} JSON from stdin.")
    p.add_argument("--model_name", default=None)

    args = p.parse_args()

    if args.hyp_ref_dir and args.model_name:
        run_batch(args.hyp_ref_dir, args.model_name)
    else:
        run_one_shot()
