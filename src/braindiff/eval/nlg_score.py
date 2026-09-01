"""BLEU-4 and METEOR over the {hyps, refs} JSONs eval_s4_pairs.py dumps.

Runs in the training env (nltk is already there); rg_er stays in radgraph_env.

BLEU-4 is corpus-level (nltk.translate.bleu_score.corpus_bleu, uniform 4-gram
weights, method-1 smoothing) -- reports are long enough that sentence-level BLEU
averaging would over-weight the short ones. METEOR is per-report and averaged,
since nltk only provides a single-segment implementation. Both tokenise on
lowercased word characters, so punctuation and case do not count as matches.

    python -m braindiff.eval.nlg_score checkpoints/s4_eval/*.json
"""
import argparse
import json
import re

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

TOKEN_RE = re.compile(r"\w+")


def tok(text):
    return TOKEN_RE.findall(text.lower())


def score(hyps, refs):
    """Drop empty pairs (same rule radgraph_score.py uses) so n matches across metrics."""
    pairs = [(h, r) for h, r in zip(hyps, refs) if h.strip() and r.strip()]
    h_tok = [tok(h) for h, _ in pairs]
    r_tok = [[tok(r)] for _, r in pairs]

    bleu4 = corpus_bleu(r_tok, h_tok, weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=SmoothingFunction().method1)
    meteor = sum(meteor_score(r, h) for r, h in zip(r_tok, h_tok)) / len(pairs)
    return {"bleu4": float(bleu4), "meteor": float(meteor), "n": len(pairs)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    args = p.parse_args()

    for path in args.paths:
        d = json.loads(open(path).read())
        r = score(d["hyps"], d["refs"])
        print(f"{d.get('checkpoint', path)}: BLEU-4={r['bleu4']:.4f}  "
              f"METEOR={r['meteor']:.4f}  (n={r['n']})", flush=True)
