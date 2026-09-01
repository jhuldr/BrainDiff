"""Change-class confusion matrix for BrainDiff's generated test reports.

Reproducible generator. True labels are the 7-way `classification` from s4_test.csv;
predicted labels are gpt-5.4's classification of BrainDiff's OWN generated comparison
report (nodelta_10, real condition), produced by the batch in confusion/README and
cached in confusion/braindiff_pred_1444.json (one predicted class per study_uid2, all
1,444 test studies). This script does NOT call any API -- it only tabulates and plots
the cached predictions against the true labels, so it is deterministic.

The 7 classes are collapsed to 4 by clinical direction (see COLLAPSE):
  Stable | Improved(+Resolved) | Worsened(+New lesion) | Conflicting(+Mixed, +Indeterminate)
Row-normalized -> diagonal is per-class recall; balanced accuracy = macro recall.

Usage:
  python paper/tables/table2_baselines/confusion_matrix              # 4-class (default)
  python paper/tables/table2_baselines/confusion_matrix --classes 7  # full 7-class
  python paper/tables/table2_baselines/confusion_matrix --out figs/cm.png
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
_TEST_CSV_DEFAULT = _os.path.join(_REPO, "paper", "cache", "s4_test.csv")
if not _os.path.exists(_TEST_CSV_DEFAULT):
    _TEST_CSV_DEFAULT = "/home/data/BRAIN_DIFF_S4/splits_extended/s4_test.csv"
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = _TEST_CSV_DEFAULT
PRED_JSON = os.path.join(_REPO, "paper", "cache", "confusion", "braindiff_pred_1444.json")       # gpt-5.4 on BrainDiff report
TFIDF_JSON = os.path.join(_REPO, "paper", "cache", "confusion", "tfidf_pred_1444.json")           # TF-IDF classifier (LLM-free)
RULE_JSON = os.path.join(_REPO, "paper", "cache", "confusion", "rule_pred_1444.json")             # temporal_score change_class (LLM-free, the paper's consistent metric)
# true-label source: 'synth' = the synthesized `classification` in s4_test.csv (made in the
# same LLM pass as the training target); 'real' = a fresh gpt-5.4 classification of the two
# REAL radiologist reports (report1->report2), cached below. See module docstring / README.
REAL_JSON = os.path.join(_REPO, "paper", "cache", "confusion", "realreport_true_1444.json")

SEVEN = ["Stable", "Improved", "Progressed", "New lesion", "Resolved",
         "Mixed interval change", "Indeterminate"]

# 7 -> 4 collapse by clinical direction. Mixed and Indeterminate both express a change
# whose net direction cannot be assigned, so they pool into "Conflicting".
COLLAPSE = {
    "Stable": "Stable",
    "Improved": "Improved", "Resolved": "Improved",
    "Progressed": "Worsened", "New lesion": "Worsened",
    "Mixed interval change": "Indeterminate", "Indeterminate": "Indeterminate",
}
FOUR = ["Stable", "Improved", "Worsened", "Indeterminate"]


def load_labels(truth="synth", pred_src="gpt"):
    if truth == "synth":
        true = {r["study_uid2"]: r["classification"]
                for r in csv.DictReader(open(TEST_CSV)) if r.get("classification")}
    elif truth == "real":
        true = json.load(open(REAL_JSON))
    else:
        raise SystemExit(f"unknown truth source {truth!r}")
    pred = json.load(open({"tfidf": TFIDF_JSON, "rule": RULE_JSON}.get(pred_src, PRED_JSON)))
    uids = [u for u in pred if u in true]
    missing_true = [u for u in pred if u not in true]
    if missing_true:
        raise SystemExit(f"{len(missing_true)} predicted uids have no {truth} true label")
    return true, pred, uids


def confusion(true, pred, uids, classes, collapse=None):
    idx = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(classes), len(classes)))
    f = (lambda x: collapse[x]) if collapse else (lambda x: x)
    for u in uids:
        m[idx[f(true[u])], idx[f(pred[u])]] += 1
    return m


def render(M, classes, out, n):
    support = M.sum(1)
    Mn = M / support[:, None]
    recall = np.diag(Mn)
    macro = float(np.mean(recall))
    acc = float(np.trace(M) / M.sum())
    k = len(classes)
    disp = [c.replace("New lesion", "New\nlesion") for c in classes]

    cmap = LinearSegmentedColormap.from_list(
        "bd", ["#f7fbff", "#c9dcef", "#6aaed6", "#2f6fb2", "#08306b"])
    sz = 6.6 if k == 4 else 8.8
    fig, ax = plt.subplots(figsize=(sz, sz * 0.86))
    im = ax.imshow(Mn, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    # minimal: matrix + class labels + colorbar only
    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3 if k == 4 else 2.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    fs = 12 if k == 4 else 10.5
    ax.set_xticklabels(disp, rotation=(20 if k == 4 else 35), ha="right", fontsize=fs)
    ax.set_yticklabels(disp, fontsize=fs); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(k):
        for j in range(k):
            v = Mn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v*100:.0f}%" if k == 4 else f"{v*100:.0f}",
                        ha="center", va="center",
                        color="white" if v > 0.55 else "#1a1a1a",
                        fontsize=(14 if k == 4 else 9.5) if i == j else (12 if k == 4 else 9.5),
                        fontweight="bold" if i == j else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False); cb.ax.tick_params(length=0)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return Mn, recall, macro, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=4, choices=[4, 7])
    ap.add_argument("--truth", default="synth", choices=["synth", "real"],
                    help="true-label source: 'synth' (s4_test classification) or "
                         "'real' (gpt-5.4 on real report1->report2)")
    ap.add_argument("--pred", default="rule", choices=["rule", "tfidf", "gpt"],
                    help="predicted-label source: 'rule' (temporal_score change_class — the "
                         "paper's consistent LLM-free metric), 'tfidf' (trained classifier), "
                         "or 'gpt' (gpt-5.4 on BrainDiff report)")
    ap.add_argument("--out", default=os.path.join(_REPO, "paper", "cache", "confusion", "braindiff_confusion.png"))
    args = ap.parse_args()
    true, pred, uids = load_labels(args.truth, args.pred)
    classes = FOUR if args.classes == 4 else SEVEN
    collapse = COLLAPSE if args.classes == 4 else None
    M = confusion(true, pred, uids, classes, collapse)
    Mn, recall, macro, acc = render(M, classes, args.out, len(uids))
    print(f"n={len(uids)}  overall_acc={acc:.4f}  balanced_acc(macro recall)={macro:.4f}")
    for c, r, s in zip(classes, recall, M.sum(1)):
        print(f"  {c:22s} recall={r:.3f}  support={int(s)}")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
