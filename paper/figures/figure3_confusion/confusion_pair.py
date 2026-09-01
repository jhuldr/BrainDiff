"""Two-panel change-class confusion figure: (a) BrainDiff, (b) base NeuroVFM.

One float, shared colourbar, sized for the paper. Both panels use the same true labels (the
7-way `classification` in s4_test.csv, collapsed to 4 by clinical direction) and the same
LLM-free predictor, so the only difference between panels is the report being classified:

  (a) BrainDiff  -- confusion/rule_pred_1444.json, from our nodelta_10 reports
  (b) NeuroVFM   -- outputs/neurovfm_base/fulltest_longitudinal_meta_reports.csv

Row-normalised, so the diagonal is per-class recall. Supports are identical across panels,
which is why panel (b) drops its y tick labels. No API calls: tabulates cached predictions.

Type is DejaVu Sans. The float is authored large and scaled down by the LaTeX include, so
every font size is a printed point size converted with SCALE. Run --check to print the
measured printed sizes instead of trusting the arithmetic.

  python paper/figures/figure3_confusion/confusion_pair.py --check
"""
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
import sys as _sys
_sys.path.insert(0, _HERE)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, _REPO)
csv.field_size_limit(10 ** 7)
from confusion_matrix import (
    COLLAPSE, FOUR, TEST_CSV, HERE, confusion)
from braindiff.eval.temporal_score import change_class, change_class_v2, change_class_v5

BASE_CSV = os.path.join(_REPO, "paper", "cache", "outputs", "neurovfm_base",
                        "fulltest_longitudinal_meta_reports.csv")
RULE_JSON = os.path.join(_REPO, "paper", "cache", "confusion", "rule_pred_1444.json")
# uid -> predicted class, precomputed from each system's generations (see matrices_v2).
PRED_V1 = os.path.join(_REPO, "paper", "cache", "confusion", "pred_v1.json")
PRED_V2 = os.path.join(_REPO, "paper", "cache", "confusion", "pred_v2.json")

# The include scales a SOURCE_PANEL_PT-wide panel down to TARGET_PANEL_PT in the paper.
SOURCE_PANEL_PT = 491.0
TARGET_PANEL_PT = 230.0
SCALE = TARGET_PANEL_PT / SOURCE_PANEL_PT          # 0.4684

# Printed sizes (pt on paper); source sizes are these / SCALE.
PRINT_CELL = 7.0
PRINT_TICK = 7.0
PRINT_AXIS = 7.5
PRINT_TITLE = 8.0
# Printed size / source size. Set from a measurement of the rendered float, not from the
# nominal SCALE: tight-bbox trims whitespace, so the real include scale is larger and
# assuming the nominal one puts every label ~10% over target. See the two-pass in main().
_S = {"v": SCALE}
src = lambda pt: pt / _S["v"]

TARGET_TOTAL_PT = 2 * TARGET_PANEL_PT + 46 * SCALE      # panels + colourbar, printed

# Display label only -- the COLLAPSE key stays "Indeterminate" so the data pipeline,
# confusion_matrix.py and the 5-class variant are all unaffected.
DISPLAY = {"Indeterminate": "Mixed/unclear"}
disp = lambda c: DISPLAY.get(c, c)

# Anchored on #5B8FC7, the decoder blue in Figure 1.
CMAP = LinearSegmentedColormap.from_list(
    "bd5b8f", ["#ffffff", "#dfeaf6", "#9dc0e3", "#5B8FC7", "#2F6BAE", "#123a6b"])


VR = _REPO + "/paper/cache/perreport"


def matrices_v2(cls=change_class_v2):
    """Same true labels as matrices(), predictions from change_class_v2.

    The predicted class per study is read from `confusion/pred_v2.json`, a uid -> class map
    precomputed by applying change_class_v2 to each system's generations. The generations
    are MR-RATE-derived text and are not redistributable, so they are not in this tree; the
    classification they feed is, since it is a label rather than report content.
      BrainDiff -> C_braindiff_real  (the canonical dump)
      NeuroVFM  -> neurovfm_base_fulltest
    """
    rows = list(csv.DictReader(open(TEST_CSV)))
    true = {r["study_uid2"]: r["classification"] for r in rows if r.get("classification")}
    pred = json.load(open(PRED_V2))
    bd, nv = pred["bd"], pred["nv"]

    out = []
    for pred in (bd, nv):
        uids = [u for u in pred if u in true]
        assert len(uids) == 1444, f"expected 1444 joined rows, got {len(uids)}"
        out.append((confusion(true, pred, uids, FOUR, COLLAPSE), len(uids)))
    return out


def matrices():
    true = {r["study_uid2"]: r["classification"]
            for r in csv.DictReader(open(TEST_CSV)) if r.get("classification")}
    bd = json.load(open(RULE_JSON))
    nv = json.load(open(PRED_V1))["nv_csv"]     # change_class over the NeuroVFM generations
    out = []
    for pred in (bd, nv):
        uids = [u for u in pred if u in true]
        M = confusion(true, pred, uids, FOUR, COLLAPSE)
        out.append((M, len(uids)))
    return out


def panel(ax, M, title, ylabels=True, xlabels=True, yrot=0, ylab=True):
    support = M.sum(1)
    Mn = M / support[:, None]
    k = len(FOUR)
    im = ax.imshow(Mn, cmap=CMAP, vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    if xlabels:
        ax.set_xticklabels([disp(c) for c in FOUR], rotation=30, ha="right",
                           rotation_mode="anchor", fontsize=src(PRINT_TICK))
    else:
        ax.set_xticklabels([])
    if ylabels:
        kw = dict(rotation=yrot, ha="right", rotation_mode="anchor") if yrot else {}
        ax.set_yticklabels([f"{disp(c)} (n={int(s)})" for c, s in zip(FOUR, support)],
                           fontsize=src(PRINT_TICK), **kw)
        if ylab:
            ax.set_ylabel("True change", fontsize=src(PRINT_AXIS))
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=src(PRINT_TITLE), fontweight="bold", pad=src(3.0))

    for i in range(k):
        for j in range(k):
            v = Mn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                        color="white" if v > 0.5 else "#1a1a1a",
                        fontsize=src(PRINT_CELL),
                        fontweight="bold" if i == j else "normal")
    return im, np.diag(Mn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_REPO, "paper", "cache", "confusion", "confusion_pair"))
    ap.add_argument("--classifier", default="v1", choices=["v1", "v2", "v5"],
                    help="v1 = change_class (published figure); v2/v5 = the wrapped variants")
    ap.add_argument("--layout", default="h", choices=["h", "v"],
                    help="h = panels side by side; v = stacked, shared x axis")
    ap.add_argument("--check", action="store_true",
                    help="report measured printed sizes rather than assuming them")
    a = ap.parse_args()

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    (Mbd, nbd), (Mnv, nnv) = matrices_v2({'v2': change_class_v2, 'v5': change_class_v5}[a.classifier]) if a.classifier != 'v1' else matrices()

    def build_v():
        """Stacked: (a) over (b), shared x axis, colourbar spanning both."""
        w_in = (SOURCE_PANEL_PT + 46) / 72.0
        h_in = (2 * 175.0 / SCALE) / 72.0
        fig, axes = plt.subplots(2, 1, figsize=(w_in, h_in))
        fig.subplots_adjust(hspace=0.16)   # ~13pt printed between panel (a) and panel (b)'s title
        im, rb = panel(axes[0], Mbd, "(a) BrainDiff", ylabels=True, xlabels=False,
                       yrot=30, ylab=False)
        _, rn = panel(axes[1], Mnv, "(b) NeuroVFM", ylabels=True, xlabels=True,
                      yrot=30, ylab=False)
        cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
        cb.set_ticks([0, .2, .4, .6, .8, 1.0])
        cb.ax.tick_params(labelsize=src(PRINT_TICK), length=0)
        cb.outline.set_visible(False)
        r = fig.canvas.get_renderer()
        x = sum(axes[1].get_position().extents[i] for i in (0, 2)) / 2
        y0 = axes[1].get_tightbbox(r).y0 / fig.get_window_extent().height
        fig.text(x, y0 - 0.02, "Predicted change", ha="center", va="top",
                 fontsize=src(PRINT_AXIS))
        # one "True change" centred on both panels, left of the widest tick label
        W, H = fig.get_window_extent().width, fig.get_window_extent().height
        xl = min(ax.get_tightbbox(r).x0 for ax in axes) / W
        ymid = (axes[0].get_position().y1 + axes[1].get_position().y0) / 2
        fig.text(xl - 0.02, ymid, "True change", rotation=90, ha="right", va="center",
                 fontsize=src(PRINT_AXIS))
        return fig, rb, rn

    def build():
        # Source geometry: two SOURCE_PANEL_PT panels + colourbar; printed height ~190pt
        # leaves ~15pt of the 205pt float for the caption.
        w_in = (2 * SOURCE_PANEL_PT + 46) / 72.0
        h_in = (190.0 / SCALE) / 72.0
        fig, axes = plt.subplots(1, 2, figsize=(w_in, h_in))
        fig.subplots_adjust(wspace=0.10)
        im, rec_bd = panel(axes[0], Mbd, "(a) BrainDiff", ylabels=True)
        _, rec_nv = panel(axes[1], Mnv, "(b) NeuroVFM", ylabels=False)
        cb = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
        cb.set_ticks([0, .2, .4, .6, .8, 1.0])
        cb.ax.tick_params(labelsize=src(PRINT_TICK), length=0)
        cb.outline.set_visible(False)
        # Single x label centred on the two panels (not the figure -- the colourbar would
        # pull it right), placed below the rotated tick labels.
        r = fig.canvas.get_renderer()
        x = sum(a.get_position().extents[i] for i, a in ((0, axes[0]), (2, axes[1]))) / 2
        y0 = min(a.get_tightbbox(r).y0 for a in axes) / fig.get_window_extent().height
        fig.text(x, y0 - 0.035, "Predicted change", ha="center", va="top",
                 fontsize=src(PRINT_AXIS))
        return fig, rec_bd, rec_nv

    # Pass 1 measures the true include scale; pass 2 sizes the type against it.
    mk = build_v if a.layout == "v" else build
    target = (TARGET_PANEL_PT + 46 * SCALE) if a.layout == "v" else TARGET_TOTAL_PT
    fig, _, _ = mk()
    _S["v"] = target / (fig.get_tightbbox(fig.canvas.get_renderer()).width * 72.0)
    plt.close(fig)
    fig, rec_bd, rec_nv = mk()
    axes = fig.axes

    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")

    print(f"(a) BrainDiff n={nbd}  recall={np.round(rec_bd, 3)}  macro={rec_bd.mean():.4f}")
    print(f"(b) NeuroVFM  n={nnv}  recall={np.round(rec_nv, 3)}  macro={rec_nv.mean():.4f}")
    print(f"figure -> {a.out}.{{svg,pdf,png}}")

    if a.check:
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w_pt, h_pt = bb.width * 72.0, bb.height * 72.0
        eff = target / w_pt   # layout-aware: vertical targets one panel + colourbar
        print(f"\n[check] source {w_pt:.1f}x{h_pt:.1f}pt -> printed "
              f"{w_pt*eff:.1f}x{h_pt*eff:.1f}pt at include scale {eff:.4f}")
        r = fig.canvas.get_renderer()
        mw = axes[0].get_window_extent().width / fig.dpi * 72.0 * eff
        pw = axes[0].get_tightbbox(r).width / fig.dpi * 72.0 * eff
        print(f"[check] panel slot   printed {pw:.1f}pt incl. labels "
              f"(target {TARGET_PANEL_PT:.0f}); matrix itself {mw:.1f}pt")
        for name, pt in [("cell %", PRINT_CELL), ("ticks", PRINT_TICK),
                         ("axis label", PRINT_AXIS), ("panel title", PRINT_TITLE)]:
            print(f"[check] {name:12s} source {src(pt):5.1f}pt -> printed {src(pt)*eff:.2f}pt "
                  f"(target {pt})")


if __name__ == "__main__":
    main()
