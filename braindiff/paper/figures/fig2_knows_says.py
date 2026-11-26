"""Paper Fig. 2 - knows vs says over training: Worsened-vs-Improved AUC on BIND val (knows = canonical-statement
likelihood, says = P[W] - P[I] of the greedy report), 95% patient-bootstrap CIs; star = gap CI excludes 0.
Reads fig2_knows_says.json -> fig2_knows_says.{pdf,png} (next to this file).

    python paper/figures/fig2_knows_says.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

D = json.load(open(os.path.join(HERE, "fig2_knows_says.json")))
PANELS = [("A0", "Frozen-decoder curriculum"), ("Stage 1-D", "Stage 1-D, standard format"),
          ("Status-first (main)", "Stage 1-D + status-first (ours)")]
INK, INK2, GRID, KN, SY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8.5, "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "axes.linewidth": 0.6})

fig, axs = plt.subplots(3, 1, figsize=(3.35, 5.3), sharey=True)
for ax, (key, title) in zip(axs, PANELS):
    rs = D[key]; x = np.arange(len(rs))
    kn, sy = np.array([r["knows"] for r in rs]), np.array([r["says"] for r in rs])
    ax.fill_between(x, sy, kn, color=KN, alpha=0.10, lw=0, zorder=0)
    for v, key2, col, mk, ls, off, fc in ((kn, "knows", KN, "o", "-", -0.05, KN), (sy, "says", SY, "s", (0, (3, 1.5)), 0.05, "white")):
        lo = v - np.array([r[f"{key2}_ci"][0] for r in rs]); hi = np.array([r[f"{key2}_ci"][1] for r in rs]) - v
        ax.errorbar(x + off, v, yerr=[lo, hi], color=col, ls="none", elinewidth=0.8, capsize=0, zorder=2)
        ax.plot(x + off, v, color=col, ls=ls, lw=1.6, marker=mk, ms=4.5, mfc=fc, mec=col, mew=1.2, zorder=3)
    for i, r in enumerate(rs):
        if not (r["gap_ci"][0] <= 0 <= r["gap_ci"][1]):
            ax.text(i, r["knows_ci"][1] + 0.008, "*", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([str(r["epoch"]) for r in rs[:-1]] + [f"{rs[-1]['epoch']}\n(final)"])
    ax.set_xlim(-0.4, len(rs) - 0.6); ax.set_title(title, fontsize=9, color=INK, loc="left", pad=4)
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8, length=2, width=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
axs[0].set_ylim(0.5, 0.865); axs[0].set_yticks([0.5, 0.6, 0.7, 0.8])   # axis floor = 0.5 = chance, no break
axs[-1].set_xlabel("epoch", fontsize=8, color=INK2, labelpad=1)
for ax in axs:
    ax.set_ylabel("W vs I AUC", fontsize=8, color=INK2)
h = [plt.Line2D([], [], color=KN, lw=1.6, marker="o", ms=4.5, mec=KN, mfc=KN, label="knows (likelihood)"),
     plt.Line2D([], [], color=SY, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=4.5, mec=SY, mfc="white", label="says (generated report)")]
fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.005), handlelength=2.2)
fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.9)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig2_knows_says.{ext}"), dpi=300)
print("saved figures/fig2_knows_says_vertical.{pdf,png}")
