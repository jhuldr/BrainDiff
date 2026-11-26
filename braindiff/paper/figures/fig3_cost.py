"""Paper Fig. 3 - reader change score (external change-reference pairs) vs visual tokens per pair; marker area ~ parameters.
* = BrainDiff better, paired 95% CI excludes 0. Font: matplotlib's bundled cmr10 / cmb10 (no LaTeX needed).
Reads fig3_cost.json.
    python paper/figures/fig3_cost.py   -> paper/figures/fig3_cost.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
HERE = os.path.dirname(os.path.abspath(__file__))

FD = os.path.join(matplotlib.get_data_path(), "fonts/ttf")
for f in ("cmr10.ttf", "cmb10.ttf"):
    fm.fontManager.addfont(os.path.join(FD, f))
D = json.load(open(os.path.join(HERE, "fig3_cost.json")))
INK, INK2, GRID, OURS, MG, MG4, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#8a5cc2", "#8a8984"
NV = "#d0464f"
plt.rcParams.update({"font.size": 8, "font.family": "cmr10", "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                     "axes.unicode_minus": False, "pdf.fonttype": 42, "axes.linewidth": 0.6})
B = {"family": "cmb10"}
# key, params (B), colour
S = [("ours", 14.95, OURS), ("nv", 14.93, NV), ("q8a", 8.77, GRAY), ("q8b", 8.77, GRAY), ("mg27a", 27.43, MG), ("mg27b", 27.43, MG),
     ("mgvol", 4.30, MG4)]
# text, anchor key, offset (pts), ha, size, bold
LAB = [("BrainDiff (ours)", "ours", (9, 0), "left", 8, True), ("NeuroVFM*", "nv", (-2, -13), "left", 7.5, True),
       ("MedGemma-27B", "mg27b", (10, 6), "left", 7.5, True), ("48 slices", "mg27b", (0, -12), "center", 6.8, False),
       ("8 slices*", "mg27a", (-12, 0), "right", 6.8, False), ("MedGemma-4B", "mgvol", (7, 5), "left", 7.5, True),
       ("volume*", "mgvol", (7, -5), "left", 6.8, False),
       ("Qwen3-VL-8B", "q8a", (-9, -13), "right", 7.5, True), ("8 slices*", "q8a", (-9, -2), "right", 6.8, False),
       ("48 slices*", "q8b", (0, -12), "center", 6.8, False)]
size = lambda p: 6.5 * p
DODGE = 1.45   # FT point drawn at tokens / DODGE (same token count as Qwen3-VL-8B 8 slices)
D["q8ft"]["x"] = D["q8ft"]["tokens"] / DODGE

fig, ax = plt.subplots(figsize=(3.35, 2.75))
for a, b, c in (("q8a", "q8b", GRAY), ("mg27a", "mg27b", MG)):
    ax.plot([D[a]["tokens"], D[b]["tokens"]], [D[a]["mean"], D[b]["mean"]], color=c, lw=0.7, ls=(0, (2, 2)), zorder=1)
for k, p, c in S:
    ax.scatter(D[k]["tokens"], D[k]["mean"], s=size(p), color=c, edgecolor="white", linewidth=0.9, zorder=3)
ax.scatter(D["q8ft"]["x"], D["q8ft"]["mean"], s=size(8.77), color=INK2, edgecolor="white", linewidth=0.9, zorder=3)
ax.annotate("Qwen3-VL-8B FT*", (D["q8ft"]["x"], D["q8ft"]["mean"]), xytext=(D["q8ft"]["x"], 3.015), textcoords="data",
            fontsize=7.5, color=INK2, ha="center", va="center", fontfamily="cmb10",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=3, shrinkB=4))
NAMECOL = {"nv": NV, "ours": OURS, "mg27a": MG, "mg27b": MG, "mgvol": MG4}
for t, k, off, ha, fs, bold in LAB:
    col = INK if not bold else NAMECOL.get(k, INK2)
    ax.annotate(t, (D[k]["tokens"], D[k]["mean"]), xytext=off, textcoords="offset points", fontsize=fs, color=col, ha=ha, va="center",
                fontfamily="cmb10" if bold else "cmr10")
ax.set_xscale("log"); ax.set_xlim(110, 180000); ax.set_ylim(2.3, 3.25)
ax.set_xlabel("visual tokens per study pair (log scale)", fontsize=8, color=INK2, labelpad=1)
ax.set_ylabel("reader score, change pairs (1-5)", fontsize=7.5, color=INK2)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=7.5, length=2, width=0.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(INK2)
h = [plt.scatter([], [], s=size(p), color=GRAY, edgecolor="white", label=f"{p}B") for p in (4, 15, 28)]
ax.legend(handles=h, title="parameters", title_fontsize=7, loc="upper right", ncol=3, frameon=False, fontsize=7, handletextpad=0.2,
          columnspacing=0.9, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fig3_cost.{ext}"), dpi=300)
print("saved figures/fig3_cost.{pdf,png}")
