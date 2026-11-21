import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
MRK = {"A": "o", "B": "s", "C": "^"}
LAB = {"A": "A: ROITOP, no retraining", "B": "B: fine-tuned on its own level", "C": "C: fine-tuned on mixed quality"}
X = {"HR": 0, "x2": 1, "x4": 2}
OFF = {"A": -0.06, "B": 0.0, "C": 0.06}   # small horizontal offset so overlapping points stay visible

# slab en-face, 50 eyes, baseline evaluation (eval_volume_lq.py)
data = {
 "large": {"A": {"HR": .6147, "x2": .5864, "x4": .5277}, "B": {"x2": .5910, "x4": .5284}, "C": {"x2": .5879, "x4": .5261}},
 "cap":   {"A": {"HR": .2888, "x2": .3031, "x4": .2917}, "B": {"x2": .2961, "x4": .2733}, "C": {"x2": .2976, "x4": .2724}},
 "bvc":   {"A": {"HR": .4216, "x2": 2.0300, "x4": 1.9137}, "B": {"x2": 2.1134, "x4": 2.0406}, "C": {"x2": 2.1165, "x4": 2.0202}},
 "psnr":  {"A": {"HR": 23.834, "x2": 23.362, "x4": 22.755}, "B": {"x2": 23.822, "x4": 23.832}, "C": {"x2": 23.771, "x4": 23.738}},
}
panels = [
 ("large", "Large-vessel Dice (higher = better)", "drops 14% from HR to ×4, retraining doesn't recover it", (0.48, 0.64), None),
 ("cap", "Capillary Dice (higher = better)", "close to chance at every input quality", (0.18, 0.32), [(0.2627, "pure noise 0.263"), (0.1998, "another eye 0.200")]),
 ("bvc", "Vessel caliber error, BVC (µm, lower = better)", "error ~5× higher from ×2 on: vessels drawn too thick", (0, 2.4), None),
 ("psnr", "3D PSNR (dB, higher = better)", "fine-tuning restores PSNR, but not the vessels", (22.5, 24.1), None),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(2, 2, figsize=(13.33, 7.2), dpi=200)
fig.patch.set_facecolor("white")
for ax, (key, title, finding, ylim, refs) in zip(axes.flat, panels):
    ax.set_facecolor("white")
    for m in ("A", "B", "C"):
        pts = data[key][m]
        xs = [X[k] + OFF[m] for k in pts]; ys = list(pts.values())
        ax.plot(xs, ys, color=COL[m], lw=2, marker=MRK[m], ms=8, mec="white", mew=1.5, label=LAB[m], zorder=3)
    if refs:
        for y, txt in refs:
            ax.axhline(y, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(2.92, y + 0.003, txt, va="bottom", ha="right", fontsize=9.5, color=INK2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["HR\n(256×256)", "×2\nvia (128×128)", "×4\nvia (64×64)"])
    ax.set_xlim(-0.3, 2.3 if not refs else 2.95)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.02, finding, transform=ax.transAxes, fontsize=10, color=INK2, va="bottom")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.995))
fig.text(0.5, 0.005, "Input OCT patch quality  ·  50 test eyes, slab en-face (304×256), baseline evaluation  ·  "
         "HR point = ROITOP (baseline baseline)", ha="center", fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.94), h_pad=2.2, w_pad=3)
fig.savefig("results_by_resolution.png", facecolor="white")
print("saved")
