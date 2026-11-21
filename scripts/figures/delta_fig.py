import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=[r for r in csv.DictReader(open("results/per_eye_all_runs.csv")) if r["projection"]=="slab"]
def get(m,k): return {int(r["patient"]):float(r[k]) for r in R if r["method"]==m}
INK, INK2, GREY, ACC, NEG, GRID = "#1f1f1e", "#5f5e5a", "#a9a8a2", "#2a78d6", "#c0503a", "#e6e5e0"
fig, axs = plt.subplots(2, 2, figsize=(12, 7.6), dpi=200, sharey="row")
panels = [("large_vessel_dice", "×2", "LQAX2", "LQBX2"), ("large_vessel_dice", "×4", "LQAX4", "LQBX4"),
          ("capillary_dice", "×2", "LQAX2", "LQBX2"), ("capillary_dice", "×4", "LQAX4", "LQBX4")]
for ax, (k, f, a, b) in zip(axs.ravel(), panels):
    A, B = get(a, k), get(b, k); p = sorted(A); d = np.array([B[i] - A[i] for i in p])
    lim = 0.08 if k.startswith("large") else 0.04
    bins = np.linspace(-lim, lim, 33)
    n, edges, patches = ax.hist(d, bins=bins, color=GREY, edgecolor="white", linewidth=1, zorder=2)
    for pt, left in zip(patches, edges[:-1]):
        pt.set_facecolor(ACC if left >= 0 else NEG); pt.set_alpha(0.85)
    ax.axvline(0, color=INK, lw=1, zorder=3)
    ax.axvline(d.mean(), color=INK, lw=1.5, ls="--", zorder=3)
    up = int(np.sum(d > 0))
    name = "Large-vessel" if k.startswith("large") else "Capillary"
    ax.set_title(f"{name} Dice, {f} input", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.text(0.98, 0.96, f"ResAdapt better: {up}/50 eyes\nResAdapt worse: {50-up}/50 eyes\nmean Δ = {d.mean():+.3f} (dashed)",
            transform=ax.transAxes, va="top", ha="right", fontsize=9.5, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=2))
    o = np.argsort(d)
    for idx, lab in ((o[0], "worst"), (o[-1], "best")):
        ax.annotate(f"{lab}: eye {p[idx]}\n{d[idx]:+.3f}", (d[idx], 0.3), xytext=(d[idx], max(n) * 0.5),
                    ha="center", fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(-lim, lim); ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlabel("ΔDice = ResAdapt − FullRes (same eye, same input)", fontsize=9.5, color=INK2)
axs[0, 0].set_ylabel("number of eyes", color=INK2); axs[1, 0].set_ylabel("number of eyes", color=INK2)
fig.suptitle("Resolution-adapted fine-tuning (ResAdapt-BBDM) barely changes large vessels and lowers capillary Dice in almost every eye",
             x=0.01, ha="left", fontsize=13, color=INK)
fig.text(0.01, 0.935, "Per-eye change in Dice, ResAdapt-BBDM (fine-tuned on that resolution) minus FullRes-BBDM, 50 test eyes, "
         "en-face slab (304×256). Blue = ResAdapt better, red = ResAdapt worse.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2)
fig.savefig("figures/delta_dice_hist.png", facecolor="white")
print("ok")
