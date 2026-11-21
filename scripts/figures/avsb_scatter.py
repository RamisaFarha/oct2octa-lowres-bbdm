import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
R=list(csv.DictReader(open("results/delta_dice_B_minus_A_per_eye.csv")))
INK, INK2, GRID, ACC, NEG = "#1f1f1e", "#5f5e5a", "#e6e5e0", "#2a78d6", "#c0503a"
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), dpi=200, sharey=True)
for ax, f, lab in zip(axs, ["x2", "x4"], ["×2", "×4"]):
    A=np.array([float(r[f"large_A_{f}"]) for r in R]); B=np.array([float(r[f"large_B_{f}"]) for r in R]); y=A-B
    eyes=[r["eye"] for r in R]
    ax.axhline(0, color=INK, lw=1, zorder=1)
    ax.scatter(A[y<0], y[y<0], s=34, color=ACC, edgecolor="white", lw=0.6, zorder=3, label="ResAdapt better (below 0)")
    ax.scatter(A[y>=0], y[y>=0], s=34, color=NEG, edgecolor="white", lw=0.6, zorder=3, label="ResAdapt worse (above 0)")
    lr=linregress(A, y); xs=np.linspace(0.22, 0.76, 50)
    # 95% band of the fit (bootstrap over eyes)
    rng=np.random.default_rng(0); bs=[]
    for _ in range(2000):
        i=rng.integers(0, len(A), len(A)); l=linregress(A[i], y[i]); bs.append(l.intercept+l.slope*xs)
    lo, hi=np.percentile(bs, [2.5, 97.5], axis=0)
    ax.fill_between(xs, lo, hi, color=INK2, alpha=0.12, lw=0, zorder=2)
    ax.plot(xs, lr.intercept+lr.slope*xs, color=INK, lw=1.6, ls="--", zorder=2)
    rho, p=spearmanr(A, y); k=A>0.3; rho2, p2=spearmanr(A[k], y[k])
    ax.text(0.98, 0.97, f"Spearman ρ = {rho:+.2f} (p = {p:.3f}), n = 50\nwithout eye 10454: ρ = {rho2:+.2f} (p = {p2:.3f})\n"
            f"slope = {lr.slope:+.3f} Dice per unit A Dice",
            transform=ax.transAxes, va="top", ha="right", fontsize=9.5, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=2))
    i=eyes.index("10454")
    ax.annotate("eye 10454\n(vitreous opacities)", (A[i], y[i]), xytext=(A[i]+0.04, y[i]+0.006), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_title(f"{lab} input: large-vessel Dice", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel(f"FullRes-BBDM {lab} large-vessel Dice (per eye, vs real OCTA)", fontsize=10, color=INK2)
    ax.set_xlim(0.22, 0.77); ax.set_ylim(-0.048, 0.105); ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("FullRes Dice − ResAdapt Dice\n(above 0: fine-tuning hurt; below 0: fine-tuning helped)", fontsize=10, color=INK2)
axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 0.80), fontsize=9, frameon=False)
fig.suptitle("Does ResAdapt fine-tuning help more on eyes that FullRes-BBDM already generates well?  50 test eyes, en-face slab (304×256)",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("figures/avsb_vs_A_scatter.png", facecolor="white")
