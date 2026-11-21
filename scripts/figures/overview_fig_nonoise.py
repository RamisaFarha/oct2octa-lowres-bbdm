import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [  # group, label, large dice, width err
 ("Reference", "A, HR input (50 eyes)", .615, .42),
 ("Low resolution", "A, ×2", .586, 2.03), ("Low resolution", "A, ×4", .528, 1.91),
 ("Retrained", "B ×2, ×2 input", .591, 2.11), ("Retrained", "B ×4, ×4 input", .528, 2.04),
 ("Retrained", "C mixed, HR input", .564, 2.10), ("Retrained", "C mixed, ×2", .588, 2.12), ("Retrained", "C mixed, ×4", .526, 2.02),
 ("Retrained", "B ×2, HR input", .559, 2.11), ("Retrained", "B ×4, HR input", .441, 1.72),
 ("Retrained", "2.5D, HR input", .566, 2.38),
 ("Cross-resolution", "B ×2, ×4 input", .543, 2.03), ("Cross-resolution", "B ×4, ×2 input", .473, 1.83),
]
INK, INK2, GREY, ACC, GRID = "#1f1f1e", "#5f5e5a", "#a9a8a2", "#2a78d6", "#e6e5e0"
ys, y, grp_pos, prev = [], 0, [], None
for g, *_ in rows:
    if g != prev:
        if prev is not None: y += 0.8
        grp_pos.append((g, y)); prev = g
    ys.append(y); y += 1
fig, axs = plt.subplots(1, 2, figsize=(13, 8.2), dpi=200, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
for ax, idx, title, unit, ref, fmt in [(axs[0], 2, "Large-vessel Dice (higher = better)", "", None, "{:.3f}"),
                                       (axs[1], 3, "Vessel width error, µm (lower = better)", "", None, "{:.2f}")]:
    for r, yy in zip(rows, ys):
        ref_row = r[1].startswith("A, HR")
        ax.barh(yy, r[idx], height=0.68, color=ACC if ref_row else GREY, zorder=2)
        ax.text(r[idx] + (0.008 if idx == 2 else 0.06), yy, fmt.format(r[idx]), va="center", fontsize=9, color=INK)
    ax.invert_yaxis(); ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
axs[0].set_yticks(ys); axs[0].set_yticklabels([r[1] for r in rows], fontsize=9.5, color=INK)
axs[1].set_yticks(ys); axs[1].set_yticklabels([])
axs[0].set_xlim(0, 0.72); axs[1].set_xlim(0, 2.8)
axs[0].axvline(0.105, color="#c0392b", ls=":", lw=1.2, zorder=3)
axs[0].text(0.11, ys[-1] + 0.62, "another person's real OCTA: 0.105", color="#a33025", fontsize=8.5, va="top")
for g, yy in grp_pos:
    axs[0].text(-0.43, yy - 0.75, g, transform=axs[0].get_yaxis_transform(), fontsize=10, fontweight="bold", color=INK2)
fig.suptitle("Only Model A on full-quality OCT keeps large vessels and their width; every degraded or retrained run is worse",
             x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
fig.text(0.01, 0.955, "En-face slab (304×256), mean over eyes. Blue = reference.",
         fontsize=9, color=INK2)
fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.04)
fig.savefig("overview_no_noise.png", facecolor="white")
print("ok")
