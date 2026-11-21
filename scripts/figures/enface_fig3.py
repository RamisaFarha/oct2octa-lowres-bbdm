import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
A = "/mnt/user-data/attachments/335993e4-7b12-5fab-86a8-1f4f3e641f01/"
en = np.asarray(Image.open(A + "eb1aa8df-image.png").convert("L"))
lv = np.asarray(Image.open(A + "2108d336-image.png").convert("L"))
tile = lambda im, i: im[14:318, i * 262:i * 262 + 256]          # (304x256) tile i
ORDER = ["GT", "ROITOP", "LQAX2", "LQAX4", "LQBX2", "LQBX4", "LQCX2", "LQCX4"]
NAMES = {"GT": ("Real OCTA (GT)", "target"),
         "ROITOP": ("HR input", "Model A (baseline ROITOP)"),
         "LQAX2": ("×2 input", "Model A, no retraining"),
         "LQAX4": ("×4 input", "Model A, no retraining"),
         "LQBX2": ("×2 input", "Model B, fine-tuned ×2"),
         "LQBX4": ("×4 input", "Model B, fine-tuned ×4"),
         "LQCX2": ("×2 input", "Model C, mixed quality"),
         "LQCX4": ("×4 input", "Model C, mixed quality")}
INK, INK2 = "#1f1f1e", "#5f5e5a"
plt.rcParams.update({"font.family": "DejaVu Sans"})

def make(cols, out, width, height=7.6):
    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(width, height), dpi=220)
    for j, c in enumerate(cols):
        i = ORDER.index(c)
        for r, (im, cmap) in enumerate(((en, "gray"), (lv, "gray"))):
            ax = axes[r, j]
            ax.imshow(tile(im, i), cmap=cmap, interpolation="nearest", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#2a78d6" if c == "GT" else "#c9c8c2"); s.set_linewidth(2 if c == "GT" else 0.8)
        t, sub = NAMES[c]
        axes[0, j].set_title(f"{t}\n\n\n", fontsize=13, fontweight="bold", color=INK, pad=2)
        axes[0, j].text(0.5, 1.012, f"{sub}\n(304×256)", transform=axes[0, j].transAxes,
                        ha="center", va="bottom", fontsize=9.5, color=INK2)
    axes[0, 0].set_ylabel("OCTA en-face\n(75 px slab below ILM)", fontsize=11.5, color=INK)
    axes[1, 0].set_ylabel("Large-vessel map\n(Frangi σ 2–4)", fontsize=11.5, color=INK)
    fig.text(0.5, 0.012,
             "One test eye (10451). Each image = 304 generated (256×256) patches stacked into a volume (304×256×256), "
             "then projected over a 75-pixel slab below the ILM.\nRows of each image are B-scans. Large vessels are reproduced at HR "
             "but thin out and break at ×4, with or without fine-tuning (Model B); capillary areas are grainy texture at every input; horizontal streaks = B-scan inconsistency.",
             ha="center", va="bottom", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6, h_pad=0.8)
    fig.savefig(out, facecolor="white")
    print("saved", out)

make(["GT", "ROITOP", "LQAX2", "LQBX2", "LQAX4", "LQBX4"], "enface_fig3_AvsB.png", 15.5, 8.0)
