import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
im = np.asarray(Image.open("bscan152_eye10451.png").convert("L"))
ys = [(216, 539), (564, 887), (912, 1235)]
tiles = [im[a + 1:b, 52:374] for a, b in ys]
T = ["HR (original)", "×2", "×4"]
S = ["(256×256), unchanged", "(256×256) → (128×128) → (256×256)", "(256×256) → (64×64) → (256×256)"]
INK, INK2 = "#1f1f1e", "#5f5e5a"
h, w = tiles[0].shape
zy, zx, zs = int(h * 0.30), int(w * 0.05), int(w * 0.28)   # zoom box: inner retina layers, left side
fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.4), dpi=200, gridspec_kw={"height_ratios": [1, 1]})
for j, t in enumerate(tiles):
    ax[0, j].imshow(t, cmap="gray", vmin=0, vmax=255)
    ax[0, j].add_patch(Rectangle((zx, zy), zs, zs, fill=False, ec="#2a78d6", lw=1.6))
    ax[0, j].set_title(f"{T[j]}\n", fontsize=13, fontweight="bold", color=INK, pad=2)
    ax[0, j].text(0.5, 1.015, S[j], transform=ax[0, j].transAxes, ha="center", va="bottom", fontsize=9, color=INK2)
    ax[1, j].imshow(t[zy:zy + zs, zx:zx + zs], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for s in ax[1, j].spines.values():
        s.set_edgecolor("#2a78d6"); s.set_linewidth(1.6)
    for a in ax[:, j]:
        a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel("OCT input patch\n(256×256)", fontsize=11, color=INK)
ax[1, 0].set_ylabel("zoom of blue box", fontsize=11, color=INK)
fig.text(0.5, 0.012, "Eye 10451, B-scan 152. Each OCT patch is shrunk by area averaging, then enlarged back with bicubic "
         "interpolation,\nso the size stays (256×256) but fine detail is lost. Input PSNR vs original: ×2 ≈ 32.6 dB, "
         "×4 ≈ 29.7 dB. The OCTA target is never changed.", ha="center", va="bottom", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=1.2)
fig.savefig("lq_input_example.png", facecolor="white")
print("ok")
