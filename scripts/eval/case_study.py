#!/usr/bin/env python3
"""
case_study.py - Model A vs Model B on selected eyes (Nov 17 meeting): en-face images and colour-coded
large-vessel overlays, for x2 and x4 input.

For each eye one figure, 2 rows (x2 input, x4 input) x 6 columns, every image (304x256):
  OCT input en-face | real OCTA | Model A output | Model B output | Model A overlay | Model B overlay
Overlay colours (large-vessel maps vs the real OCTA):
  white  = vessel in both (correct)
  orange = real vessel the model missed
  blue   = vessel the model drew that is not in the real OCTA (fabricated)

Run from ~/BBDM (needs eval_volume_lq.py and eval_capdice_lq.py):
    python case_study.py --out case_study
    python case_study.py --eyes 10454 10470 --out case_study     # other eyes
Default eyes (from the per-eye Dice differences, large vessels, slab):
    10454 worst (x2 and x4), 10470 best x2, 10492 best x4, 10498 unchanged x2, 10493 unchanged x4
"""
import argparse, glob, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_volume_lq import windows, stack, enface_slab, vessel_features, dice
from eval_capdice_lq import NB, C, pred_index, crop_info, ilm_rows

DATA = os.environ.get("DATA", "/data/BBDM_data")
PAIRS = {"×2": ("LQAX2", "LQBX2", "X2"), "×4": ("LQAX4", "LQBX4", "X4")}
ROLE = {10454: "B worsens most (×2 and ×4)", 10470: "B improves most at ×2", 10492: "B improves most at ×4",
        10498: "A ≈ B at ×2", 10493: "A ≈ B at ×4"}
WHITE, ORANGE, BLUE = (1.0, 1.0, 1.0), (0.95, 0.56, 0.17), (0.18, 0.47, 0.84)


def input_files(cond):
    """name -> path of the OCT input patches (256x256) for this input condition."""
    out = {}
    for p in glob.glob(f"{DATA}/OCTA500_3mm_VOL_LQ{cond}_*/test/A/*"):
        out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def overlay(gen, gt):
    rgb = np.zeros(gt.shape + (3,))
    rgb[gen & gt] = WHITE
    rgb[gt & ~gen] = ORANGE
    rgb[gen & ~gt] = BLUE
    return rgb


def norm(e):
    lo, hi = np.percentile(e, 1), np.percentile(e, 99.5)
    return np.clip((e - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eyes", nargs="+", type=int, default=[10454, 10470, 10492, 10498, 10493])
    ap.add_argument("--depth", type=int, default=75)
    ap.add_argument("--out", default="case_study")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arms = sorted({a for v in PAIRS.values() for a in v[:2]})
    files = {a: pred_index(a) for a in arms}
    info = {a: crop_info(a) for a in arms}
    info_top = crop_info("ROITOP")
    inp = {cond: input_files(cond) for _, _, cond in PAIRS.values()}
    rows_csv = ["eye,input,dice_A,dice_B,delta,missed_A,missed_B,fabricated_A,fabricated_B"]

    for pid in args.eyes:
        y0 = windows(pid, info_top)
        ilm = np.stack([ilm_rows(pid, n) for n in range(1, NB + 1)])
        ilm = np.clip(np.nan_to_num(ilm, nan=np.nanmedian(ilm)) - y0[:, None], 0, C - 1)
        eg = enface_slab(stack(pid, y0), ilm, args.depth)
        _, _, glarge, _ = vessel_features(eg)

        fig, ax = plt.subplots(2, 6, figsize=(16.5, 8.8), dpi=170)
        for r, (lab, (a, b, cond)) in enumerate(PAIRS.items()):
            oct_vol = stack(pid, y0, "LQINPUT", inp[cond], info_top)
            eo = enface_slab(oct_vol, ilm, args.depth)
            ea = enface_slab(stack(pid, y0, a, files[a], info[a]), ilm, args.depth)
            eb = enface_slab(stack(pid, y0, b, files[b], info[b]), ilm, args.depth)
            _, _, la, _ = vessel_features(ea)
            _, _, lb, _ = vessel_features(eb)
            da, db = dice(la, glarge), dice(lb, glarge)
            g = glarge.sum()
            miss = lambda m: (glarge & ~m).sum() / max(g, 1)
            fab = lambda m: (m & ~glarge).sum() / max(m.sum(), 1)
            rows_csv.append(f"{pid},{lab},{da:.4f},{db:.4f},{db - da:+.4f},{miss(la):.3f},{miss(lb):.3f},"
                            f"{fab(la):.3f},{fab(lb):.3f}")
            tiles = [(norm(eo), f"OCT input, {lab}"), (norm(eg), "real OCTA"),
                     (norm(ea), f"Model A, {lab} input"), (norm(eb), f"Model B, {lab} input"),
                     (overlay(la, glarge), f"A overlay  Dice {da:.3f}"),
                     (overlay(lb, glarge), f"B overlay  Dice {db:.3f}  (Δ {db - da:+.3f})".replace("-0.000", "+0.000"))]
            for c, (im, t) in enumerate(tiles):
                ax[r, c].imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=1,
                                interpolation="nearest")
                ax[r, c].set_title(f"{t}\n(304×256)", fontsize=9)
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        fig.suptitle(f"Eye {pid}: {ROLE.get(pid, 'selected eye')}. Large-vessel overlays vs real OCTA: "
                     "white = correct, orange = missed, blue = fabricated", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
        fig.savefig(os.path.join(args.out, f"case_{pid}.png"), facecolor="white")
        plt.close(fig)
        print(f"  saved {args.out}/case_{pid}.png")

    open(os.path.join(args.out, "case_dice.csv"), "w").write("\n".join(rows_csv) + "\n")
    print("\n".join(rows_csv))
    print(f"\nSaved in {args.out}/")


if __name__ == "__main__":
    main()
