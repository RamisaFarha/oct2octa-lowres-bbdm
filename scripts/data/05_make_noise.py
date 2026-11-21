#!/usr/bin/env python3
"""Noise-robustness test data (proposal, optional experiment): speckle noise on the OCT input.

Builds test chunks $DATA/OCTA500_3mm_VOL_LQSP1_<i> (light) and ..._LQSP2_<i> (strong), laid out like the
other LQ chunks: test/A = noisy OCT patch (256x256), test/B = link to the real OCTA, 1-patch train/val links,
roi_log.csv copied. OCTA targets are never changed.

Noise model (lqlib.speckle): OCT patch -> linear intensity (40 dB range) -> multiply by gamma noise with L looks
(mean 1) -> back to log. Lower L = stronger noise. Default L = 16 (SP1) and 4 (SP2), aiming at input PSNR of
about 31 and 25 dB.

Usage (from anywhere):
  source ~/lq2hq/config.sh
  $PY ~/lq2hq/scripts/05_make_noise.py --limit 50            # quick check: prints input PSNR / SSIM
  $PY ~/lq2hq/scripts/05_make_noise.py                       # full build, chunks 8 9 10 11 (20 eyes)
  $PY ~/lq2hq/scripts/05_make_noise.py --looks1 12 --looks2 3   # only if PSNR is far from the targets
  $PY ~/lq2hq/scripts/05_make_noise.py --conds NL4 NL1 NA4 NA8 --chunks 0 2 4 8 9 10 11   # calibrated levels, 50 eyes
"""
import argparse
import os
import shutil
import sys

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lqlib  # noqa: E402
from lqlib import degrade, images, link, read_gray, write_img  # noqa: E402

DATA = os.environ.get("DATA", "/data/BBDM_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", default=["8", "9", "10", "11"])
    ap.add_argument("--conds", nargs="+", default=["SP1", "SP2"],
                    help="SP1 SP2 (old speckle) or calibrated levels NL8 NL4 NL2 NL1 NA2 NA4 NA6 NA8 (noisemodel.py)")
    ap.add_argument("--limit", type=int, default=0, help="patches per chunk (0 = all)")
    ap.add_argument("--looks1", type=float, default=lqlib.SPECKLE_LOOKS["SP1"])
    ap.add_argument("--looks2", type=float, default=lqlib.SPECKLE_LOOKS["SP2"])
    args = ap.parse_args()
    lqlib.SPECKLE_LOOKS.update({"SP1": args.looks1, "SP2": args.looks2})
    print(f"looks: SP1 = {args.looks1}, SP2 = {args.looks2}; chunks {args.chunks}")

    for cond in args.conds:
        ps, ss = [], []
        for i in args.chunks:
            src = f"{DATA}/OCTA500_3mm_VOL_ROITOP_{i}"
            dst = f"{DATA}/OCTA500_3mm_VOL_LQ{cond}_{i}"
            names = images(f"{src}/test/A")
            if args.limit:
                names = names[: args.limit]
            for split in ("test", "train", "val"):
                for sub in ("A", "B"):
                    os.makedirs(f"{dst}/{split}/{sub}", exist_ok=True)
            for k, n in enumerate(names):
                clean = read_gray(f"{src}/test/A/{n}")
                noisy = degrade(clean, cond, n)
                write_img(f"{dst}/test/A/{n}", noisy)
                link(f"{src}/test/B/{n}", f"{dst}/test/B/{n}")
                if len(ps) < 300:
                    a = clean if clean.ndim == 2 else clean[..., 0]
                    b = noisy if noisy.ndim == 2 else noisy[..., 0]
                    ps.append(psnr(a, b, data_range=255)); ss.append(ssim(a, b, data_range=255))
            for split in ("train", "val"):   # BBDM builds these even in test mode
                link(f"{src}/test/A/{names[0]}", f"{dst}/{split}/A/{names[0]}")
                link(f"{src}/test/B/{names[0]}", f"{dst}/{split}/B/{names[0]}")
            shutil.copy2(f"{src}/roi_log.csv", dst)
            print(f"  {cond} chunk {i}: {len(names)} patches")
        print(f"{cond}: input PSNR {np.mean(ps):.2f} dB, SSIM {np.mean(ss):.3f}  (vs clean OCT, {len(ps)} patches)")
    print("\nFor reference: x2 input PSNR ~32.6 dB, x4 ~29.7 dB. Aim: SP1 ~31 dB, SP2 ~25 dB.")


if __name__ == "__main__":
    main()
