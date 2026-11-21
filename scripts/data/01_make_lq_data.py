#!/usr/bin/env python3
"""Phase 1: build every low-quality (LQ) dataset, laid out exactly like the baseline ROITOP ROITOP data.

Training data (from $ROITOP_TRAIN_DATA = OCTA500_3mm_P200_ROITOP256):
  $DATA/OCTA500_3mm_P200_LQX2_256   A = x2 OCT,  B = link to original HQ OCTA
  $DATA/OCTA500_3mm_P200_LQX4_256   A = x4 OCT,  B = link
  $DATA/OCTA500_3mm_P200_LQMIX_256  A = each patch at ONE randomly assigned quality (HR / x2 / x4),
                                    so it is the same size as the others (equal training compute)
Test volumes (from $DATA/OCTA500_3mm_VOL_ROITOP_<i>, every chunk in $CHUNKS):
  $DATA/OCTA500_3mm_VOL_LQX2_<i>, ..._LQX4_<i>   test/A = degraded OCT, test/B = link, roi_log.csv copied
  (HR test input = the original ROITOP chunks; nothing to build)

Degradation (proposal definition, per patch): (256x256) -> (128x128) or (64x64) -> bicubic (256x256).
OCTA targets are never modified (symlinks to the originals).

Usage:
  source ~/lq2hq/config.sh
  $PY ~/lq2hq/scripts/01_make_lq_data.py --limit 20      # quick test, then look at the images
  $PY ~/lq2hq/scripts/01_make_lq_data.py                 # full run (about 4 GB of new files)
  $PY ~/lq2hq/scripts/01_make_lq_data.py --speckle       # also SP1/SP2 test chunks (zero-shot only)
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lqlib import degrade, images, link, read_gray, seed_for, write_img  # noqa: E402

DATA = os.environ.get("DATA", "/data/BBDM_data")
SRC_TRAIN = os.environ.get("ROITOP_TRAIN_DATA", f"{DATA}/OCTA500_3mm_P200_ROITOP256")
CHUNKS = os.environ.get("CHUNKS", "0 2 4 8 9 10 11").split()


def build_split(src_split, dst_split, cond, names, quality_rows=None):
    """src_split/{A,B} -> dst_split/{A,B}; A degraded (or linked if cond == 'HR'), B linked."""
    for sub in ("A", "B"):
        os.makedirs(os.path.join(dst_split, sub), exist_ok=True)
    ps, ss = [], []
    for n in names:
        link(os.path.join(src_split, "B", n), os.path.join(dst_split, "B", n))
        src_a = os.path.join(src_split, "A", n)
        dst_a = os.path.join(dst_split, "A", n)
        if cond == "HR":
            link(src_a, dst_a)
            continue
        clean = read_gray(src_a)
        lq = degrade(clean, cond, n)
        write_img(dst_a, lq)
        if quality_rows is not None and len(ps) < 500:  # input-quality check on a sample
            a = clean if clean.ndim == 2 else clean[..., 0]
            b = lq if lq.ndim == 2 else lq[..., 0]
            ps.append(psnr(a, b, data_range=255))
            ss.append(ssim(a, b, data_range=255))
    if quality_rows is not None and ps:
        quality_rows.append({"set": dst_split, "cond": cond, "n_sampled": len(ps),
                             "input_psnr": round(float(np.mean(ps)), 2), "input_ssim": round(float(np.mean(ss)), 4)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="patches per split/chunk (0 = all)")
    ap.add_argument("--speckle", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_test", action="store_true")
    args = ap.parse_args()
    lim = (lambda xs: xs[: args.limit]) if args.limit else (lambda xs: xs)
    q = []

    # ---------------- training data: LQX2, LQX4 ----------------
    if not args.skip_train:
        for cond in ("X2", "X4"):
            dst = f"{DATA}/OCTA500_3mm_P200_LQ{cond}_256"
            for split in ("train", "val", "test"):
                names = lim(images(os.path.join(SRC_TRAIN, split, "A")))
                build_split(os.path.join(SRC_TRAIN, split), os.path.join(dst, split), cond, names,
                            q if split == "val" else None)
                print(f"train-data {cond} {split}: {len(names)}")
            for f in os.listdir(SRC_TRAIN):
                if f.endswith(".csv"):
                    shutil.copy2(os.path.join(SRC_TRAIN, f), dst)

        # ---------------- training data: LQMIX (one quality per patch) ----------------
        dst = f"{DATA}/OCTA500_3mm_P200_LQMIX_256"
        assign = []
        for split in ("train", "val", "test"):
            names = lim(images(os.path.join(SRC_TRAIN, split, "A")))
            for sub in ("A", "B"):
                os.makedirs(os.path.join(dst, split, sub), exist_ok=True)
            for n in names:
                cond = "HR" if split == "test" else ("HR", "X2", "X4")[seed_for(n, "MIX") % 3]
                src_a = (os.path.join(SRC_TRAIN, split, "A", n) if cond == "HR"
                         else f"{DATA}/OCTA500_3mm_P200_LQ{cond}_256/{split}/A/{n}")
                link(src_a, os.path.join(dst, split, "A", n))
                link(os.path.join(SRC_TRAIN, split, "B", n), os.path.join(dst, split, "B", n))
                assign.append((split, n, cond))
            print(f"train-data MIX {split}: {len(names)}")
        with open(os.path.join(dst, "mix_assignment.csv"), "w", newline="") as f:
            csv.writer(f).writerows([("split", "name", "quality")] + assign)
        counts = {c: sum(1 for s, _, cc in assign if s == "train" and cc == c) for c in ("HR", "X2", "X4")}
        print(f"  MIX train quality counts: {counts}")

    # ---------------- test volumes ----------------
    if not args.skip_test:
        # HR = links to the original ROITOP chunk (so the baseline ROITOP folders are never modified)
        conds = ["HR", "X2", "X4"] + (["SP1", "SP2"] if args.speckle else [])
        for cond in conds:
            for i in CHUNKS:
                src = f"{DATA}/OCTA500_3mm_VOL_ROITOP_{i}"
                dst = f"{DATA}/OCTA500_3mm_VOL_LQ{cond}_{i}"
                names = lim(images(os.path.join(src, "test", "A")))
                build_split(os.path.join(src, "test"), os.path.join(dst, "test"), cond, names,
                            q if i == CHUNKS[0] else None)
                # BBDM builds train/val datasets even in test mode: give each one linked patch
                build_split(os.path.join(src, "test"), os.path.join(dst, "train"), "HR", names[:1])
                build_split(os.path.join(src, "test"), os.path.join(dst, "val"), "HR", names[:1])
                shutil.copy2(os.path.join(src, "roi_log.csv"), dst)
                print(f"test-vol {cond} chunk {i}: {len(names)}")

    out = os.path.join(os.environ.get("WORK", "."), "input_quality.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "cond", "n_sampled", "input_psnr", "input_ssim"])
        w.writeheader()
        w.writerows(q)
    print(f"\nInput quality vs. clean OCT -> {out}")
    for r in q:
        print(f"  {r['cond']:<4} PSNR={r['input_psnr']:6.2f} dB  SSIM={r['input_ssim']:.4f}   ({r['set']})")
    print("Check: X4 worse than X2. Then look at a few images before training.")


if __name__ == "__main__":
    main()
