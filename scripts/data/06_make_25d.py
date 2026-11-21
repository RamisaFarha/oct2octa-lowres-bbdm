#!/usr/bin/env python3
"""2.5D slice-consistency test (paper step 1, version 1).

Idea: the encoder takes 3-channel images, and today each OCT patch is grey, so all 3 channels are the same
B-scan. Here the 3 channels are B-scans n-1, n, n+1 of the same eye, cropped with the SAME window as B-scan n
(rows roi_y0..roi_y0+255, columns x0..x0+255 of the original (640x304) B-scan). The model can then see its
neighbours when it generates B-scan n. Targets (OCTA) are unchanged and stay grey.
At the first and last B-scan the missing neighbour is replaced by B-scan n itself.

Builds:
  $DATA/OCTA500_3mm_P200_LQN3_256      training set (train/val/test), A = 3-slice OCT, B = link to OCTA
  $DATA/OCTA500_3mm_VOL_LQN3_<i>       test chunks (all of $CHUNKS), same layout as the other LQ chunks
  ~/BBDM/configs/OCTA500_3mm_P200_LQN3.yaml   training config = ROITOP config + model_load_path (like Model B)

Usage:
  source ~/lq2hq/config.sh
  $PY ~/lq2hq/scripts/06_make_25d.py --check 200    # verify: crops from the raw B-scans match the baseline ROITOP patches
  $PY ~/lq2hq/scripts/06_make_25d.py                # full build (about 15 min) + config
"""
import argparse
import csv
import functools
import glob
import os
import re
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lqlib import images, link, read_gray  # noqa: E402

DATA = os.environ.get("DATA", "/data/BBDM_data")
BBDM = os.environ.get("BBDM", os.path.expanduser("~/BBDM"))
RAW = os.environ.get("RAW", "/data/OCTA-500/OCTA_3mm")
SRC_TRAIN = os.environ.get("ROITOP_TRAIN_DATA", f"{DATA}/OCTA500_3mm_P200_ROITOP256")
BASE_CFG = os.environ.get("ROITOP_TRAIN_CFG", f"{BBDM}/configs/OCTA500_3mm_P200_ROITOP.yaml")
CKPT = os.environ.get("ROITOP_CKPT", "")
CHUNKS = os.environ.get("CHUNKS", "0 2 4 8 9 10 11").split()
FT = int(os.environ.get("FT_EPOCHS", "3"))
NB, C, X0_DEFAULT = 304, 256, 24


@functools.lru_cache(maxsize=2048)
def raw_oct(pid, n):
    p = f"{RAW}/OCT/{pid}/{n}.bmp"
    a = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise IOError(p)
    return a


def read_log(root):
    """name (without extension) -> (x0, y0) from any csv in root with name/roi_y0 columns."""
    out = {}
    for f in glob.glob(f"{root}/*.csv"):
        with open(f) as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or "name" not in rd.fieldnames or "roi_y0" not in rd.fieldnames:
                continue
            for r in rd:
                stem = os.path.splitext(r["name"])[0]
                out[stem] = (int(float(r.get("x0") or X0_DEFAULT)), int(float(r["roi_y0"])))
    return out


def parse(stem):
    m = re.search(r"(\d{5})_0*(\d+)$", stem)
    if not m:
        raise ValueError(f"cannot read eye id and B-scan number from '{stem}'")
    return m.group(1), int(m.group(2))


def three_slice(stem, log):
    pid, n = parse(stem)
    x0, y0 = log[stem]
    sl = []
    for m in (n - 1, n, n + 1):
        m = min(max(m, 1), NB)
        sl.append(raw_oct(pid, m)[y0:y0 + C, x0:x0 + C])
    return sl          # [prev, centre, next], each (256x256)


def write_rgb(path, sl):
    # cv2 writes BGR; reversing makes the file read as RGB = (prev, centre, next)
    if not cv2.imwrite(path, np.stack(sl[::-1], axis=2)):
        raise IOError(path)


def build(src_root, src_split, dst_root, dst_split, names, log, check):
    for sub in ("A", "B"):
        os.makedirs(f"{dst_root}/{dst_split}/{sub}", exist_ok=True)
    diffs = []
    for name in names:
        stem = os.path.splitext(name)[0]
        sl = three_slice(stem, log)
        orig = read_gray(f"{src_root}/{src_split}/A/{name}")
        orig = orig if orig.ndim == 2 else orig[..., 0]
        diffs.append(float(np.abs(orig.astype(int) - sl[1].astype(int)).mean()))
        if check:
            continue
        write_rgb(f"{dst_root}/{dst_split}/A/{name}", sl)
        link(f"{src_root}/{src_split}/B/{name}", f"{dst_root}/{dst_split}/B/{name}")
    return diffs


def report(label, diffs):
    d = np.array(diffs)
    print(f"  {label}: {len(d)} patches, centre-slice vs baseline patch mean |diff| = {d.mean():.3f} grey levels "
          f"(max {d.max():.2f})")
    if d.mean() > 1.0:
        sys.exit("  MISMATCH: the raw crop does not reproduce the baseline ROITOP patches. Stop and send this output.")


def make_config():
    t = open(BASE_CFG).read()
    if re.search(r"^\s*model_load_path:", t, flags=re.M):
        sys.exit(f"{BASE_CFG} already has model_load_path; expected the ROITOP training config.")

    def sub1(p, r):
        nonlocal t
        t, k = re.subn(p, r, t, count=1, flags=re.M)
        if k != 1:
            sys.exit(f"pattern not found in {BASE_CFG}: {p}")
    sub1(r"(^\s*dataset_name:\s*).*$", r"\1'OCTA500_3mm_P200_LQN3'")
    sub1(r"(^\s*dataset_path:\s*).*$", rf"\1'{DATA}/OCTA500_3mm_P200_LQN3_256'")
    sub1(r"(^\s*n_epochs:\s*).*$", rf"\g<1>{8 + FT}")
    sub1(r"(^\s*save_interval:\s*).*$", r"\g<1>1")
    sub1(r"(^\s*validation_interval:\s*).*$", r"\g<1>1")
    sub1(r"^(\s*)(only_load_latent_mean_std:.*)$",
         rf"\1\2\n\n\1# 2.5D fine-tuning: start from ROITOP (resumes at epoch 8)\n\1model_load_path: '{CKPT}'")
    dst = f"{BBDM}/configs/OCTA500_3mm_P200_LQN3.yaml"
    open(dst, "w").write(t)
    print(f"wrote {dst}  (diff it against {BASE_CFG})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0, help="only verify N patches per set, write nothing")
    args = ap.parse_args()
    lim = (lambda xs: xs[: args.check]) if args.check else (lambda xs: xs)

    log_tr = read_log(SRC_TRAIN)
    if not log_tr:
        sys.exit(f"no roi_log csv with name/roi_y0 in {SRC_TRAIN}")
    dst = f"{DATA}/OCTA500_3mm_P200_LQN3_256"
    for split in ("train", "val", "test"):
        names = lim(images(f"{SRC_TRAIN}/{split}/A"))
        report(f"train-set {split}", build(SRC_TRAIN, split, dst, split, names, log_tr, args.check))
    if not args.check:
        for f in glob.glob(f"{SRC_TRAIN}/*.csv"):
            shutil.copy2(f, dst)

    for i in CHUNKS:
        src = f"{DATA}/OCTA500_3mm_VOL_ROITOP_{i}"
        dchunk = f"{DATA}/OCTA500_3mm_VOL_LQN3_{i}"
        log = read_log(src)
        names = lim(images(f"{src}/test/A"))
        report(f"test chunk {i}", build(src, "test", dchunk, "test", names, log, args.check))
        if not args.check:
            for split in ("train", "val"):
                build(src, "test", dchunk, split, names[:1], log, False)
            shutil.copy2(f"{src}/roi_log.csv", dchunk)

    if args.check:
        print("\nCheck passed. Run again without --check to build.")
        return
    make_config()
    print("\nDone. Next: check that BBDM reads the 3 channels (step 3c in the instructions).")


if __name__ == "__main__":
    main()
