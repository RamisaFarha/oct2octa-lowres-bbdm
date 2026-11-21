#!/usr/bin/env python3
"""Phase 1: training configs for the fine-tuned models, copied line-by-line from the ROITOP config.

Changes ONLY: dataset_name, dataset_path, n_epochs (8 + FT_EPOCHS), save/validation interval (every
epoch), and adds model_load_path = ROITOP latest_model_8.pth (BBDM then resumes at epoch 8, so
n_epochs = 8 + FT_EPOCHS means FT_EPOCHS extra epochs). Comments and every other setting are kept.

Writes ~/BBDM/configs/OCTA500_3mm_P200_LQX2.yaml, _LQX4.yaml, _LQMIX.yaml

Usage: source ~/lq2hq/config.sh && $PY ~/lq2hq/scripts/02_make_train_configs.py [--force]
"""
import argparse
import os
import re
import sys

BBDM = os.environ.get("BBDM", os.path.expanduser("~/BBDM"))
DATA = os.environ.get("DATA", "/data/BBDM_data")
BASE = os.environ.get("ROITOP_TRAIN_CFG", f"{BBDM}/configs/OCTA500_3mm_P200_ROITOP.yaml")
CKPT = os.environ["ROITOP_CKPT"]
FT = int(os.environ.get("FT_EPOCHS", "3"))
BASE_EPOCHS = 8


def sub1(pattern, repl, text, label):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"Could not find '{label}' in {BASE}; edit the config by hand.")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    base = open(BASE).read()
    if re.search(r"^\s*model_load_path:", base, flags=re.M):
        sys.exit("Base config already has model_load_path; expected the ROITOP *training* config.")

    for arm in ("LQX2", "LQX4", "LQMIX"):
        t = base
        t = sub1(r"(^\s*dataset_name:\s*).*$", rf"\1'OCTA500_3mm_P200_{arm}'", t, "dataset_name")
        t = sub1(r"(^\s*dataset_path:\s*).*$", rf"\1'{DATA}/OCTA500_3mm_P200_{arm}_256'", t, "dataset_path")
        t = sub1(r"(^\s*n_epochs:\s*).*$", rf"\g<1>{BASE_EPOCHS + FT}", t, "n_epochs")
        t = sub1(r"(^\s*save_interval:\s*).*$", r"\g<1>1", t, "save_interval")
        t = sub1(r"(^\s*validation_interval:\s*).*$", r"\g<1>1", t, "validation_interval")
        t = sub1(r"^(\s*)(only_load_latent_mean_std:.*)$",
                 rf"\1\2\n\n\1# LQ fine-tuning: start from ROITOP (resumes at epoch {BASE_EPOCHS}, step counter restored)\n"
                 rf"\1model_load_path: '{CKPT}'", t, "only_load_latent_mean_std")
        dst = f"{BBDM}/configs/OCTA500_3mm_P200_{arm}.yaml"
        if os.path.exists(dst) and not args.force:
            sys.exit(f"{dst} exists; use --force to overwrite")
        open(dst, "w").write(t)
        print(f"wrote {dst}")

    print(f"\nCheck one with:  diff {BASE} {BBDM}/configs/OCTA500_3mm_P200_LQX4.yaml")
    print(f"Expected: dataset_name, dataset_path, n_epochs {BASE_EPOCHS}->{BASE_EPOCHS + FT}, "
          "save_interval, validation_interval, + model_load_path")


if __name__ == "__main__":
    main()
