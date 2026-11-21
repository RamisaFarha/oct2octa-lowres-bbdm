#!/usr/bin/env python3
"""Make one test config: OCTA500_3mm_VOL_<RUN>_<chunk>_TEST.yaml, copied from the baseline ROITOP ROITOP test config
of the same chunk. Changes only dataset_name, dataset_path and model_load_path.

  make_test_cfg.py <RUN> <chunk> <data_cond> <model>
    data_cond: HR | X2 | X4 | SP1 | SP2         (HR = linked copy of the ROITOP chunk)
    model:     ROITOP | LQX2 | LQX4 | LQMIX     (fine-tuned: newest checkpoint of that training run)
Prints the config path.
"""
import glob
import os
import re
import sys

BBDM = os.environ.get("BBDM", os.path.expanduser("~/BBDM"))
DATA = os.environ.get("DATA", "/data/BBDM_data")
RES = os.environ.get("RES", "/data/BBDM_results")


def checkpoint(model):
    if model == "ROITOP":
        return os.environ["ROITOP_CKPT"]
    d = f"{RES}/OCTA500_3mm_P200_{model}/LBBDM-f4/checkpoint"
    cands = glob.glob(f"{d}/latest_model_*.pth")
    if cands:  # highest epoch number
        return max(cands, key=lambda p: int(re.search(r"latest_model_(\d+)\.pth", p).group(1)))
    if os.path.exists(f"{d}/last_model.pth"):
        return f"{d}/last_model.pth"
    sys.exit(f"no checkpoint in {d}")


def main():
    run, chunk, cond, model = sys.argv[1:5]
    base_path = f"{BBDM}/configs/OCTA500_3mm_VOL_ROITOP_{chunk}_TEST.yaml"
    t = open(base_path).read()
    data = f"{DATA}/OCTA500_3mm_VOL_LQ{cond}_{chunk}"   # HR = LQHR (links to the ROITOP chunk + train/val)
    ck = checkpoint(model)
    for pat, rep in ((r"(^\s*dataset_name:\s*).*$", rf"\1'OCTA500_3mm_VOL_{run}_{chunk}_TEST'"),
                     (r"(^\s*dataset_path:\s*).*$", rf"\1'{data}'"),
                     (r"(^\s*model_load_path:\s*).*$", rf"\1'{ck}'")):
        t, n = re.subn(pat, rep, t, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"pattern not found in {base_path}: {pat}")
    tb = os.environ.get("TEST_BATCH")
    if tb:  # test batch size = the last batch_size line (train, val, test)
        m = list(re.finditer(r"^(\s*batch_size:\s*)\d+", t, flags=re.M))
        if len(m) < 3:
            sys.exit(f"expected 3 batch_size lines in {base_path}")
        t = t[:m[-1].start()] + m[-1].group(1) + tb + t[m[-1].end():]
    out = f"{BBDM}/configs/OCTA500_3mm_VOL_{run}_{chunk}_TEST.yaml"
    open(out, "w").write(t)
    print(out)
    print(f"  data={data}\n  ckpt={ck}", file=sys.stderr)


if __name__ == "__main__":
    main()
