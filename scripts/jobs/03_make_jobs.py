#!/usr/bin/env python3
"""Phase 2: write the job list ($WORK/jobs.tsv) in priority order. Workers take jobs top-down.

Runs (test arms, named so they sit next to the baseline ROITOP arms in BBDM_results):
  ROITOP  = model A on HR  (baseline results already exist, not rerun)
  LQAX2/LQAX4   model A (ROITOP, no retraining) on x2 / x4 input
  LQBX2/LQBX4   model B (fine-tuned on x2 / x4)  on its own input
  LQCHR/LQCX2/LQCX4  model C (fine-tuned on mixed quality) on HR / x2 / x4
  (--speckle: LQASP1/2 and LQCSP1/2, zero-shot)

Usage: source ~/lq2hq/config.sh && $PY ~/lq2hq/scripts/03_make_jobs.py [--speckle]
"""
import argparse
import os

CHUNKS = os.environ.get("CHUNKS", "0 2 4 8 9 10 11").split()
WORK = os.environ.get("WORK", "/data/lq2hq_work")

# run name -> (data condition, model, training job it waits for)
RUNS = [
    ("LQAX4", "X4", "ROITOP", ""),
    ("LQBX4", "X4", "LQX4", "train_LQX4"),
    ("LQCX4", "X4", "LQMIX", "train_LQMIX"),
    ("LQAX2", "X2", "ROITOP", ""),
    ("LQCX2", "X2", "LQMIX", "train_LQMIX"),
    ("LQBX2", "X2", "LQX2", "train_LQX2"),
    ("LQCHR", "HR", "LQMIX", "train_LQMIX"),
]
SPECKLE_RUNS = [
    ("LQASP1", "SP1", "ROITOP", ""), ("LQASP2", "SP2", "ROITOP", ""),
    ("LQCSP1", "SP1", "LQMIX", "train_LQMIX"), ("LQCSP2", "SP2", "LQMIX", "train_LQMIX"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speckle", action="store_true")
    args = ap.parse_args()
    jobs = [("train_LQX4", "", "train LQX4"), ("train_LQMIX", "", "train LQMIX"), ("train_LQX2", "", "train LQX2")]
    for run, cond, model, req in RUNS + (SPECKLE_RUNS if args.speckle else []):
        for i in CHUNKS:
            jobs.append((f"{run}_{i}", req, f"test {run} {i} {cond} {model}"))
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, "jobs.tsv")
    with open(path, "w") as f:
        for name, req, cmd in jobs:
            f.write(f"{name}\t{req or '-'}\t{cmd}\n")  # "-" = no requirement
    print(f"{len(jobs)} jobs -> {path}")


if __name__ == "__main__":
    main()
