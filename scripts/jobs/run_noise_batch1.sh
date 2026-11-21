#!/usr/bin/env bash
# run_noise_batch1.sh - Step 4, batch 1 of the calibrated noise study, start to finish:
#   1. build the noisy test data (NL4 NL1 NA4 NA8, all 50 eyes)      CPU, ~1-2 h
#   2. add the 28 Model A test jobs to the queue (skips ones already there)
#   3. start the GPU workers (GPUs from config.sh = 0 1 2; GPU 3 stays free)
#   4. wait until all 28 jobs are done (~43 h); stops if one fails
#   5. evaluate (eval_volume_lq.py) and run vessel continuity (Step 5)
# Start it once, in the background, and log out if you like:
#   nohup bash ~/lq2hq/scripts/run_noise_batch1.sh > ~/noise_batch1.log 2>&1 &
# Watch:   tail -f ~/noise_batch1.log      (Ctrl+C stops watching, not the run)
set -u
source "$HOME/lq2hq/config.sh"
CONDS="NL4 NL1 NA4 NA8"
CHUNKS_ALL="0 2 4 8 9 10 11"
ARMS="ROITOP"; for c in $CONDS; do ARMS="$ARMS LQA$c"; done
say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

# 1. data (skip a condition whose 7 chunks already exist with all patches)
for c in $CONDS; do
  done_all=1
  for i in $CHUNKS_ALL; do
    src=$(ls "$DATA/OCTA500_3mm_VOL_ROITOP_$i/test/A" 2>/dev/null | wc -l)
    dst=$(ls "$DATA/OCTA500_3mm_VOL_LQ${c}_$i/test/A" 2>/dev/null | wc -l)
    [ "$src" -gt 0 ] && [ "$dst" -ge "$src" ] || done_all=0
  done
  if [ $done_all = 1 ]; then say "data $c already built"; continue; fi
  say "building data $c"
  "$PY" "$HOME/lq2hq/scripts/05_make_noise.py" --conds "$c" --chunks $CHUNKS_ALL || { say "data build FAILED for $c"; exit 1; }
done

# 2. jobs
added=0
for c in $CONDS; do for i in $CHUNKS_ALL; do
  name="LQA${c}_$i"
  grep -q "^${name}	" "$WORK/jobs.tsv" || { printf "%s\t-\ttest LQA%s %s %s ROITOP\n" "$name" "$c" "$i" "$c" >> "$WORK/jobs.tsv"; added=$((added+1)); }
done; done
say "added $added jobs"

# 3. workers (fine if they are already running: they pick up new jobs)
bash "$HOME/lq2hq/scripts/run_all.sh" || say "workers already running, they will take the new jobs"

# 4. wait
while true; do
  n_done=0; n_fail=0
  for c in $CONDS; do for i in $CHUNKS_ALL; do
    [ -f "$WORK/done/LQA${c}_$i" ] && n_done=$((n_done+1))
    [ -f "$WORK/failed/LQA${c}_$i" ] && { n_fail=$((n_fail+1)); say "FAILED LQA${c}_$i: $(cat "$WORK/failed/LQA${c}_$i")"; }
  done; done
  [ $n_fail -gt 0 ] && { say "stopping: $n_fail job(s) failed. Fix, delete the file(s) in $WORK/failed, and start this script again."; exit 1; }
  say "$n_done / 28 jobs done"
  [ $n_done -ge 28 ] && break
  sleep 1800
done

# 5. evaluation (CPU)
cd "$HOME/BBDM" || exit 1
say "evaluating"
"$PY" eval_volume_lq.py --arms $ARMS --out eval_noise_batch1 || say "eval_volume_lq FAILED"
say "vessel continuity"
"$PY" vessel_continuity.py --arms $ARMS --workers 8 --out continuity_noise_batch1 || say "vessel_continuity FAILED"
say "ALL DONE. Send: eval_noise_batch1/summary.csv, continuity_noise_batch1/summary.csv and wilcoxon.csv"
