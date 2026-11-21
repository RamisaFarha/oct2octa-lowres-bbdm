#!/usr/bin/env bash
# Stop all workers and the BBDM jobs they started. Finished jobs stay done; run_all.sh continues later.
source "$HOME/lq2hq/config.sh"
for p in "$WORK"/pids/*; do
  [ -e "$p" ] || continue
  pid=$(cat "$p")
  pkill -P "$pid" 2>/dev/null          # the worker's running job (and its children)
  kill "$pid" 2>/dev/null && echo "stopped $(basename "$p") (pid $pid)"
  rm -f "$p"
done
sleep 2
pkill -f "main.py --config .*OCTA500_3mm_(P200_LQ|VOL_LQ)" && echo "stopped leftover BBDM jobs"
echo "Partly finished jobs restart from the beginning next time."
