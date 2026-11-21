#!/usr/bin/env bash
# Start one worker per GPU in the background (survives logout).
#   bash ~/lq2hq/scripts/run_all.sh
source "$HOME/lq2hq/config.sh"
mkdir -p "$WORK/pids" "$WORK/logs"
for p in "$WORK"/pids/*; do
  [ -e "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null && { echo "Workers already running (see $WORK/pids). Use stop_all.sh first."; exit 1; }
done
rm -f "$WORK"/pids/*
rm -rf "$WORK/claim"; mkdir -p "$WORK/claim"     # clear claims left by a stopped run
for g in $GPUS; do
  nohup bash "$HOME/lq2hq/scripts/worker.sh" "$g" > "$WORK/logs/worker_gpu$g.log" 2>&1 < /dev/null &
  echo $! > "$WORK/pids/gpu$g"
  disown
  echo "started worker on GPU $g (pid $!)"
done
echo "Watch: bash ~/lq2hq/scripts/status.sh"
