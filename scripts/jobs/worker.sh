#!/usr/bin/env bash
# One worker per GPU. Takes the first job in $WORK/jobs.tsv that is not done, not taken by another
# worker, and whose required job is done. Started by run_all.sh; do not run by hand.
#   worker.sh <gpu>
GPU=$1
source "$HOME/lq2hq/config.sh"
PORT=$((12400 + GPU))
mkdir -p "$WORK/done" "$WORK/claim" "$WORK/failed" "$WORK/logs"
cd "$BBDM" || exit 1

run_job() {  # run_job <name> <kind> <args...>
  local name=$1 kind=$2; shift 2
  local log="$WORK/logs/$name.log"
  if [ "$kind" = train ]; then
    local arm=$1
    CUDA_VISIBLE_DEVICES=$GPU "$PY" -u main.py --config "configs/OCTA500_3mm_P200_${arm}.yaml" \
        --train --gpu_ids 0 --port "$PORT" --result_path "$RES" > "$log" 2>&1 < /dev/null || return 1
    ls "$RES/OCTA500_3mm_P200_${arm}/LBBDM-f4/checkpoint/"*model*.pth > /dev/null 2>&1 || return 1
  else  # test <RUN> <chunk> <cond> <model>
    local run=$1 chunk=$2 cond=$3 model=$4
    local cfg
    cfg=$("$PY" "$HOME/lq2hq/scripts/make_test_cfg.py" "$run" "$chunk" "$cond" "$model" 2>> "$log") || return 1
    CUDA_VISIBLE_DEVICES=$GPU "$PY" -u main.py --config "$cfg" --sample_to_eval --gpu_ids 0 \
        --port "$PORT" --result_path "$RES" >> "$log" 2>&1 < /dev/null || return 1
    # verify: one output per input patch
    local data="$DATA/OCTA500_3mm_VOL_LQ${cond}_$chunk"
    local n_in n_out
    n_in=$(ls "$data/test/A" | wc -l)
    n_out=$(ls "$RES/OCTA500_3mm_VOL_${run}_${chunk}_TEST/LBBDM-f4/sample_to_eval/200" 2>/dev/null | wc -l)
    echo "outputs: $n_out / $n_in" >> "$log"
    [ "$n_out" -ge "$n_in" ] || return 1
  fi
}

echo "worker GPU $GPU started $(date)"
while true; do
  picked=""; open=0
  while IFS=$'\t' read -r name req cmd; do
    [ -z "$name" ] && continue
    [ "$req" = "-" ] && req=""
    [ -f "$WORK/done/$name" ] || [ -f "$WORK/failed/$name" ] && continue
    if [ -n "$req" ] && [ -f "$WORK/failed/$req" ]; then        # dependency failed -> this cannot run
      echo "blocked by failed $req" > "$WORK/failed/$name"; continue
    fi
    open=1
    [ -n "$req" ] && [ ! -f "$WORK/done/$req" ] && continue      # waiting for training
    mkdir "$WORK/claim/$name" 2>/dev/null || continue             # taken by another worker
    picked="$name"; job="$cmd"; break
  done < "$WORK/jobs.tsv"

  if [ -n "$picked" ]; then
    echo "[GPU $GPU] start $picked  $(date '+%m-%d %H:%M')"
    t0=$(date +%s)
    # shellcheck disable=SC2086
    if run_job "$picked" $job; then
      echo "$(( ($(date +%s) - t0) / 60 )) min on GPU $GPU" > "$WORK/done/$picked"
      echo "[GPU $GPU] done  $picked  ($(( ($(date +%s) - t0) / 60 )) min)"
    else
      echo "see $WORK/logs/$picked.log" > "$WORK/failed/$picked"
      echo "[GPU $GPU] FAILED $picked -> $WORK/logs/$picked.log"
    fi
    rmdir "$WORK/claim/$picked" 2>/dev/null
  elif [ "$open" = 0 ]; then
    echo "worker GPU $GPU: nothing left $(date)"; exit 0
  else
    sleep 120   # everything left is running elsewhere or waiting for a training job
  fi
done
