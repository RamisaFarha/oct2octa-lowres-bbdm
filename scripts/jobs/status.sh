#!/usr/bin/env bash
# Progress of the job queue.   bash ~/lq2hq/scripts/status.sh
source "$HOME/lq2hq/config.sh"
total=$(grep -c . "$WORK/jobs.tsv")
done_n=$(ls "$WORK/done" 2>/dev/null | wc -l)
fail_n=$(ls "$WORK/failed" 2>/dev/null | wc -l)
echo "Jobs: $done_n done, $fail_n failed, $total total"
echo "Running:"; for c in "$WORK"/claim/*; do [ -e "$c" ] && echo "  $(basename "$c")"; done
for c in "$WORK"/claim/*; do   # progress of running test jobs
  [ -e "$c" ] || continue; n=$(basename "$c")
  case $n in train_*) arm=${n#train_}; echo "  $n: $(tr '\r' '\n' < "$WORK/logs/$n.log" | grep -E 'epoch|it/s|s/it' | tail -1)";;
    *) run=${n%_*}; ch=${n##*_}; d="$RES/OCTA500_3mm_VOL_${run}_${ch}_TEST/LBBDM-f4/sample_to_eval/200"
       echo "  $n: $(ls "$d" 2>/dev/null | wc -l) images so far";;
  esac
done
[ "$fail_n" -gt 0 ] && { echo "Failed:"; for f in "$WORK"/failed/*; do echo "  $(basename "$f"): $(cat "$f")"; done; }
echo "Workers:"; any=0; for p in "$WORK"/pids/*; do [ -e "$p" ] && kill -0 "$(cat "$p")" 2>/dev/null && { echo "  $(basename "$p") running (pid $(cat "$p"))"; any=1; }; done; [ $any = 1 ] || echo "  none running"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU /'
