#!/usr/bin/env bash
# Phase 0: check paths, GPUs, disk, and measure speed from the baseline ROITOP runs you already did.
#   source ~/lq2hq/config.sh && bash ~/lq2hq/scripts/00_check.sh
ok() { [ -e "$2" ] && echo "  [OK]      $1" || echo "  [MISSING] $1  ($2)"; }

echo "== Paths =="
ok "BBDM main.py"          "$BBDM/main.py"
ok "ROITOP train config"   "$ROITOP_TRAIN_CFG"
ok "ROITOP train data"     "$ROITOP_TRAIN_DATA/train/A"
ok "ROITOP checkpoint"     "$ROITOP_CKPT"
ok "VQGAN checkpoint"      "$BBDM/pretrained/vq-f4/model.ckpt"
for i in $CHUNKS; do
  ok "test chunk $i (+roi_log)" "$DATA/OCTA500_3mm_VOL_ROITOP_$i/roi_log.csv"
  ok "test config chunk $i"     "$BBDM/configs/OCTA500_3mm_VOL_ROITOP_${i}_TEST.yaml"
  ok "HR baseline outputs $i"   "$RES/OCTA500_3mm_VOL_ROITOP_${i}_TEST/LBBDM-f4/sample_to_eval/200"
done

echo; echo "== Inference speed (from the baseline ROITOP chunk-0 test run: file timestamps) =="
D=$RES/OCTA500_3mm_VOL_ROITOP_0_TEST/LBBDM-f4/sample_to_eval/200
if [ -d "$D" ]; then
  first=$(find "$D" -name '*.png' -printf '%T@\n' | sort -n | head -1)
  last=$(find "$D" -name '*.png' -printf '%T@\n' | sort -n | tail -1)
  n=$(ls "$D" | wc -l)
  $PY - <<EOF
f, l, n = $first, $last, $n
s = (l - f) / max(n - 1, 1)
print(f"  {n} patches in {(l-f)/3600:.1f} h  ->  {s:.2f} s/patch on one 1080 Ti")
print(f"  one test run (50 eyes, 15,200 patches): {15200*s/3600:.1f} GPU-h")
print(f"  7 new test runs: {7*15200*s/3600:.0f} GPU-h  = {7*15200*s/3600/4:.0f} h on 4 GPUs")
EOF
fi

echo; echo "== Training speed (ROITOP training log) =="
L=$BBDM/logs_p200/p200_roitop_train.log
if [ -f "$L" ]; then
  tr '\r' '\n' < "$L" | grep -E "it/s|s/it" | tail -2
  echo "  ROITOP checkpoint saved: $(date -r "$ROITOP_CKPT")"
  echo "  training log started:    $(stat -c %w "$L" 2>/dev/null | cut -d. -f1)  (may be '-' if unsupported)"
else
  echo "  log not found: $L"
fi

echo; echo "== GPUs =="; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo; echo "== Disk =="; df -h "$DATA" /home | tail -2
