#!/usr/bin/env bash
# Measure test-time speed: 32 patches at batch 1 (GPU 0) and batch 8 (GPU 1), in parallel.
#   bash ~/lq2hq/scripts/speed_test.sh
source "$HOME/lq2hq/config.sh"
T=$WORK/speed; rm -rf "$T"; mkdir -p "$T/data/test/A" "$T/data/test/B"
SRC=$DATA/OCTA500_3mm_VOL_ROITOP_8
for f in $(ls "$SRC/test/A" | head -32); do
  ln -s "$SRC/test/A/$f" "$T/data/test/A/$f"
  ln -s "$SRC/test/B/$f" "$T/data/test/B/$f"
done

# configs: text edits only (your configs contain python tags that yaml.safe_load refuses)
for b in 1 8; do
  "$PY" - "$BBDM/configs/OCTA500_3mm_VOL_ROITOP_8_TEST.yaml" "$T/b$b.yaml" "$b" "$T/data" << 'PYEOF'
import re, sys
src, dst, b, data = sys.argv[1:5]
t = open(src).read()
t = re.sub(r"(^\s*dataset_name:\s*).*$", rf"\1'LQSPEED_b{b}'", t, count=1, flags=re.M)
t = re.sub(r"(^\s*dataset_path:\s*).*$", rf"\1'{data}'", t, count=1, flags=re.M)
m = list(re.finditer(r"^(\s*batch_size:\s*)\d+", t, flags=re.M))   # train, val, test -> last one = test
assert len(m) >= 3, f"expected 3 batch_size lines, found {len(m)}"
t = t[:m[-1].start()] + m[-1].group(1) + b + t[m[-1].end():]
open(dst, "w").write(t)
PYEOF
done
grep -n "batch_size" "$T/b8.yaml"

# BBDM builds train/val datasets even in test mode: one linked patch each
for s in train val; do mkdir -p "$T/data/$s/A" "$T/data/$s/B"; ln -s "$SRC/test/A/$f" "$T/data/$s/A/$f"; ln -s "$SRC/test/B/$f" "$T/data/$s/B/$f"; done
cd "$BBDM"
for b in 1 8; do
  g=$([ "$b" = 1 ] && echo 0 || echo 1)
  (
    s=$(date +%s)
    CUDA_VISIBLE_DEVICES=$g "$PY" -u main.py --config "$T/b$b.yaml" --sample_to_eval --gpu_ids 0 \
        --port $((12500 + b)) --result_path "$T/results" > "$T/b$b.log" 2>&1
    e=$(date +%s)
    n=$(find "$T/results" -path "*LQSPEED_b$b*" -name '*.png' -path '*/200/*' | wc -l)
    echo "batch $b: $n images in $((e - s)) s -> $(( (e - s) / 32 )) s/patch (includes 1-2 min model loading)"
  ) &
done
wait
echo "Logs: $T/b1.log  $T/b8.log"
