#!/usr/bin/env bash
# G3 = corrected synthetic potholes (size/height from real D40 of the SAME target domain, size-aware filters).
# Full generation -> import as training set G3 -> fair runs (YOLO26s and our P5 recipe, same data)
# -> then the parked DETR baselines. Run detached (setsid nohup).
cd "$(dirname "$0")/.."
echo "=== gen v3 full start $(date)" >> experiments/logs/queue.log
HF_HUB_OFFLINE=1 python3 augmentation/colab_generate_potholes.py --pkg data/gen_pkg_local --out data/splits/gen_v3_raw --n 2000 \
  > experiments/logs/gen_v3.log 2>&1
rc=$?
echo "=== gen v3 full done rc=$rc $(date)" >> experiments/logs/queue.log
[ $rc -eq 0 ] || exit $rc
python3 augmentation/import_generated.py data/splits/gen_v3_raw --tag G3 --models B04_yolo26s P5_A4full \
  > experiments/logs/import_G3.log 2>&1 || exit 1
for e in G3_B04_yolo26s_s0 G3_P5_A4full_s0 $(cat experiments/queue_after_generation.txt); do
  grep -qx "$e" experiments/queue.txt || echo "$e" >> experiments/queue.txt
done
pgrep -x -f "bash scripts/run_queue.sh" >/dev/null || setsid nohup bash scripts/run_queue.sh >> experiments/logs_queue_runner.out 2>&1 < /dev/null &
