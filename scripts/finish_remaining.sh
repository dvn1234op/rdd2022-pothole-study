#!/usr/bin/env bash
# Remaining implementation work, run detached: tiled inference on Norway (inference only), then the two transformer
# baselines through the normal queue runner (RT-DETR-L with warmup_bias_lr=0.0, then RF-DETR Small).
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
for r in B04_yolo26s_s0 P5_A4full_comp_s0 B01_yolov8s_s0; do
  echo "=== tiled Norway $r start $(date)" >> experiments/logs/queue.log
  python3 evaluation/tiled_inference.py $r --domain Norway > experiments/logs/tiled_$r.log 2>&1
  echo "=== tiled Norway $r done rc=$? $(date)" >> experiments/logs/queue.log
done
for e in B03_rtdetrl_s0 B05_rfdetr_small_s0; do grep -qx "$e" experiments/queue.txt || echo "$e" >> experiments/queue.txt; done
pgrep -x -f "bash scripts/run_queue.sh" >/dev/null || exec bash scripts/run_queue.sh >> experiments/logs_queue_runner.out 2>&1
