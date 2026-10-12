#!/usr/bin/env bash
# Waits until the training queue has finished and the GPU is free, then runs a 50-image pothole-generation test.
cd "$(dirname "$0")/.."
# anchored/exact matches: an unanchored -f also matches any shell whose command line merely contains these strings
while pgrep -x -f "bash scripts/run_queue.sh" >/dev/null || pgrep -f "^python3 training/train" >/dev/null || pgrep -f "^python3 evaluation/evaluate\.py" >/dev/null; do sleep 60; done
echo "=== gen test start $(date)" >> experiments/logs/queue.log
HF_HUB_OFFLINE=1 python3 augmentation/colab_generate_potholes.py --pkg data/gen_pkg_local --out data/splits/gen_test --n 50 \
  > experiments/logs/gen_test.log 2>&1
echo "=== gen test done rc=$? $(date)" >> experiments/logs/queue.log
