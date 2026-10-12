#!/usr/bin/env bash
# After the 50-image generation test has finished, queue P6 (YOLO26s-P2) and restart the training runner.
cd "$(dirname "$0")/.."
until grep -q "=== gen test done" experiments/logs/queue.log; do sleep 60; done
grep -qx P6_yolo26s_p2_s0 experiments/queue.txt || echo P6_yolo26s_p2_s0 >> experiments/queue.txt
pgrep -x -f "bash scripts/run_queue.sh" >/dev/null || setsid nohup bash scripts/run_queue.sh >> experiments/logs_queue_runner.out 2>&1 < /dev/null &
