#!/usr/bin/env bash
# Sequential GPU job queue. Each line of experiments/queue.txt is an experiment config name (without .yaml).
# A job is: train (unless already trained) -> evaluate on val+test with evaluation/evaluate.py.
# Configs with "model: rfdetr:<Class>" are trained with training/train_rfdetr.py in the isolated rfdetr env.
# Lines can be appended while the queue runs. Finished names go to experiments/done.txt.
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
RFPY=/home/divon/rfdetr_env/bin/python
mkdir -p experiments/logs
touch experiments/queue.txt experiments/done.txt
while true; do
  # never run two GPU jobs at once (e.g. a job left over from a previous runner instance)
  while pgrep -f "training/train(_rfdetr)?\.py" >/dev/null || pgrep -f "evaluation/evaluate\.py" >/dev/null; do sleep 60; done
  next=$(grep -v -x -F -f experiments/done.txt experiments/queue.txt | grep -v '^\s*$' | grep -v '^#' | head -1)
  [ -z "$next" ] && { echo "queue empty $(date)"; break; }
  cfg="configs/experiments/$next.yaml"
  echo "=== $next start $(date)" | tee -a experiments/logs/queue.log
  if grep -q "^model: rfdetr:" "$cfg"; then
    [ -f "results/$next/preds_test.json" ] || $RFPY training/train_rfdetr.py "$cfg" > "experiments/logs/$next.train.log" 2>&1
    ok=$([ -f "results/$next/preds_test.json" ] && echo 1)
  else
    # best.pt alone is not proof of a finished run (it exists after epoch 1); .train_done is written only on a clean exit
    [ -f "experiments/$next/.train_done" ] || { python3 training/train.py "$cfg" > "experiments/logs/$next.train.log" 2>&1 && touch "experiments/$next/.train_done"; }
    ok=$([ -f "experiments/$next/.train_done" ] && [ -f "experiments/$next/weights/best.pt" ] && echo 1)
  fi
  if [ -n "$ok" ]; then
    python3 evaluation/evaluate.py "$next" > "experiments/logs/$next.eval.log" 2>&1
    echo "=== $next done rc=$? $(date)" | tee -a experiments/logs/queue.log
  else
    echo "=== $next FAILED (no checkpoint/predictions) $(date)" | tee -a experiments/logs/queue.log
  fi
  echo "$next" >> experiments/done.txt
done
