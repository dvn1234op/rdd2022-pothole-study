# Pilot handoff (state as of 2026-10-04 ~12:50 UTC)

## What is running
A background GPU queue inside WSL (`scripts/run_queue.sh`) trains + evaluates the pilot runs one after another.
It does NOT depend on VS Code / Claude being open. Keep the laptop on, plugged in, awake.

Queue (`experiments/queue.txt`), finished runs in `experiments/done.txt`, log `experiments/logs/queue.log`:
1. B01_yolov8s_s0 — DONE (test mAP50 52.2, D40 AP50 42.3, small AP50 21.2)
2. A4_s0 — our model, no compensation (EXP-1) — running (batch 8; ~17 min/epoch)
3. A0_s0 — same-init control without multi-resolution parts
4. B02_yolo11s_s0, 5. B04_yolo26s_s0 — baselines
6. P2_A4_comp_s0 — our model + domain-class repeat-factor sampling (EXP-2)
7. B03_rtdetrl_s0 — RT-DETR-L, 8. B05_rfdetr_small_s0 — RF-DETR (own pipeline, `training/train_rfdetr.py`, env `/home/divon/rfdetr_env`)

User decision: compensation is applied to OUR model only (no compensated baseline). Report gains of
"our model + compensation" as the combined approach; architecture effect only from A4 vs A0 (both uncompensated).

## Commands (run inside Ubuntu: `wsl -d Ubuntu`, then `cd ~/BTP`)
```bash
tail -5 experiments/logs/queue.log                         # which run is active / finished
tail -c 300 experiments/logs/<RUN>.train.log | tr '\r' '\n' | tail -1   # live progress
cut -d, -f1,6-9 experiments/<RUN>/results.csv              # val mAP per epoch
python3 evaluation/make_tables.py                          # regenerate results/tables.md from all finished runs
python3 evaluation/compare.py A0_s0 A4_s0 --save           # paired bootstrap: architecture effect
python3 evaluation/compare.py A4_s0 P2_A4_comp_s0 --save   # compensation effect on our model
python3 evaluation/compare.py B01_yolov8s_s0 P2_A4_comp_s0 --save
# if the runner died (e.g. after a reboot): it skips finished runs automatically
(setsid nohup scripts/run_queue.sh > experiments/logs_queue_runner.out 2>&1 &)
```
A run interrupted mid-training restarts from scratch: delete `experiments/<RUN>/` first if it has no weights/best.pt.

## Cooling settings applied (reset on reboot)
GPU clock cap (admin PowerShell): `nvidia-smi -lgc 300,1500` (undo: `nvidia-smi -rgc`).
CPU boost off: `powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PERFBOOSTMODE 0` (undo: value 2) and
PROCTHROTTLEMAX 99 (undo: 100), then `powercfg /setactive SCHEME_CURRENT`.
Keep the GPU cap for the whole pilot so FPS numbers are comparable across models.

## Remaining deliverables when runs finish
1. `results/tables.md` (make_tables.py) — master, D40, small-object, per-domain tables.
2. Paired comparisons above.
3. `PILOT_RESULT.md` — sections listed in the pilot brief (hypothesis, dataset problem, why the compensation,
   models, protocol, exact results, D40, small-object, ablation interpretation, what works, what is unverified,
   proposed full-scale experiment) + GO / MAYBE / NO-GO using the criteria fixed in `PILOT_EXPERIMENT_PLAN.md` §6.
   Only actual numbers; no SOTA language.

Key docs: `PILOT_EXPERIMENT_PLAN.md`, `RESEARCH_AUDIT.md`, `BASELINE_PROTOCOL.md`, `LITERATURE_COMPARISON.md`, `README.md`.
