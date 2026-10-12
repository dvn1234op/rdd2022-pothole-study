# Road-Damage Detection on RDD2022 — leakage-controlled baselines, synthetic-data audit, and a multi-resolution ablation

Read in this order: `RESEARCH_AUDIT.md` (what was wrong) → `BASELINE_PROTOCOL.md` (fixed protocol) →
`EXPERIMENT_PLAN.md` → `RESULTS.md` (measured results + scientific assessment) → `paper/paper.md` (write-up) →
`LITERATURE_COMPARISON.md`.

## Repository layout
```
configs/        protocol.yaml (shared training protocol), data/*.yaml, experiments/*.yaml (one per run)
datasets/       parse_rdd.py → audit_dataset.py → near_duplicates.py → build_split.py → export_yolo.py
augmentation/   audit_synthetic.py, manual_audit_sheet.py, build_synthetic_sets.py
models/         multires.py (A0–A7 ablation family), test_multires.py (unit checks)
training/       train.py (single entry point for every run)
evaluation/     evaluate.py (canonical metrics), compare.py (paired bootstrap), make_tables.py
scripts/        run_queue.sh (sequential GPU queue)
data/           raw/ (RDD2022 zip + extracted), processed/ (parsed tables, hashes, embeddings), splits/ (frozen split + YOLO export)
experiments/    Ultralytics run dirs (weights, results.csv, resolved_config.yaml), logs/, queue.txt, done.txt
results/        <run>/metrics.json + raw predictions; dataset_audit/; synthetic_audit/; tables.md
visualizations/ dataset_audit/, synthetic_audit/
checkpoints/    pretrained/ (COCO weights used for init)
reports/        split_leakage_audit.json, legacy LEAKAGE_REPORT.md
paper/          paper.md (final write-up)
ALL MD/         original project documents (kept unmodified for traceability)
```

## Environment
* Ubuntu (WSL2) · Python 3.12.3 · PyTorch 2.14.1+cu130 · Ultralytics 8.4.171 · torchvision 0.29.1 · pycocotools ·
  scipy 1.18 · scikit-learn · OpenCV 5.0 · imagehash · pandas 3.0 · matplotlib 3.11
* Hardware used for every number: 1× NVIDIA RTX 4050 Laptop GPU (6 GB), 16 CPU threads, 11 GB RAM.

```bash
python3 -m pip install --user ultralytics==8.4.171 pycocotools imagehash scikit-learn opencv-python pandas matplotlib scipy
```

## Data preparation (≈ 10 min on 16 cores)
```bash
# 1. place RDD2022_released_through_CRDDC2022.zip contents under data/raw/RDD2022/RDD2022_all_countries/<Subset>/<Subset>/
python3 datasets/parse_rdd.py          # -> data/processed/{images_all,annotations_all}.csv (38,385 labeled images, 65,712 boxes)
python3 datasets/audit_dataset.py      # -> results/dataset_audit/*, visualizations/dataset_audit/*
python3 datasets/near_duplicates.py    # -> pHash + ResNet-50 embeddings, near-duplicate pairs (GPU)
python3 datasets/build_split.py        # -> data/splits/split_v1.csv (+ .sha256). Refuses to overwrite: the split is frozen.
python3 datasets/export_yolo.py        # -> data/splits/rdd_v1/ (YOLO format, long side 640) + COCO GT json + configs/data/rdd_v1.yaml
sha256sum -c data/splits/split_v1.sha256   # run inside data/splits/
```
Split: train 26,868 / val 3,839 / test 7,678 images; near-duplicate groups never cross splits; seed 20261004.

Synthetic-data experiment (requires the legacy `data/synthetic_yolo_format/` and `data/yolo_format/` folders):
```bash
python3 augmentation/audit_synthetic.py && python3 augmentation/manual_audit_sheet.py && python3 augmentation/build_synthetic_sets.py
```

## Training and evaluation
```bash
export PYTHONPATH=$PWD
python3 training/train.py configs/experiments/B01_yolov8s_s0.yaml      # any config
python3 evaluation/evaluate.py B01_yolov8s_s0                           # -> results/B01_yolov8s_s0/metrics.json
python3 evaluation/compare.py B01_yolov8s_s0 A4_s0 --save               # paired bootstrap of the difference
python3 evaluation/make_tables.py                                       # -> results/tables.md
# or run everything in experiments/queue.txt sequentially:
nohup scripts/run_queue.sh &
```
Seeds: the `seed` field of each experiment config (0, 1, 2); Ultralytics `deterministic=True`. cuDNN
non-determinism can still cause small run-to-run differences — this is why seed variance is reported.

Model configurations: baselines use Ultralytics checkpoints (`yolov8s.pt`, `yolo11s.pt`, `rtdetr-l.pt`, COCO
pretrained, in `checkpoints/pretrained/`). Ablation variants A0–A7 are defined in `models/multires.py`
(`VARIANTS` table) and initialised from YOLOv8s backbone layers 0–9 only.

Checkpoints: `experiments/<run>/weights/{best,last}.pt` (best = max val fitness). ~22–45 MB each.

## Expected outputs
* `results/dataset_audit/audit_summary.json`: 38,385 images, 55,006 target boxes, Cramér's V (domain × class) 0.369.
* `reports/split_leakage_audit.json`: 0 near-duplicate groups across train/val/test.
* `results/synthetic_audit/summary.json` + `manual_audit_labels.csv`: 0/60 sampled inserted boxes contain a visible pothole.
* `results/<run>/metrics.json` for every run; `results/tables.md` regenerated from them.
  With an identical GPU/software stack, metrics should reproduce to within the seed spread reported in `RESULTS.md`.
