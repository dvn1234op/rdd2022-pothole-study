# Baseline & Evaluation Protocol (fixed before any model result was seen)

Frozen on 2026-10-04, before the first training run finished. Any later change is listed in §8 with its date and
reason. Every number in `RESULTS.md` is produced under exactly this protocol unless the row says otherwise.

## 1. Data

| Item | Value |
|---|---|
| Source | RDD2022 (`RDD2022_released_through_CRDDC2022.zip`), official **train** folders of all 7 subsets (38,385 labeled images). Official test folders are unlabeled and unused. |
| Classes | D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole. All other raw labels (D43, D44, D50, Repair, D01, D11, Block crack, D0w0) are dropped; their regions become unlabeled background (CRDDC'22 convention). One zero-width D20 box dropped. |
| Split | `data/splits/split_v1.csv` (SHA-256 in `split_v1.sha256`), built by `datasets/build_split.py`, seed 20261004. |
| Split method | Near-duplicate groups (pHash Hamming ≤ 4 **or** ResNet-50 cosine ≥ 0.985; 2,436 images in 2,261-edge groups) never straddle splits. StratifiedGroupKFold(10) stratified by subset × image-has-D40. Folds 0–1 → test (20 %), fold 2 → val (10 %), rest → train (70 %). |
| Sizes | train 26,868 · val 3,839 · test 7,678 images (negatives — images with no target box — included in all splits). |
| Leakage check | 0 test and 0 val images have a near-duplicate partner in train (`reports/split_leakage_audit.json`). 1,451 test images have a *looser* look-alike in train (pHash ≤ 8: same road, different location) — reported, not removed. |
| Image preprocessing | Images resized once so the long side = 640 px (`cv2.INTER_AREA`, JPEG q95; `datasets/export_yolo.py`). Detectors then letterbox to 640. This only removes repeated decoding of 4,000-px Norway images. |
| Test-set rule | The test split is used **only** by `evaluation/evaluate.py` after training. Checkpoint selection, threshold selection and every design decision use val. Synthetic/augmented/duplicated images are only ever added to train. |

### Cross-domain protocol (X-runs)
Question: how much of D40 performance on India depends on having seen Indian training data?
Train = `split_v1` train minus India (21,473 images); model selection on val minus India (3,068 images);
evaluate on the full `split_v1` test, reported per domain, India being the held-out domain.
Comparison partner: B01 (same protocol, India included), evaluated on the identical India test images (1,540).

## 2. Training protocol P20 (`configs/protocol.yaml`)

| Item | YOLO family (YOLOv8s, YOLO11s, A0–A7) | RT-DETR-L |
|---|---|---|
| Init | COCO-pretrained (baselines: all layers; **A0–A7: backbone layers 0–9 only**, neck/head random) | COCO-pretrained, all layers |
| Epochs | 20 | 20 |
| Input | 640 | 640 |
| Batch / effective batch | 16 / 64 (grad accumulation, nbs 64); A-variants 8 / 64 | 4 / 64 (see change log 2026-10-06) |
| Optimiser | SGD, lr0 0.01, lrf 0.01 (linear), momentum 0.937, wd 5e-4, warmup 3 ep | AdamW, lr0 1e-4, lrf 0.01, wd 1e-4, warmup 3 ep |
| Augmentation | Mosaic 1.0 (off for last 5 epochs), HSV (0.015, 0.7, 0.4), translate 0.1, scale 0.5, fliplr 0.5, random erasing 0.4; no mixup, no copy-paste, no rotation/shear | same |
| AMP / determinism | AMP on, `deterministic=True`, seed per config (0 unless stated) | same |
| Model selection | `best.pt` = max val fitness (0.1·mAP50 + 0.9·mAP50:95), Ultralytics default | same |
| Hardware | 1× NVIDIA RTX 4050 Laptop GPU (6 GB), 16 CPU threads, 11 GB RAM (WSL2) | same |

**Budget justification.** One YOLOv8s epoch ≈ 6 min on this GPU; the study needs ≈ 25 runs. 20 epochs is a
reduced budget — absolute AP is below fully converged literature numbers — but it is identical for every model,
so differences are attributable to the variable under test. RT-DETR is known to need longer schedules than YOLO;
under an equal-epoch budget it is disadvantaged, and this is stated wherever it is compared.

**Deviations from the original proposal (deliberate).**
1. RT-DETR-style decoder → YOLOv8 Detect head for the ablation series (compute; the hypothesis concerns the
   representation, so the head is held fixed for all A-variants).
2. Dual-patch ViT (patch 8 / 16) → P2 (stride 4) high-resolution branch on a pretrained CNN backbone
   (patch-8 ≡ stride 8 ≡ P3, which the baseline already has; see `RESEARCH_AUDIT.md` row 27).
3. RF-DETR not run: its DINOv2 backbone at 640 px exceeds the 6 GB memory budget at a usable batch size and it
   is not available in the installed toolchain; it is listed as missing in `RESULTS.md`.
4. **Full-init run of the full model (added 2026-10-05, after A0/A4/B01 test results were seen).** The paired
   bootstrap showed B01 − A0 = +5.6 mAP@50 (CI [+4.8, +6.2]) with identical architecture, i.e. backbone-only init
   is the largest handicap of every A-variant. `P5_A4full_comp_s0` therefore initialises A4 from COCO yolov8s in
   **every layer that has a counterpart** (`init_scope: full`, `models/multires.load_full`: backbone, PAN neck,
   Detect box/cls stems; P2 path, gates, CRA and the nc-specific cls output keep their own init; 317 of 355
   tensors). Its only fair comparison is `P3_yolov8s_comp_s0` (same init source, same RFS data, same steps).
   Because this run was designed after seeing test results, it is reported as a post-hoc follow-up, not as part
   of the pre-registered ablation; the A0–A7 ablation keeps backbone-only init.
5. **P2 branch on the strongest baseline (added 2026-10-06, post hoc, after B04/P5 test results).** P5 was best on
   Norway D40 (median test pothole side 10 px at 640, 91 % < 32 px) but behind YOLO26s overall, so the P2 (stride-4)
   idea is tested on YOLO26s itself: `P6_yolo26s_p2_s0` = Ultralytics `yolo26s-p2.yaml`, all yolo26s.pt tensors with
   a counterpart transferred via an index remap (`training/train.load_p2_shifted`; 766/902 tensors incl. BN stats;
   the Detect stems change width and start fresh), batch 8 / nbs 64, otherwise the B04 recipe. Fair comparison:
   `B04_yolo26s_s0`. Reported as a post-hoc follow-up.

## 3. Evaluation (`evaluation/evaluate.py`, identical for all models)

| Item | Value |
|---|---|
| Inference | conf 0.001, NMS IoU 0.7, max 300 detections/image, imgsz 640, FP16, class-wise NMS |
| Metric implementation | pycocotools `COCOeval` (bbox), maxDets 100, 101-point interpolation, ground truth at the 640-px scale |
| Overall | mAP@50 and mAP@50:95 = mean over the 4 classes |
| Per class | AP@50, AP@50:95 for D00/D10/D20/D40 |
| Size buckets | box area at the 640-px input: small < 32², medium 32²–96², large ≥ 96² (COCO thresholds applied at network-input scale) |
| Per domain | COCOeval restricted to the domain's test images; domains with < 30 GT instances of a class are flagged as unreliable for that class |
| P / R / F1 | IoU 0.5, per-class confidence threshold that maximises F1 on **val**, applied unchanged to test |
| Uncertainty | 1,000-sample image-level bootstrap on test → 95 % CI for per-class AP@50 and mAP@50; paired bootstrap (same resamples) for model differences (`evaluation/compare.py`) |
| Sanity check | an independent IoU-0.5 matcher re-computes per-class AP@50 and must equal pycocotools (stored as `sanity_own_AP50_vs_coco`) |
| Complexity | parameters, GFLOPs @640, FP16 batch-1 forward latency / FPS (200 iters after 30 warm-up, NMS excluded), peak inference memory, checkpoint size — same GPU for all |

## 4. Comparison rules
* Only models evaluated by this script on `split_v1` test appear in the same table.
* Literature numbers are listed separately (`LITERATURE_COMPARISON.md`) with their own dataset/split/metric and a
  "directly comparable?" column; they are never mixed into our tables.
* mAP@50 and mAP@50:95, overall and per-class, are never interchanged. D40 claims use D40 AP only.
* A difference is called an improvement only if the paired-bootstrap 95 % CI excludes 0 **and** (where seeds
  exist) it exceeds the seed-to-seed spread.

## 5. Experiment registry
Every run = `configs/experiments/<name>.yaml` → `experiments/<name>/` (Ultralytics run dir, `resolved_config.yaml`)
→ `results/<name>/metrics.json` (+ raw predictions `preds_{val,test}.json`, per-image matches). Queue:
`experiments/queue.txt`, runner `scripts/run_queue.sh`, finished list `experiments/done.txt`.

## 8. Change log
* 2026-10-04 — protocol frozen.
* 2026-10-06 — B03 RT-DETR-L batch 8 → 4 (memory: 6 GB GPU; effective batch unchanged via nbs 64), set before B03 was run.
* 2026-10-05 — added post-hoc full-init run `P5_A4full_comp_s0` (deviation 4); A0–A7 protocol unchanged.
* 2026-10-06 — added post-hoc run `P6_yolo26s_p2_s0` (deviation 5).
* 2026-10-08 — RT-DETR: `warmup_bias_lr: 0.0` (protocol.yaml, rtdetr section). Ultralytics' default 0.1 is an SGD value; with AdamW lr0 1e-4 the first B03 attempt trained biases at 0.067/0.033 during warm-up and reached only mAP50 0.061 after 10 epochs. That attempt is kept as `experiments/_bugged_B03_rtdetrl_s0_warmupbias` and not reported. YOLO runs (SGD lr0 0.01) are unaffected.
