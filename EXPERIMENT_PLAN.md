# Experiment Plan

Written before results were available (2026-10-04). Each block states the question, the comparison that answers
it, and what result would count as negative. Run status lives in `RESULTS.md` and `experiments/done.txt`.

## Research questions (refined after the dataset audit)

* **RQ1 (diagnosis).** Under a leakage-free split, which classes, object sizes and domains are actually weak for a
  standard detector? Is D40 the weak class, as the project assumed?
* **RQ2 (synthetic data).** Why did the previous SD-inpainted D40 augmentation not help, and does it help at any
  ratio once leakage is removed?
* **RQ3 (architecture).** Does preserving higher-resolution spatial information (P2, stride 4) while injecting
  low-resolution context (cross-resolution attention, adaptive fusion) improve detection — particularly for small
  and difficult instances — relative to an otherwise identical detector?
* **RQ4 (domain).** How much of D40 performance on India depends on India-specific training data
  (domain-class entanglement), as opposed to the class being intrinsically hard?

## Phase A — baselines (RQ1)
| Run | Model | Purpose |
|---|---|---|
| B01_yolov8s_s0/s1/s2 | YOLOv8s, full COCO init | reference baseline, 3 seeds for seed variance |
| B02_yolo11s_s0 | YOLO11s | stronger YOLO variant |
| B03_rtdetrl_s0 | RT-DETR-L | transformer detector with multi-scale deformable attention decoder |
Negative result for the project premise: D40 AP@50 is *not* the lowest class AP.

## Phase B — synthetic data (RQ2)
Eligible pool: 2,767 SD-inpainted images whose source is in `split_v1` train.
| Run | Train set | Isolates |
|---|---|---|
| B01 | real only (0 %) | reference |
| S025 / S050 / S100 | + 25 / 50 / 100 % synthetic | dose–response of the original method |
| R100 | + source pixels with the synthetic D40 boxes | effect of the inserted **labels** without generated pixels |
| D100 | + source pixels, original labels | effect of **duplication / extra optimisation steps** |
Interpretation: S100 ≈ R100 → generated pixels contribute nothing (consistent with the manual audit);
R100 < D100 → the inserted labels act as label noise; D100 ≈ B01 → duplication is neutral.
Measured: D40 AP@50 & AP@50:95, overall mAP@50 & mAP@50:95, P/R/F1, per-domain (esp. Czech/USA where boxes were inserted).

## Phase C — architecture ablation (RQ3), all with backbone-only COCO init
| Run | Variant |
|---|---|
| A0 | YOLOv8s P3–P5 (ablation reference) |
| A1 | + P2 high-resolution branch (multi-resolution representation only) |
| A2 | A1 + deformable cross-resolution attention (P2←P5, P5←P2) |
| A3 | A1 + adaptive fusion gates (top-down) |
| A4 | A1 + CRA + gates (full model) |
| A5 | A4 with dense cross-attention instead of deformable |
| A6 | A0 + CRA + gates (full model without high-res branch) |
| A7 | P2/P3-only network (full model without low-res branch; also a large capacity drop — interpret with care) |
Primary endpoints: small-object AP@50, D40 AP@50, overall mAP@50:95. Seeds 1–2 are run for A0 and the best
variant if A-differences are within ~1 AP (otherwise single-seed with that limitation stated).
Negative result: A4 − A0 CI includes 0 on small-object AP and on mAP@50:95.

## Phase D — domain (RQ4)
| Run | Train | Evaluate |
|---|---|---|
| X01 | train − India (val − India for selection) | full test, per domain; India = held out |
Compared to B01 on the same India test images. Large D40 drop on India with little drop elsewhere → D40
performance is domain-bound (entanglement), not class-intrinsic.

## Phase E — follow-up only if justified by evidence
* If small-object AP is dominated by Norway's down-sampled boxes: test **resolution** (tiled / higher-res input
  for high-resolution domains) against **architecture** (A-variants) — a cheaper and more direct hypothesis.
* If D40 is weak only out-of-domain: domain-aware balancing rather than synthetic generation.

## Compute budget and stopping rules
≈ 6 min/epoch for YOLOv8s → ≈ 2 h per 20-epoch run; P2 variants ≈ 1.3–1.6×; RT-DETR-L ≈ 3×. Priority order
= queue order (`experiments/queue.txt`). If the budget runs out, the plan stops at whatever block is incomplete and
`RESULTS.md` lists the missing runs explicitly; nothing is extrapolated.
