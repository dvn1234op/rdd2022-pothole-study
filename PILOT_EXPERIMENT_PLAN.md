# Pilot Experiment Plan — go / no-go for the multi-resolution + imbalance-handling direction

Fixed 2026-10-04, before any pilot model finished training. This is a **pilot**: one seed, reduced budget, the
question is only whether there is enough signal to justify the full-scale study.

**Pilot question.** Under one identical protocol on the real RDD2022 distribution, does our multi-resolution model,
with one real-data-only compensation for the dataset's imbalance, move overall mAP@50 and D40 AP@50 relative to
strong current detectors — and is any gain due to the architecture, the compensation, or both?

## 1. Data (unchanged from `BASELINE_PROTOCOL.md`)
Real RDD2022, 38,385 labeled images, frozen leakage-controlled split `split_v1`
(train 26,868 / val 3,839 / test 7,678). Val and test are never resampled, rebalanced or augmented.
Baselines see the real training distribution with no compensation of any kind.

## 2. Training-split problem (measured, `results/compensation/`)
| | D00 | D10 | D20 | D40 |
|---|---:|---:|---:|---:|
| train boxes | 18,128 | 8,297 | 7,458 | 4,567 |

D00:D40 = 3.97 : 1 — a moderate class imbalance. The stronger structure is **domain-class entanglement**:

| Domain | train images | D40 boxes | D40 % of the domain's boxes | % of all train D40 |
|---|---:|---:|---:|---:|
| China_Drone | 1,680 | 60 | 2.8 | 1.3 |
| China_MotorBike | 1,382 | 162 | 5.0 | 3.6 |
| Czech | 1,982 | 137 | 11.4 | 3.0 |
| India | 5,395 | 2,234 | 46.4 | 48.9 |
| Japan | 7,354 | 1,570 | 13.7 | 34.4 |
| Norway | 5,712 | 308 | 4.0 | 6.7 |
| United_States | 3,363 | 96 | 1.2 | 2.1 |

Domain × class Cramér's V (train boxes) = 0.367. A detector can learn "Indian road texture ⇒ pothole".

## 3. Compensation: chosen empirically, before training (`augmentation/compensation.py`)
Candidates were simulated at the **same total exposure (1.25× the train set, fixed a priori)** with
repeat-factor sampling (Gupta et al., LVIS, CVPR 2019: r = max(1, √(t/f)), image repeat = max over its categories):

| Sampler | D40 share of exposure | D40 exposure from India | Cramér's V of exposure |
|---|---:|---:|---:|
| none | 11.9 % | 48.9 % | 0.367 |
| A — class-aware RFS (categories = class) | 16.8 % | 48.9 % | 0.377 |
| **C — domain×class-aware RFS (categories = (domain, class) cells)** | 14.2 % | 38.0 % | **0.299** |

Decision rule (stated in code before training): choose the sampler with the lower exposure entanglement.
Class-aware sampling raises D40 exposure but *strengthens* the India⇔pothole correlation (V ↑, India's
within-domain D40 share 46 → 54 %). **Chosen: C, domain-class-aware repeat-factor sampling.** It is the only
compensation used; no loss reweighting, no synthetic data, no augmentation change.

Equal compute: the compensated list has 33,595 entries (stochastic rounding, seed 0) and is trained for
**16 epochs** (vs 20 × 26,868) → 1.0003× the baseline's optimisation steps; mosaic closed for the last 4 epochs.

## 4. Models
| Role | Model | Why |
|---|---|---|
| Baseline | YOLOv8s (COCO) | the project's original baseline |
| Baseline | YOLO11s (COCO) | widely used current YOLO |
| Baseline | YOLO26s (COCO, Ultralytics, Jan 2026) | newest YOLO; NMS-free head, small-target-aware assignment |
| Baseline | RT-DETR-L (COCO) | transformer detector with multi-scale deformable decoder (the decoder family the proposal named) |
| Baseline (if it fits 6 GB) | RF-DETR (DINOv2 backbone; ICLR 2026) | current strong DETR; trained with its own pipeline, evaluated with our script |
| Ours | A4 = YOLOv8s backbone (COCO, layers 0–9) + P2 stride-4 high-resolution branch + bidirectional deformable cross-resolution attention (P2↔P5) + adaptive fusion gates; YOLOv8 head (`models/multires.py`) | small trainable proof of concept |
| Control | A0 = same as A4 without the multi-resolution components, same backbone-only init | isolates the architecture from initialisation |

Deviations from the original proposal, kept deliberately for a pilot: no from-scratch dual-patch ViT (patch 8 at
640 px = stride 8 = P3, already in every baseline; a from-scratch ViT would confound resolution with pretraining
and does not fit 6 GB) and a YOLO head instead of an RT-DETR decoder (held fixed for A0/A4; RT-DETR-L is
benchmarked separately). The central idea — finer resolution + cross-resolution interaction + gated fusion — is intact.

## 5. Experiments
| ID | Architecture | Compensation | Run name |
|---|---|---|---|
| EXP-0 | each baseline | none | B01 (YOLOv8s), B02 (YOLO11s), B04 (YOLO26s), B03 (RT-DETR-L), RF-DETR |
| EXP-1 | ours (A4) | none | A4_s0 |
| EXP-2 | ours (A4) | domain-class RFS | P2_A4_comp_s0 |
| EXP-3 | best EXP-0 baseline (by test mAP@50 — chosen after EXP-0, before EXP-3 is run) | domain-class RFS | P3_<model>_comp_s0 |
| control | A0 | none / domain-class RFS | A0_s0 / P4_A0_comp_s0 (if compute allows) |

Protocol: 640 px, seed 0, Ultralytics YOLO family SGD lr 0.01 / RT-DETR AdamW 1e-4, effective batch 64,
20 epochs (16 for compensated runs, equal steps), best checkpoint on val. Same evaluation script for all.

## 6. Metrics and decision criteria
Reported for every run (test split): mAP@50, mAP@50:95, D40 AP@50, D40 AP@50:95, per-class AP, P/R/F1 at val-chosen
thresholds, small/medium/large AP@50 (640-px input scale), per-domain AP, params/GFLOPs/FPS, bootstrap 95 % CIs;
paired-bootstrap CIs for EXP-1 − A0, EXP-2 − EXP-1, EXP-3 − EXP-0, EXP-2 − EXP-3.

Signal is judged on two primary endpoints — **overall mAP@50** and **D40 AP@50** — plus small-object AP@50 for the
multi-resolution hypothesis. A difference counts only if its paired-bootstrap 95 % CI excludes 0. Single-seed
caveat: seed-to-seed spread of YOLOv8s on this protocol is unknown in the pilot; differences below ~1 AP are
treated as "≈" regardless of CI.

* **GO** — EXP-2 > EXP-3 on mAP@50 or D40 AP@50 (CI excludes 0), A4 ≥ A0 on small-object AP, and A4 is competitive
  (within ~2 AP mAP@50) with the best baseline.
* **MAYBE** — one ingredient shows a CI-backed gain (architecture vs A0, or compensation for both models) but the
  combination does not beat the compensated best baseline, or gains are < ~1 AP.
* **NO-GO** — neither the architecture (A4 vs A0) nor the compensation (EXP-2 vs EXP-1, EXP-3 vs EXP-0) shows a
  CI-backed gain on the primary endpoints.

## 7. Budget
≈ 2 h per YOLO-size run, ≈ 3 h for A4, ≈ 6 h for RT-DETR-L on the RTX 4050 (6 GB): ≈ 20–25 GPU-hours total.
The full-attack backlog (synthetic controls, A1–A7, seeds, cross-domain) is parked in
`experiments/queue_full_attack_backlog.txt` and is not run in the pilot.
