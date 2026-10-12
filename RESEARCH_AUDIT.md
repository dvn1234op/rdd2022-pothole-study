# Research Audit — Multi-Resolution Attention for Road Damage Detection (RDD2022)

Audit date: 2026-10-04. Auditor stance: hostile but fair. Nothing below is accepted because it appears in an
earlier document; every claim was re-derived from the raw data, from code that exists on disk, or from a
primary source that was actually retrieved.

**Scope.** The "attached research document" was not available in the repository. The audit covers the four
documents that are on disk — `ALL MD/BTP_FULL_CONTEXT_HANDOFF.md`, `ALL MD/BTP_Data_First_Final_Output.md`,
`ALL MD/BTP_Data_First_Research_Roadmap.md`, `Project_Work_Summary.md` — plus `reports/LEAKAGE_REPORT.md`,
the surviving training run `data/raw/RDD2022_runs/yolov8s_optimized/`, the derived datasets
`data/yolo_format*`, `data/synthetic_yolo_format`, and the claims restated in the project brief
(e.g. "~4,000 synthetic potholes", "D40 AP50 0.243 → 0.233"). Phrases such as "monumental breakthrough" or
"revolutionary" do not occur in the on-disk documents; if they occur in the unavailable document they are
covered by rows 38–40.

**Status legend.**
`VE` verified experimentally (this audit re-ran it) · `VS` verified from an external primary source ·
`INF` inferred · `UNS` unsupported (no evidence on disk / not found in source) · `INC` incorrect ·
`REQ` requires additional experiment (status updated in `RESULTS.md` once the run finishes)

Evidence files referenced below: `results/dataset_audit/*`, `reports/split_leakage_audit.json`,
`results/synthetic_audit/*`, `visualizations/*`.

---

## 1. Dataset facts

| # | Claim | Current Evidence | Status | Problem | Required Fix |
|---|---|---|---|---|---|
| 1 | RDD2022 has 47,420 images | Arya et al., *Geoscience Data Journal* 11(4) 2024 / arXiv:2209.08538; local count: 38,385 labeled (official train) + 9,035 unlabeled (official test) = 47,420 | VS + VE | Only **38,385** images have public labels. Official test labels were never released, so 47,420 is not the size of any evaluable dataset. | State "47,420 released, 38,385 labeled"; all experiments use the 38,385. |
| 2 | 6 countries | Raw folders | VE | There are **7 subsets**: China is two different capture platforms (Drone = top-down, MotorBike = close-range oblique). Treating "China" as one domain hides a large viewpoint shift. | Report 7 domains. |
| 3 | 4 classes | Raw XML | INC (as stated) | The raw annotations contain **12** labels (D00, D10, D20, D40, D44 5,057, D50 3,581, Repair 1,046, D43 793, D01 179, D11 45, Block crack 3, D0w0 1). The 4-class task is a protocol choice (CRDDC'22), not a property of the data. | State the label mapping; we drop non-target labels (their regions become unlabeled background). |
| 4 | D00 ≈ 26,016, D10 ≈ 11,830, D20 ≈ 10,617, D40 ≈ 6,544 | `results/dataset_audit/audit_summary.json` | VE | Exact. One D20 box is degenerate (zero width) → 10,616 usable. | None. |
| 5 | "> 55,000 instances" | 55,006 target-class boxes | VE | — | — |
| 6 | "~4-to-1 D00:D40 imbalance" | 26,016 / 6,544 = 3.98 (box level); 13,548 / 3,674 = 3.69 (image level) | VE | Fine, but the imbalance is moderate by long-tail standards (Gini 0.27) and is *not* the dominant structure in the data — see row 8. | Report both box- and image-level ratios. |
| 7 | D40 is geographically concentrated (mostly India) | `d40_by_domain.csv` | VE | India holds 48.7 % of all D40 boxes with 12.4 % of all target boxes (3.9× over-representation); Japan 34.3 %. India+Japan = 83 % of D40. US holds 2.1 % of D40 with 20 % of the boxes. | Quantified; keep. |
| 8 | (implicit) class imbalance is *the* dataset problem | Contingency analysis | INC / incomplete | Country and class are strongly **dependent**: χ² = 22,445 (dof 18), bias-corrected Cramér's V = **0.369** (box level); image-level D40-presence vs domain V = 0.226. 46.7 % of India's boxes are D40 vs 1.2 % of USA's. The problem is **domain-class entanglement**, which class-rebalancing alone cannot remove. | Separate class imbalance, domain imbalance and entanglement explicitly (done in `RESULTS.md` §1). |
| 9 | Potholes are small objects / D40 is the small-object class | `size640_by_class.csv` (640-px input scale, COCO thresholds) | INC | D40: 26 % small, 58 % medium, 16 % large. D00 has more small boxes (6,151) than D40 (1,700). **74 % of all small boxes come from Norway**, whose 3,643–4,040-px images are downscaled ~6× to 640 px; 1,267 boxes (almost all Norway) are < 8×8 px at the network input. | "Small object" must be analysed per domain; the dominant cause of small objects at 640 px is input down-sampling of one domain. |
| 10 | Image resolution is ~uniform / 640 input is fine | `domain_counts.csv` | INC (implicit) | Resolutions: 512² (China), 600² (Czech, most Japan), 720² (India), 640² (USA), 540–1080² (Japan), **3643–4040 × 2035–2044 (Norway)**. | Report; treat input resolution as a variable in the scale analysis. |

## 2. Splits, duplicates and leakage

| # | Claim | Current Evidence | Status | Problem | Required Fix |
|---|---|---|---|---|---|
| 11 | "73 exact perceptual duplicates; they will be kept in the same split" | `reports/LEAKAGE_REPORT.md`; re-run: 73 images share a pHash with another image (83 exact-hash pairs) | VE (count) / INC (sufficiency) | Exact hash equality misses near-duplicates. With pHash ≤ 4 or ResNet-50 cosine ≥ 0.985 (threshold set by visual inspection, `datasets/near_duplicates.py`) there are 2,261 near-duplicate pairs, 2,436 images in multi-image groups — consecutive dashcam frames and the same physical crack photographed twice. The grouping was never applied to the old split. | Group-aware split (done: `split_v1`). |
| 12 | The existing train/val split (`data/yolo_format`) is valid | `reports/split_leakage_audit.json` | INC | Random image-level split; **231 of 4,744 val images (4.9 %) have a near-duplicate in train, 18 have an exact duplicate**; no test set (val doubles as test); **all 14,618 images without target boxes were dropped**, so false positives on clean road are never measured; script not preserved. | Replaced by `split_v1`: 70/10/20 train/val/test, StratifiedGroupKFold over near-dup groups, stratified by domain × D40-presence, negatives kept, SHA-256 frozen. 0 near-dup groups straddle splits. |
| 13 | Image IDs are sequential so sequence leakage can be handled by ID blocks | Similarity of consecutive IDs vs random pairs | INC | Consecutive IDs are no more similar than random pairs in any subset (IDs were shuffled by the dataset authors). ID-block splitting would do nothing. | Similarity-based grouping (row 12). |
| 14 | Synthetic images never touch val/test | `results/synthetic_audit/summary.json` | VE (old split) / INC (new split) | Under the old split all synthetic sources were train images. Under any clean re-split, **1,238 of 4,005 synthetic images are generated from val/test source scenes**; using them would leak test scenes. | Only the 2,767 synthetic images with `split_v1`-train sources are eligible. |

## 3. Synthetic data (Stable-Diffusion inpainting)

| # | Claim | Current Evidence | Status | Problem | Required Fix |
|---|---|---|---|---|---|
| 15 | "~4,000 synthetic potholes were generated" | 4,005 images, exactly 1 inserted D40 box each (`augmentation/audit_synthetic.py`) | INC | **Manual audit of 60 random inserted boxes: 0 / 60 contain a visible pothole** (95 % Clopper–Pearson upper bound 6 %); 23 / 60 (38 %) are not on the road at all (sky, buildings, vegetation, vehicles, signs); inpainting sometimes *removed* real content (signs, existing cracks). For 59 % of boxes, pixel change inside the box is < 2× the change in the rest of the image. Box centres are uniform in y ∈ [0.48, 0.80] and areas in a narrow 0.5–2 % band — a fixed random placement rule, unlike real D40 (`visualizations/synthetic_audit/`). | Describe the set as "4,005 SD-inpainted images with a D40 label at a random location", not as potholes. |
| 16 | Synthetic data adds pothole diversity to D40-poor domains | Sources: 2,000 Czech + 2,005 USA | INC | Czech + USA hold 5.1 % of real D40. The set adds D40 *labels* to domains where D40 is rare without adding pothole *appearance* — it is D40 label noise. Each synthetic image also duplicates all non-D40 labels of its source (7,821 boxes). | Controlled experiment S/R/D (Phase 5). |
| 17 | Synthetic generation code / prompts / filtering | — | UNS | Not on disk. Prompt, strength, seed, quality filter unknown. Generation is not reproducible. | Documented as an irrecoverable limitation; the audit treats the set as a black box. |
| 18 | YOLOv8 baseline D40 AP@50 = 0.243 | No log, no per-class output, no weights for this number on disk | UNS | The only surviving run (`yolov8s_optimized`) is a **12-class** model (all raw labels incl. D43/D44/D50/Repair), imgsz 416, 2-GPU Colab, unknown split; its per-class APs were not saved. The number cannot be traced or reproduced. | Re-run under the fixed protocol (B01). |
| 19 | Synthetic augmentation D40 AP@50 = 0.233 | none | UNS | Same as row 18. Also: a 0.010 difference from a single run has no uncertainty estimate. | S/R/D experiments with bootstrap CIs. |
| 20 | (implied) the failure is caused by a synthetic-to-real domain gap | — | UNS → largely INC | The audit shows the dominant problem is that most boxes contain **no pothole at all**. Realism of a pothole that is absent cannot be the cause. Remaining hypotheses (label noise, duplication, extra steps) are tested by S100 vs R100 vs D100 vs baseline. | Phase 5 controls. |
| 21 | Data-first roadmap recommends perspective-aware Copy-Paste and explicitly rejects GAN/diffusion as primary method | `BTP_Data_First_Final_Output.md` §2 | VS (document) | The method actually executed was SD inpainting — the option the project's own roadmap argued against — with none of the roadmap's validation steps (size check, nearest-neighbour, visual inspection) performed. | Note the inconsistency; run the validation that was promised (done here). |
| 22 | Maeda et al.: GAN synthesis improved F-measure by 5 % (small data) / 2 % (larger data) | Maeda et al., *Computer-Aided Civil and Infrastructure Engineering* 36:47–60 (2021) | VS | Supports "diminishing returns". The additional claim "doubling training data with GAN output can make F-measure worse" was **not** verified in the retrieved abstract. | Cite only the verified part. |

## 4. Baselines, architecture, metrics

| # | Claim | Current Evidence | Status | Problem | Required Fix |
|---|---|---|---|---|---|
| 23 | Previous YOLOv8s achieved mAP@50 = 0.404 | `results.csv` last epoch: mAP50 0.404, mAP50-95 0.233 | VE (number) / INC (interpretation) | 12-class mean incl. rare classes (D0w0 has 1 instance), imgsz 416, unknown split with near-dup leakage. Not comparable with any 4-class result. | Do not quote. |
| 24 | YOLOv8s vs YOLOv8n inconsistency | Surviving run is YOLOv8s | VE | — | Use YOLOv8s throughout. |
| 25 | D40 is the weakest class and the main detection failure | No in-house per-class result. External: a public YOLOv8s RDD2022 model card (merged 70/15/15 split) reports pothole AP50 **73.8** — the *best* of the 4 classes. | REQ | The project's central premise has never been measured; the only external per-class evidence found contradicts it. | B01 per-class AP on `split_v1` test. |
| 26 | `multires_road_damage.py`, ~39 M params, forward/backward tested | Not on disk | UNS | File missing; cannot be checked. | Re-implemented as `models/multires.py` (unit-tested: `models/test_multires.py`). |
| 27 | Dual-branch patch-8 / patch-16 ViT is the "high-res vs low-res" design | Design text | INC (as a contrast to baselines) | Patch 8 at 640 px = stride 8 = the P3 level that YOLOv8 and RT-DETR **already** use for detection. A patch-16 branch = P4. The proposed "multi-resolution" is therefore what modern detectors already do; only resolution *finer* than stride 8 (P2, stride 4) is new relative to the baseline. A from-scratch ViT would also confound resolution with pretraining. | Implemented as P2 high-res branch on a shared pretrained backbone (`models/multires.py` docstring). |
| 28 | Deformable cross-resolution attention is novel ("Level 2: not found anywhere in the literature") | Deformable DETR (Zhu et al., ICLR 2021) multi-scale deformable attention samples learned offsets **across feature levels**; RT-DETR's decoder uses it | INC | The mechanism (queries at one resolution sampling learned offsets in another) is the core of Deformable DETR. Bidirectional placement inside a PAN neck is at most an engineering variant. | Do not claim novelty; ablate whether it helps. |
| 29 | Adaptive fusion gate is a contribution | Gated feature fusion in FPNs exists (e.g. Gated Fully Fusion, Li et al., AAAI 2020) | INC (as novelty) | Prior art. | Ablate; no novelty claim. |
| 30 | RT-DETR decoder "reused unmodified" so gains are attributable to the backbone | Design text; never implemented | UNS | Never built. Under the 6 GB / 20-epoch budget an RT-DETR-decoder ablation series is infeasible; the ablation holds the YOLOv8 head fixed instead (same logic: only the multi-resolution components vary). RT-DETR-L is run as a separate baseline. | Documented deviation (`BASELINE_PROTOCOL.md`). |
| 31 | Metrics will be per class, per size, per country | Plan only | REQ | No such numbers existed. | `evaluation/evaluate.py` produces all of them for every model. |

## 5. Literature claims

| # | Claim | Current Evidence | Status | Problem | Required Fix |
|---|---|---|---|---|---|
| 32 | CRDDC'22 winner F1 ≈ 76–77 % | arXiv:2211.11362 (76 %); GDJ 2024 abstract (76.9 %) | VS | F1 on the private official test set — not comparable to any number we can compute. | Cite as context only. |
| 33 | "India 58.3 % vs USA 84.4 %" country breakdown | — | UNS | No source found (already flagged by the team). | Remove. |
| 34 | PROBE: 50.3 % on CRDDC'22 (WACV 2026) | arXiv:2511.12410 abstract | VS | Cross-domain-adaptation protocol and metric (mAP) on different splits; not comparable. "42.5 % baseline" not found. | Cite as context; remove 42.5 %. |
| 35 | YOLO-RD: 25.75 % on RDD2022-Japan, +4.93 % small objects (Sensors 2025, 25(5):1442) | PMC11902777 abstract | VS | Japan-only, metric not defined in abstract, different split. | Context only. |
| 36 | Pavement-DETR (Sensors 2025, 25(8):2426) "+7.7–8.9 mAP" | Paper exists; full text inaccessible (403 / captcha) | VS (existence) / UNS (number) | Also: UAV pavement dataset with 6 classes, **not RDD2022**. | Do not quote the number. |
| 37 | CrossViT, DiT (arXiv:2308.03409), Crack-Segmenter (arXiv:2510.10378), DMTC (Electronics 2023, 12(4):898), EcuPDD domain-transfer study (Frontiers in Built Environment, 2026, doi:10.3389/fbuil.2026.1895873) | Retrieved | VS (existence) | EcuPDD study published **26 Aug 2026**, not July. DMTC/Crack-Segmenter are segmentation papers on other datasets — not comparable. "Crack-Segmenter explicitly excludes potholes" and the "Dual-Path Sensors 2026 26(11):3286" paper were not re-verified. | See `LITERATURE_COMPARISON.md`. |

## 6. Claims about results, novelty and framing

| # | Claim | Current Evidence | Status | Problem | Required Fix |
|---|---|---|---|---|---|
| 38 | Any SOTA claim | None possible | INC | Official test labels are private; published numbers use different splits/metrics. A SOTA claim on RDD2022 is not establishable in this project. | Banned from the paper. |
| 39 | Improvement percentages for the architecture | None measured | UNS | — | Only report measured, CI-bounded differences. |
| 40 | Domain generalization claims | No experiment | REQ | — | X01: train without India → test on India (and on all domains). |
| 41 | "The architecture addresses small-object detection" | No experiment; see rows 9–10 | REQ | At 640 px most small boxes are Norway objects shrunk below a few pixels; an architecture working on the same 640 input cannot recover destroyed pixels. | Size-stratified + per-domain AP for A0–A7. |
| 42 | Contribution = "causal attribution of architecture vs data vs domain" | Plan | REQ | The proposed 2×2 design is reasonable; but "causal" is too strong for observational domain differences — domain and class are confounded in the data (row 8). | Use "controlled comparison", not "causal". |
