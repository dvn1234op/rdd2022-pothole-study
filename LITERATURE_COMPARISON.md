# Literature Comparison

Retrieved 2026-10-04. "Retrieved" = the abstract or results page was actually read during this audit; numbers
are quoted only from retrieved text. Rows marked *not retrieved* list the work for context but quote no numbers.

## 0. Can any published number be compared with ours?

**No published RDD2022 number in this table is directly comparable with our results.** Reasons, in order of
importance:
1. Official CRDDC'22 test labels are private. Every paper below that reports mAP on RDD2022 used its **own**
   random split (or a subset / single country). None of these splits is published as a file list, and none
   controls near-duplicate leakage — our audit found 4.9 % of a random split's val images have a near-duplicate in
   train, which inflates scores.
2. Different class sets (4 vs 6–12 classes), input sizes, epoch budgets (20 here vs 50–300) and metrics
   (F1, mAP@50, mAP@50:95, "accuracy").
3. Our budget is deliberately reduced and identical across our models; absolute values are expected to be lower
   than fully converged literature numbers.

Therefore: literature numbers are context, never a column in our results tables, and no SOTA claim is made.

## 1. RDD2022 detection results

| Paper | Year | Architecture | Data / split | Input | Metric | Reported result | Directly comparable? |
|---|---|---|---|---|---|---|---|
| Arya et al., "RDD2022: A multi-national image dataset…", *Geoscience Data Journal* 11(4):846–862 (arXiv:2209.08538) — retrieved | 2024 | dataset paper; best CRDDC'22 model | official private test, all 6 countries | — | F1 | top model F1 = 76.9 % | No (F1 on private test) |
| Arya et al., "CRDDC'2022", arXiv:2211.11362 — retrieved | 2022 | ensemble of YOLO + Faster-RCNN models | official private test, 6 countries | — | F1 | 76 % | No |
| Xiao et al., "PROBE: Self-Supervised Visual Prompting for Cross-Domain Road Damage Detection", WACV 2026 (arXiv:2511.12410) — retrieved | 2026 | frozen ViT + self-supervised prompts + MMD alignment | cross-domain adaptation protocol on CRDDC'22 | n/r | mAP | 50.3 % on CRDDC'22 (+2.1 over CDTrans) | No (unsupervised domain adaptation protocol) |
| Wang et al., "YOLO-RD", *Sensors* 25(5):1442 — retrieved | 2025 | YOLOv8 + Star Op + MAF + wavelet conv | RDD2022 **Japan** only | n/r | "detection accuracy" (abstract) | 25.75 %; +4.93 % small-object over YOLOv8 | No (single country; metric undefined in abstract) |
| YOLOv8-PD, *Scientific Reports* 2024 (PMC11130172) — retrieved | 2024 | improved YOLOv8n | **random 10,000-image subset** of RDD2022, 8:2 train/val (val = test) | 640 | mAP@50, mAP@50:95, per class | YOLOv8n baseline: all 69.2 / 38.0; D00 74.2, D10 79.3, D20 71.9, **D40 51.2** (AP@50). YOLOv8-PD: all 70.6 / 39.5; D40 53.1 | No (subset, random split without dedup, 300 epochs) |
| dronefreak/rdd2022-yolov8s (Hugging Face model card) — retrieved | 2025 | YOLOv8s | merged 6 countries, random 70/15/15, 15 % held-out test | 640 | mAP@50 / @50:95 | all 61.45 / 33.53; D00 54.95, D10 54.18, D20 62.91, **D40 73.77** (AP@50) | No (unpublished split; model card, not peer reviewed) |
| PD-DETR (search-engine summary only) — *not retrieved* | n/r | RT-DETR-based | "RDD2022" | n/r | mAP | — (not quoted) | No |

Observation (context only): the two retrieved per-class results disagree on whether D40 is the weakest
(YOLOv8-PD) or the strongest (model card) class. Because both use undocumented random splits, the question can
only be settled on a fixed, leakage-controlled split — which is what our B01 does.

## 2. Road / pavement damage methods relevant to the proposed mechanisms (other datasets)

| Paper | Year | What it does | Dataset | Relevance / comparability |
|---|---|---|---|---|
| Zuo et al., "Pavement-DETR", *Sensors* 25(8):2426 — retrieved (abstract only) | 2025 | RT-DETR + channel-spatial shuffle attention + Conv3XC + combined loss for small/overlapping defects | UAV pavement imagery, 6 classes (incl. oblique cracks, repairs) | Not RDD2022; the "+7.7–8.9 mAP" figure in earlier project docs could not be verified (full text inaccessible). |
| Xu et al., "DMTC", *Electronics* 12(4):898 — retrieved (abstract) | 2023 | Transformer with cross-shaped attention, segmentation | own pothole/pavement dataset | Segmentation; not comparable. |
| Kyem et al., "Crack-Segmenter" (arXiv:2510.10378) — retrieved | 2025 | self-supervised multi-scale transformer + attention-guided fusion, pixel-level crack segmentation | 10 public crack datasets | Segmentation, cracks only. Multi-scale + attention-guided fusion for pavement already exists. |
| EcuPDD domain-transfer study, *Frontiers in Built Environment*, doi:10.3389/fbuil.2026.1895873 — retrieved | 2026 (26 Aug) | YOLOv8n trained with RDD2022-Japan / China_MotorBike / McMaster; Gini for class imbalance; Domain Gap Index = 1 − mAP50_Ecuador / mAP50_general | Ecuador + RDD2022 subsets | Closest prior work for the domain-gap framing: shows international data does not close the gap to a new country. Different target (external country, not RDD2022-internal hold-out). |
| Maeda et al., "Generative adversarial network for road damage detection", *Computer-Aided Civil and Infrastructure Engineering* 36:47–60 — retrieved | 2021 | PG-GAN potholes + Poisson blending | RDD2018/2019 | F-measure +5 % (small data) / +2 % (larger data): gains shrink with data size. |
| Siripathitti, Forest, Fink, "Cut-and-Paste with Precision: a content and perspective-aware data augmentation for road damage detection", ESREL 2024 (arXiv:2406.18586) — retrieved (abstract) | 2024 | road-mask (Panoptic-DeepLab) + perspective-matched copy-paste | RDD2022 (subset details not retrieved) | States naive random copy-paste has limited effectiveness for road damage because it ignores road location and perspective — exactly the failure mode our synthetic audit found (random y-band placement, 38 % of boxes off-road). |

## 3. Generic mechanisms (prior art for the proposed components)

| Work | Year | Mechanism | Overlap with the proposed architecture |
|---|---|---|---|
| Lin et al., Feature Pyramid Networks, CVPR 2017 | 2017 | top-down fusion of low-resolution context into high-resolution maps | multi-resolution fusion (the basic idea) |
| Liu et al., PANet, CVPR 2018 | 2018 | bottom-up path added to FPN | YOLOv8's neck = FPN+PAN: already bidirectional multi-resolution fusion |
| Chen, Fan, Panda, CrossViT, ICCV 2021 (arXiv:2103.14899) | 2021 | dual-branch (small/large patch) ViT, cross-attention between branches via CLS tokens | dual-patch-size branches + cross-branch attention (classification) |
| Zhu et al., Deformable DETR, ICLR 2021 (arXiv:2010.04159) | 2021 | multi-scale deformable attention: each query predicts sampling offsets and weights **across several feature levels** | essentially the proposed "deformable cross-resolution attention" |
| Zhao et al., RT-DETR ("DETRs Beat YOLOs on Real-time Object Detection"), CVPR 2024 (arXiv:2304.08069) | 2024 | hybrid encoder (intra-scale attention + cross-scale CNN fusion) + deformable decoder | the proposal's own decoder already performs cross-level deformable sampling |
| Li et al., Gated Fully Fusion, AAAI 2020 (arXiv:1904.01803) | 2020 | gated fusion of multi-level features | the "adaptive fusion gate" |
| Ma, Fei, Huang, DiT: Dynamic Token Routing (arXiv:2308.03409) | 2023 | data-dependent routing of tokens between fine/coarse paths, incl. detection | the "scale-conditioned routing" extension |
| Robinson et al., RF-DETR, ICLR 2026 (arXiv:2511.09554) — retrieved | 2025/26 | NAS over a DINOv2-backbone DETR for target datasets | strong modern baseline; not run here (6 GB memory budget) |
| Akyon et al., SAHI, ICIP 2022 (arXiv:2202.06934) | 2022 | sliced (tiled) inference/fine-tuning for small objects | the direct alternative to architecture for down-sampled small objects (relevant to Norway) |

## 4. Implication for novelty (before experiments)
Every component of the proposed model has direct prior art (rows in §3). The only defensible novelty space left
is (a) empirical/analytical findings about RDD2022 (domain-class entanglement, scale-domain entanglement,
leakage, the synthetic-data failure analysis) and (b) a carefully controlled ablation of whether these mechanisms
help road-damage detection under a fixed budget. Final novelty assessment: `RESULTS.md` §7, after the experiments.
