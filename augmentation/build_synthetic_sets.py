"""Build the controlled training sets for the synthetic-data experiment (Phase 5).

Eligible synthetic images = those whose real source image lies in split_v1 TRAIN (synthetic images made from
val/test sources are excluded: they would leak test scenes into training).

For each eligible synthetic image s with source image o and inserted box(es) b:
  S  (synthetic)     : pixels of s,  labels = labels(o) + b        <- the original idea
  R  (relabel-only)  : pixels of o,  labels = labels(o) + b        <- same boxes, NO generated pixels
  D  (duplicate-only): pixels of o,  labels = labels(o)            <- just more copies of the same backgrounds
S vs R isolates what the generated pixels contribute; R vs D isolates the inserted labels; D vs baseline isolates
duplication / extra optimisation steps.

Ratios r in {25, 50, 100}% of the eligible pool (nested random subsets, seed 0) for S; R and D at 100%.
Writes data/splits/synth_v1/{images,labels}/train/ and configs/data/rdd_v1_<S|R|D><r>.yaml (train = txt list).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/splits/rdd_v1"
OUT = ROOT / "data/splits/synth_v1"
CFG = ROOT / "configs/data"


def main():
    boxes = pd.read_csv(ROOT / "results/synthetic_audit/synthetic_boxes.csv")
    boxes = boxes[(boxes.status == "ok") & (boxes.source_split_v1 == "train")]
    stems = sorted(boxes.stem.unique())
    rng = np.random.default_rng(0)
    order = list(rng.permutation(stems))
    (OUT / "images/train").mkdir(parents=True, exist_ok=True)
    (OUT / "labels/train").mkdir(parents=True, exist_ok=True)
    by_stem = {k: g for k, g in boxes.groupby("stem")}
    for s in stems:
        g = by_stem[s]
        src = g.source.iloc[0]
        base_lbl = (BASE / "labels/train" / f"{src}.txt").read_text()
        new = "".join(f"3 {r.xc:.6f} {r.yc:.6f} {r.bw:.6f} {r.bh:.6f}\n" for r in g.itertuples())
        for kind, img_src, lbl in (("S", ROOT / "data/synthetic_yolo_format/images" / f"{s}.jpg", base_lbl + new),
                                   ("R", BASE / "images/train" / f"{src}.jpg", base_lbl + new),
                                   ("D", BASE / "images/train" / f"{src}.jpg", base_lbl)):
            name = f"{kind}__{s}"
            dst = OUT / "images/train" / f"{name}.jpg"
            if not dst.exists():
                os.symlink(img_src, dst)
            (OUT / "labels/train" / f"{name}.txt").write_text(lbl)
    base_list = sorted(str(p) for p in (BASE / "images/train").glob("*.jpg"))
    names = "names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n"
    plan = [("S", 25), ("S", 50), ("S", 100), ("R", 100), ("D", 100)]
    for kind, r in plan:
        k = round(len(order) * r / 100)
        extra = [str(OUT / "images/train" / f"{kind}__{s}.jpg") for s in order[:k]]
        lst = ROOT / f"data/splits/train_{kind}{r}.txt"
        lst.write_text("\n".join(base_list + extra) + "\n")
        (CFG / f"rdd_v1_{kind}{r}.yaml").write_text(
            f"path: {BASE}\ntrain: {lst}\nval: images/val\ntest: images/test\n{names}")
        print(f"{kind}{r}: base {len(base_list)} + extra {len(extra)} images")
    print(f"eligible synthetic images: {len(stems)}, inserted boxes: {len(boxes)}")


if __name__ == "__main__":
    main()
