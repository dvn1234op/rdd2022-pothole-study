"""Pilot: choose and build ONE real-data-only compensation method from the TRAIN-split statistics.

    python augmentation/compensation.py          (analysis + builds the chosen training list)

1. Training-split distribution: class counts, domain counts, D40 count and share by domain, Cramer's V.
2. Simulates candidate samplers WITHOUT training (all use repeat-factor sampling, Gupta et al., LVIS, CVPR 2019,
   r = max(1, sqrt(t / f)), image repeat = max over its categories), each calibrated to the SAME total exposure
   (1.25x the real train set, fixed a priori):
     A  class-aware RFS         categories = class                  (f = fraction of train images containing it)
     C  domain-class-aware RFS  categories = (domain, class) cells
   and reports, for the resulting EXPOSURE distribution: class shares, D40 share coming from India, and the
   domain x class Cramer's V of the exposure (lower = less entanglement).
3. Builds the chosen sampler's training list (symlinks rK__<image>.jpg, stochastic rounding, seed 0), and the
   epoch count that keeps the number of optimisation steps equal to the 20-epoch baseline.
Val/test are untouched.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/splits/rdd_v1"
OUT = ROOT / "data/splits/comp_v1"
RES = ROOT / "results/compensation"
C = ["D00", "D10", "D20", "D40"]
DOMS = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]
TARGET_EXPOSURE = 1.25
BASE_EPOCHS = 20


def cramers_v(t):
    chi2 = chi2_contingency(t)[0]
    n = t.sum(); r, k = t.shape
    return float(np.sqrt(chi2 / n / (min(r, k) - 1)))


def rfs(img_cells, freq, t):
    return np.array([max([1.0] + [max(1.0, np.sqrt(t / freq[c])) for c in cells]) for cells in img_cells])


def calibrate(img_cells, freq):
    lo, hi = 1e-6, 1.0
    for _ in range(60):
        t = np.sqrt(lo * hi)
        if rfs(img_cells, freq, t).mean() > TARGET_EXPOSURE: hi = t
        else: lo = t
    return np.sqrt(lo * hi)


def exposure(boxes, rep):
    w = boxes.image_id.map(rep)
    return pd.crosstab(boxes.subset, boxes.label, values=w, aggfunc="sum").reindex(index=DOMS, columns=C).fillna(0)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    spl = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    train = spl[spl.split == "train"].set_index("image_id")
    ann = pd.read_csv(ROOT / "data/processed/annotations_all.csv")
    ann = ann[ann.label.isin(C) & (ann.x2 > ann.x1) & (ann.y2 > ann.y1) & ann.image_id.isin(train.index)]
    S = {}
    ct = pd.crosstab(ann.subset, ann.label).reindex(index=DOMS, columns=C).fillna(0).astype(int)
    S["train_images"] = int(len(train)); S["train_images_by_domain"] = train.subset.value_counts().reindex(DOMS).to_dict()
    S["train_boxes_by_class"] = ct.sum().to_dict()
    S["class_ratio_D00_D40"] = float(ct.D00.sum() / ct.D40.sum())
    d40 = pd.DataFrame({"D40_boxes": ct.D40, "D40_pct_of_domain_boxes": (100 * ct.D40 / ct.sum(1)).round(2),
                        "pct_of_all_D40": (100 * ct.D40 / ct.D40.sum()).round(2),
                        "images": train.subset.value_counts().reindex(DOMS),
                        "images_with_D40": ann[ann.label == "D40"].groupby("subset").image_id.nunique().reindex(DOMS).fillna(0).astype(int)})
    d40.to_csv(RES / "train_d40_by_domain.csv"); ct.to_csv(RES / "train_contingency.csv")
    S["train_cramers_v_domain_class"] = cramers_v(ct.values)

    # candidate samplers
    img_cls = ann.groupby("image_id").label.agg(lambda s: sorted(set(s)))
    img_cell = ann.assign(cell=ann.subset + "|" + ann.label).groupby("image_id").cell.agg(lambda s: sorted(set(s)))
    ids = train.index.tolist()
    cells_A = [img_cls.get(i, []) for i in ids]
    cells_C = [img_cell.get(i, []) for i in ids]
    n = len(ids)
    fA = {c: sum(c in x for x in cells_A) / n for c in C}
    fC = {}
    for x in cells_C:
        for c in x: fC[c] = fC.get(c, 0) + 1 / n
    cands = {}
    for name, cells, f in (("none", None, None), ("A_class_RFS", cells_A, fA), ("C_domain_class_RFS", cells_C, fC)):
        if cells is None:
            rep = np.ones(n); t = None
        else:
            t = calibrate(cells, f); rep = rfs(cells, f, t)
        e = exposure(ann, dict(zip(ids, rep)))
        cands[name] = {"t": t, "exposure_factor": float(rep.mean()), "images_repeated_pct": float(100 * (rep > 1).mean()),
                       "class_share_pct": (100 * e.sum() / e.values.sum()).round(2).to_dict(),
                       "D40_exposure_from_India_pct": float(100 * e.loc["India", "D40"] / e.D40.sum()),
                       "D40_exposure_from_India_Japan_pct": float(100 * e.loc[["India", "Japan"], "D40"].sum() / e.D40.sum()),
                       "domain_share_pct": (100 * e.sum(1) / e.values.sum()).round(2).to_dict(),
                       "D40_share_within_domain_pct": (100 * e.D40 / e.sum(1)).round(2).to_dict(),
                       "cramers_v_exposure": cramers_v(e.values)}
        cands[name]["_rep"] = rep
    S["candidates"] = {k: {kk: vv for kk, vv in v.items() if kk != "_rep"} for k, v in cands.items()}

    # ---- decision rule (fixed before any compensated model is trained) -------------------------------------
    # Prefer the sampler that increases D40 exposure without increasing domain-class entanglement.
    a, c = S["candidates"]["A_class_RFS"], S["candidates"]["C_domain_class_RFS"]
    chosen = "C_domain_class_RFS" if c["cramers_v_exposure"] < a["cramers_v_exposure"] else "A_class_RFS"
    S["chosen"] = chosen

    # ---- build the chosen list ---------------------------------------------------------------------------------
    rep = cands[chosen]["_rep"]
    rng = np.random.default_rng(0)
    copies = np.floor(rep).astype(int) + (rng.random(n) < rep - np.floor(rep))
    (OUT / "images/train").mkdir(parents=True, exist_ok=True); (OUT / "labels/train").mkdir(parents=True, exist_ok=True)
    lines = []
    for i, k in zip(ids, copies):
        for r in range(k):
            name = f"r{r}__{i}"
            dst = OUT / "images/train" / f"{name}.jpg"
            if not dst.exists():
                os.symlink(BASE / "images/train" / f"{i}.jpg", dst)
            lbl = OUT / "labels/train" / f"{name}.txt"
            if not lbl.exists():
                lbl.write_text((BASE / "labels/train" / f"{i}.txt").read_text())
            lines.append(str(dst))
    lst = ROOT / "data/splits/train_comp_v1.txt"
    lst.write_text("\n".join(lines) + "\n")
    epochs = round(BASE_EPOCHS * n / len(lines))
    S["built"] = {"list": str(lst.relative_to(ROOT)), "n_entries": len(lines), "realised_exposure": len(lines) / n,
                  "epochs_for_equal_steps": epochs, "steps_ratio_vs_baseline": epochs * len(lines) / (BASE_EPOCHS * n),
                  "close_mosaic": max(1, round(5 * epochs / BASE_EPOCHS))}
    (ROOT / "configs/data/rdd_v1_comp.yaml").write_text(
        f"path: {BASE}\ntrain: {lst}\nval: images/val\ntest: images/test\nnames:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n")
    (RES / "compensation_summary.json").write_text(json.dumps(S, indent=2, default=float))
    print(d40.to_string()); print(ct.to_string())
    print(json.dumps({k: v for k, v in S.items() if k not in ("candidates",)}, indent=1, default=float))
    for k, v in S["candidates"].items():
        print(f"\n{k}: exposure {v['exposure_factor']:.2f}  repeated {v['images_repeated_pct']:.1f}%  V={v['cramers_v_exposure']:.3f}  "
              f"D40 share {v['class_share_pct']['D40']}%  D40-from-India {v['D40_exposure_from_India_pct']:.1f}%  "
              f"D40-from-India+Japan {v['D40_exposure_from_India_Japan_pct']:.1f}%\n   D40 share within domain: {v['D40_share_within_domain_pct']}")


if __name__ == "__main__":
    main()
