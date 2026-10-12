"""Audit the pre-existing Stable-Diffusion-inpainted D40 set (data/synthetic_yolo_format, 4,005 images).

The generation code was not preserved, so this audit treats the images as a black box and measures, for every
synthetic image vs. its real source image (same file name minus the 'synthetic_D40_<k>_' prefix):
  * which label lines are new (the inserted D40 boxes) vs copied from the source annotation
  * mean absolute pixel change inside each inserted box vs outside all inserted boxes
  * box geometry (relative area, vertical centre) vs. real D40 boxes
  * whether the source image sits in train / val / test of split_v1 (leakage if not train)
Output: results/synthetic_audit/synthetic_boxes.csv, summary.json, figures in visualizations/synthetic_audit/
"""
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data/synthetic_yolo_format"
RAW = ROOT / "data/raw/RDD2022/RDD2022_all_countries"
OUT = ROOT / "results/synthetic_audit"
FIG = ROOT / "visualizations/synthetic_audit"


def src_name(name):
    s = re.sub(r"^synthetic_D40_\d+_", "", name)
    s = re.sub(r"^synthetic_D40_", "", s)
    return s


def read_lines(p):
    return [l.split() for l in Path(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []


def job(stem):
    src = src_name(stem + ".jpg")[:-4]
    subset = src.rsplit("_", 1)[0]
    syn = cv2.imread(str(SYN / "images" / f"{stem}.jpg"))
    orig = cv2.imread(str(RAW / subset / subset / "train/images" / f"{src}.jpg"))
    syn_l = read_lines(SYN / "labels" / f"{stem}.txt")
    src_l = read_lines(ROOT / f"data/yolo_format/labels/train/{src}.txt")
    src_set = {tuple(round(float(v), 4) for v in l) for l in src_l}
    rows = []
    if orig is None or syn is None:
        return [dict(stem=stem, source=src, subset=subset, status="source_missing")]
    if orig.shape != syn.shape:
        orig = cv2.resize(orig, (syn.shape[1], syn.shape[0]))
    h, w = syn.shape[:2]
    diff = np.abs(syn.astype(np.int16) - orig.astype(np.int16)).mean(2)
    new = [l for l in syn_l if tuple(round(float(v), 4) for v in l) not in src_set]
    mask = np.zeros((h, w), bool)
    boxes = []
    for l in new:
        c, x, y, bw, bh = map(float, l)
        x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), int(np.ceil((x + bw / 2) * w)), int(np.ceil((y + bh / 2) * h))
        x1, y1 = max(x1, 0), max(y1, 0)
        boxes.append((int(c), x, y, bw, bh, x1, y1, x2, y2))
        mask[y1:y2, x1:x2] = True
    out_diff = float(diff[~mask].mean()) if (~mask).any() else float("nan")
    for c, x, y, bw, bh, x1, y1, x2, y2 in boxes:
        d = diff[y1:y2, x1:x2]
        rows.append(dict(stem=stem, source=src, subset=subset, status="ok", cls=c, xc=x, yc=y, bw=bw, bh=bh,
                         rel_area=bw * bh, in_box_mad=float(d.mean()), in_box_frac_changed_gt10=float((d > 10).mean()),
                         outside_mad=out_diff, n_new_boxes=len(boxes), n_src_boxes=len(src_l), n_syn_boxes=len(syn_l)))
    if not boxes:
        rows.append(dict(stem=stem, source=src, subset=subset, status="no_new_box", outside_mad=out_diff, n_src_boxes=len(src_l), n_syn_boxes=len(syn_l)))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (SYN / "images").glob("*.jpg"))
    with ProcessPoolExecutor() as ex:
        rows = [r for rs in ex.map(job, stems, chunksize=32) for r in rs]
    df = pd.DataFrame(rows)
    spl = pd.read_csv(ROOT / "data/splits/split_v1.csv").set_index("image_id").split
    df["source_split_v1"] = df.source.map(spl).fillna("unknown")
    df.to_csv(OUT / "synthetic_boxes.csv", index=False)
    ok = df[df.status == "ok"]
    real = pd.read_csv(ROOT / "data/processed/annotations_all.csv")
    real = real[real.label == "D40"]
    real_rel = (real.x2 - real.x1) * (real.y2 - real.y1) / (real.img_w * real.img_h)
    real_yc = ((real.y1 + real.y2) / 2 / real.img_h)
    S = {
        "n_synthetic_images": len(stems),
        "n_unique_sources": int(df.source.nunique()),
        "status_counts": df.drop_duplicates("stem").status.value_counts().to_dict(),
        "n_inserted_boxes": int(len(ok)),
        "inserted_box_classes": ok.cls.value_counts().to_dict(),
        "source_subsets": df.drop_duplicates("stem").subset.value_counts().to_dict(),
        "source_split_v1_images": df.drop_duplicates("stem").source_split_v1.value_counts().to_dict(),
        "copied_non_D40_boxes_total": int(df.drop_duplicates("stem").n_src_boxes.sum()),
        "in_box_mean_abs_diff_quantiles": ok.in_box_mad.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
        "outside_mean_abs_diff_quantiles": ok.outside_mad.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(),
        "pct_boxes_in_box_change_lt_outside_x2": float(100 * (ok.in_box_mad < 2 * ok.outside_mad).mean()),
        "pct_boxes_in_box_mad_lt_3": float(100 * (ok.in_box_mad < 3).mean()),
        "pct_boxes_center_above_image_middle": float(100 * (ok.yc < 0.5).mean()),
        "real_D40_pct_center_above_image_middle": float(100 * (real_yc < 0.5).mean()),
        "rel_area_median_synthetic": float(ok.rel_area.median()),
        "rel_area_median_real_D40": float(real_rel.median()),
        "boxes_per_source_image_max": int(df.drop_duplicates("stem").groupby("source").size().max()),
    }
    (OUT / "summary.json").write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps(S, indent=1, default=str))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].hist(ok.in_box_mad, bins=60, color="#2a78d6", label="inside inserted box")
    axes[0].hist(ok.outside_mad, bins=60, color="#eb6834", alpha=0.7, label="rest of image")
    axes[0].set_xlabel("mean |synthetic − source| (8-bit)"); axes[0].legend(frameon=False); axes[0].set_title("Pixel change vs source", loc="left")
    b = np.linspace(0, 1, 41)
    axes[1].hist(real_yc, bins=b, density=True, histtype="step", lw=2, color="#1baf7a", label="real D40")
    axes[1].hist(ok.yc, bins=b, density=True, histtype="step", lw=2, color="#e34948", label="synthetic D40")
    axes[1].set_xlabel("box centre y (0=top)"); axes[1].legend(frameon=False); axes[1].set_title("Vertical placement", loc="left")
    bl = np.logspace(-5, 0, 50)
    axes[2].hist(real_rel, bins=bl, density=True, histtype="step", lw=2, color="#1baf7a", label="real D40")
    axes[2].hist(ok.rel_area, bins=bl, density=True, histtype="step", lw=2, color="#e34948", label="synthetic D40")
    axes[2].set_xscale("log"); axes[2].set_xlabel("box area / image area"); axes[2].legend(frameon=False); axes[2].set_title("Relative size", loc="left")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "synthetic_vs_real_stats.png", dpi=150)


if __name__ == "__main__":
    main()
