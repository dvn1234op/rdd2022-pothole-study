"""RDD2022 dataset audit: class / domain / size statistics and domain-class entanglement.

Reads data/processed/{images_all,annotations_all}.csv (from parse_rdd.py).
Writes results/dataset_audit/*.csv, results/dataset_audit/audit_summary.json and
visualizations/dataset_audit/*.png.

Size buckets are defined on the MODEL-INPUT scale (image letterboxed so the long side is 640 px),
because that is what every detector in this project actually sees:
    small  : area < 32^2      medium : 32^2 <= area < 96^2      large : area >= 96^2
Native-resolution buckets are also reported because Norway images are ~4000 px wide.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
OUT = ROOT / "results/dataset_audit"
FIG = ROOT / "visualizations/dataset_audit"
TARGET = ["D00", "D10", "D20", "D40"]
SUBSETS = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CLS_COLOR = dict(zip(TARGET, SERIES[:4]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def size_bucket(area):
    return np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))


def cramers_v(table):
    """Bias-corrected Cramer's V (Bergsma 2013) plus raw V and chi2 test."""
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.sum()
    r, k = table.shape
    phi2 = chi2 / n
    v = np.sqrt(phi2 / (min(r, k) - 1))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    vc = np.sqrt(phi2c / (min(rc, kc) - 1))
    return dict(chi2=float(chi2), p=float(p), dof=int(dof), n=int(n), cramers_v=float(v), cramers_v_corrected=float(vc)), expected


def normalized_mi(table):
    pj = table / table.sum()
    pr, pc = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    nz = pj > 0
    mi = (pj[nz] * np.log(pj[nz] / (pr @ pc)[nz])).sum()
    h = lambda q: -(q[q > 0] * np.log(q[q > 0])).sum()
    return float(mi), float(mi / h(pc.ravel())), float(mi / np.sqrt(h(pr.ravel()) * h(pc.ravel())))


def js_div(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0]))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    img = pd.read_csv(P / "images_all.csv")
    ann = pd.read_csv(P / "annotations_all.csv")
    ann["bw"], ann["bh"] = ann.x2 - ann.x1, ann.y2 - ann.y1
    degenerate = ann[(ann.bw <= 0) | (ann.bh <= 0)]
    ann = ann[(ann.bw > 0) & (ann.bh > 0)].copy()
    scale = 640.0 / ann[["img_w", "img_h"]].max(axis=1)
    ann["area_native"] = ann.bw * ann.bh
    ann["area640"] = ann.area_native * scale ** 2
    ann["rel_area"] = ann.area_native / (ann.img_w * ann.img_h)
    ann["size640"] = size_bucket(ann.area640)
    ann["size_native"] = size_bucket(ann.area_native)
    tgt = ann[ann.label.isin(TARGET)].copy()
    S = {}

    # ---- 1. global counts ---------------------------------------------------------------
    S["n_images_labeled"] = int(len(img))
    S["n_images_with_any_box"] = int((img.n_boxes_raw > 0).sum())
    S["n_images_with_target_box"] = int(tgt.image_id.nunique())
    S["n_images_without_target_box"] = int(len(img) - tgt.image_id.nunique())
    S["n_boxes_all_labels"] = int(len(ann) + len(degenerate))
    S["n_degenerate_boxes"] = int(len(degenerate))
    S["n_boxes_target"] = int(len(tgt))
    S["raw_label_counts"] = {k: int(v) for k, v in pd.concat([ann.label, degenerate.label]).value_counts().items()}
    cls_counts = tgt.label.value_counts().reindex(TARGET)
    S["target_class_counts"] = {k: int(v) for k, v in cls_counts.items()}
    S["class_imbalance_ratio_max_min"] = float(cls_counts.max() / cls_counts.min())
    S["class_gini"] = gini(cls_counts.values)
    img_per_cls = tgt.groupby("label").image_id.nunique().reindex(TARGET)
    S["images_containing_class"] = {k: int(v) for k, v in img_per_cls.items()}

    # ---- 2. domain (subset) counts ------------------------------------------------------
    dom = img.groupby("subset").agg(images=("image_id", "size")).reindex(SUBSETS)
    dom["images_with_target"] = tgt.groupby("subset").image_id.nunique().reindex(SUBSETS)
    dom["target_boxes"] = tgt.groupby("subset").size().reindex(SUBSETS)
    dom["boxes_per_labeled_image"] = dom.target_boxes / dom.images_with_target
    dom["share_of_images_%"] = 100 * dom.images / dom.images.sum()
    dom["share_of_target_boxes_%"] = 100 * dom.target_boxes / dom.target_boxes.sum()
    res = img.groupby("subset").apply(lambda d: ", ".join(f"{w}x{h}:{n}" for (w, h), n in d.groupby(["width", "height"]).size().items()), include_groups=False)
    dom["resolutions"] = res.reindex(SUBSETS)
    dom.to_csv(OUT / "domain_counts.csv")
    S["domain_gini_images"] = gini(dom.images.values)
    S["domain_imbalance_ratio_images_max_min"] = float(dom.images.max() / dom.images.min())

    # ---- 3. country x class contingency -------------------------------------------------
    ct = pd.crosstab(tgt.subset, tgt.label).reindex(index=SUBSETS, columns=TARGET).fillna(0).astype(int)
    ct.to_csv(OUT / "contingency_counts.csv")
    row_norm = 100 * ct.div(ct.sum(1), axis=0)
    col_norm = 100 * ct.div(ct.sum(0), axis=1)
    row_norm.round(2).to_csv(OUT / "contingency_row_normalized_pct.csv")
    col_norm.round(2).to_csv(OUT / "contingency_col_normalized_pct.csv")
    stats, expected = cramers_v(ct.values)
    std_resid = (ct.values - expected) / np.sqrt(expected)
    pd.DataFrame(std_resid, index=SUBSETS, columns=TARGET).round(1).to_csv(OUT / "contingency_pearson_residuals.csv")
    mi, nmi_cls, nmi_sym = normalized_mi(ct.values.astype(float))
    pooled = ct.sum(0).values / ct.values.sum()
    S["box_level_association"] = {**stats, "mutual_information_nats": mi,
                                  "uncertainty_coeff_class_given_domain": nmi_cls, "nmi_sqrt": nmi_sym,
                                  "js_divergence_to_pooled_bits": {s: js_div(ct.loc[s].values / ct.loc[s].sum(), pooled) for s in SUBSETS}}

    # Image-level association (boxes within an image are not independent; this is the safer test):
    pres = pd.crosstab(tgt.image_id, tgt.label).reindex(columns=TARGET).fillna(0) > 0
    pres = pres.join(img.set_index("image_id").subset).reindex(img.image_id).fillna({c: False for c in TARGET})
    pres["subset"] = img.set_index("image_id").subset
    d40_img = pd.crosstab(pres.subset, pres.D40.astype(bool)).reindex(SUBSETS)
    st_img, _ = cramers_v(d40_img.values)
    S["image_level_D40_presence_vs_domain"] = st_img
    d40 = pd.DataFrame({
        "D40_boxes": ct.D40,
        "D40_pct_of_domain_boxes": row_norm.D40.round(2),
        "pct_of_all_D40": col_norm.D40.round(2),
        "images_with_D40": d40_img[True],
        "pct_images_with_D40": (100 * d40_img[True] / d40_img.sum(1)).round(2),
        "domain_share_of_all_target_boxes_%": dom["share_of_target_boxes_%"].round(2),
    })
    d40["over_representation_factor"] = (d40.pct_of_all_D40 / d40["domain_share_of_all_target_boxes_%"]).round(2)
    d40.to_csv(OUT / "d40_by_domain.csv")

    # ---- 4. box sizes -------------------------------------------------------------------
    sz = pd.crosstab([tgt.label], tgt.size640).reindex(columns=["small", "medium", "large"]).fillna(0).astype(int)
    sz.to_csv(OUT / "size640_by_class.csv")
    szd = pd.crosstab([tgt.subset], tgt.size640).reindex(index=SUBSETS, columns=["small", "medium", "large"]).fillna(0).astype(int)
    szd.to_csv(OUT / "size640_by_domain.csv")
    szdc = pd.crosstab([tgt.subset, tgt.label], tgt.size640).reindex(columns=["small", "medium", "large"]).fillna(0).astype(int)
    szdc.to_csv(OUT / "size640_by_domain_class.csv")
    szn = pd.crosstab([tgt.label], tgt.size_native).reindex(columns=["small", "medium", "large"]).fillna(0).astype(int)
    szn.to_csv(OUT / "size_native_by_class.csv")
    q = tgt.groupby(["subset", "label"]).rel_area.describe(percentiles=[.25, .5, .75])[["count", "25%", "50%", "75%"]]
    (q * [1, 100, 100, 100]).round(3).rename(columns=lambda c: c if c == "count" else f"rel_area_{c}_pct").to_csv(OUT / "relative_area_quantiles.csv")
    S["size640_overall"] = {k: int(v) for k, v in tgt.size640.value_counts().items()}
    S["size640_D40"] = {k: int(v) for k, v in tgt[tgt.label == "D40"].size640.value_counts().items()}

    # ---- 5. figures ---------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK2, "figure.dpi": 150})
    short = {"China_Drone": "China-Drone", "China_MotorBike": "China-Moto", "United_States": "USA"}
    lab = [short.get(s, s) for s in SUBSETS]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(SUBSETS))
    bars = ax.bar(x, ct.D40.values, width=0.6, color=CLS_COLOR["D40"])
    for b, v, pct in zip(bars, ct.D40.values, col_norm.D40.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, lab); ax.set_ylabel("D40 boxes")
    ax.set_ylim(0, ct.D40.max() * 1.25)
    style(ax, "D40 (pothole) boxes per domain — count and share of all D40")
    fig.tight_layout(); fig.savefig(FIG / "d40_count_by_domain.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bottom = np.zeros(len(SUBSETS))
    for c in TARGET:
        ax.bar(x, row_norm[c].values, bottom=bottom, width=0.6, color=CLS_COLOR[c], label=c, edgecolor="white", linewidth=1.5)
        for xi, (b, v) in enumerate(zip(bottom, row_norm[c].values)):
            if v >= 6:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        bottom += row_norm[c].values
    ax.set_xticks(x, lab); ax.set_ylabel("% of domain's target boxes"); ax.set_ylim(0, 100)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    style(ax, "Class mix within each domain (row-normalized)")
    fig.tight_layout(); fig.savefig(FIG / "class_mix_by_domain.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, data, ttl, fmt in ((axes[0], ct, "Counts (boxes)", "{:.0f}"), (axes[1], pd.DataFrame(std_resid, index=SUBSETS, columns=TARGET), "Pearson residuals (obs−exp)/√exp", "{:+.0f}")):
        v = data.values.astype(float)
        if ax is axes[0]:
            im = ax.imshow(v, cmap="Blues", aspect="auto")
        else:
            m = np.abs(v).max()
            im = ax.imshow(v, cmap="RdBu_r", vmin=-m, vmax=m, aspect="auto")
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                frac = (v[i, j] - v.min()) / (v.max() - v.min() + 1e-9)
                dark = frac > 0.6 if ax is axes[0] else abs(v[i, j]) > 0.6 * np.abs(v).max()
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=8, color="white" if dark else INK)
        ax.set_xticks(range(4), TARGET); ax.set_yticks(range(len(SUBSETS)), lab)
        ax.set_title(ttl, loc="left", fontsize=10, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Domain × class contingency  (χ²={stats['chi2']:.0f}, dof={stats['dof']}, Cramér's V={stats['cramers_v_corrected']:.3f})", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "contingency_heatmap.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.logspace(np.log10(4), np.log10(640 * 640), 50)
    for c in TARGET:
        axes[0].hist(tgt[tgt.label == c].area640, bins=bins, histtype="step", lw=2, color=CLS_COLOR[c], label=c, density=True)
    axes[0].set_xscale("log"); axes[0].axvline(32 ** 2, color=INK2, ls="--", lw=1); axes[0].axvline(96 ** 2, color=INK2, ls="--", lw=1)
    axes[0].set_xlabel("box area at 640-px input (px², log)"); axes[0].legend(frameon=False)
    style(axes[0], "Box area by class (dashed: 32², 96²)")
    szp = 100 * szd.div(szd.sum(1), axis=0)
    bottom = np.zeros(len(SUBSETS))
    for k, col in zip(["small", "medium", "large"], ["#2a78d6", "#1baf7a", "#eda100"]):
        axes[1].bar(x, szp[k].values, bottom=bottom, color=col, width=0.6, label=k, edgecolor="white", linewidth=1.5)
        bottom += szp[k].values
    axes[1].set_xticks(x, lab, rotation=30); axes[1].set_ylabel("% of domain's boxes"); axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    style(axes[1], "Size bucket mix per domain (640-px scale)")
    fig.tight_layout(); fig.savefig(FIG / "box_size_distribution.png"); plt.close(fig)

    with open(OUT / "audit_summary.json", "w") as f:
        json.dump(S, f, indent=2)
    pd.set_option("display.width", 200)
    print(json.dumps(S, indent=1))
    for name in ["domain_counts", "contingency_counts", "contingency_row_normalized_pct", "contingency_pearson_residuals", "d40_by_domain", "size640_by_class", "size640_by_domain", "size_native_by_class"]:
        print(f"\n== {name}\n", pd.read_csv(OUT / f"{name}.csv").to_string(index=False))


if __name__ == "__main__":
    main()
