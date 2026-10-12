"""Paired image-level bootstrap between two evaluated models on the same test split.

    python evaluation/compare.py <model_A> <model_B> [--B 1000] [--domain India | --domain Norway,Czech]

Uses results/<name>/matches_test.npz written by evaluate.py. Each bootstrap resample draws the same test images
for both models, so the CI is on the *difference* B - A (per-class AP@50 and mAP@50, area 'all', IoU 0.5).
Prints and returns a dict; with --save writes results/comparisons/<A>__vs__<B>[__domain].json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.evaluate import CLASSES, ap_from  # noqa: E402


def load(name):
    z = np.load(ROOT / "results" / name / "matches_test.npz")
    return {c: {k: z[f"{c}_{k}"] for k in ("img", "score", "tp", "ngt")} for c in CLASSES}


def ap_all(m, cnt):
    return np.array([ap_from(m[c]["score"], m[c]["tp"], (m[c]["ngt"] * cnt).sum(), cnt[m[c]["img"]].astype(float)) for c in CLASSES])


def compare(a, b, B=1000, domain=None, seed=0):
    ma, mb = load(a), load(b)
    n = len(ma[CLASSES[0]]["ngt"])
    mask = np.ones(n, bool)
    if domain:
        gt = json.loads((ROOT / "data/splits/rdd_v1/coco_test.json").read_text())
        doms = set(domain.split(","))  # "Norway,Czech,United_States" pools several domains into one test
        mask = np.array([im["subset"] in doms for im in gt["images"]])
    idx = np.where(mask)[0]
    base = np.zeros(n); base[idx] = 1
    pa, pb = ap_all(ma, base), ap_all(mb, base)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        cnt = np.bincount(idx[rng.integers(0, len(idx), len(idx))], minlength=n)
        da, db = ap_all(ma, cnt), ap_all(mb, cnt)
        diffs.append(np.append(db - da, np.nanmean(db) - np.nanmean(da)))
    diffs = np.array(diffs)
    out = {"A": a, "B": b, "domain": domain or "all", "n_images": int(len(idx)), "B_resamples": B}
    for k, c in enumerate(CLASSES + ["mAP50"]):
        pa_k = float(np.nanmean(pa)) if c == "mAP50" else float(pa[k])
        pb_k = float(np.nanmean(pb)) if c == "mAP50" else float(pb[k])
        d = diffs[:, k]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        out[c] = {"A": pa_k, "B": pb_k, "diff": pb_k - pa_k, "ci95": [float(lo), float(hi)],
                  "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("a"); p.add_argument("b"); p.add_argument("--B", type=int, default=1000)
    p.add_argument("--domain"); p.add_argument("--save", action="store_true")
    x = p.parse_args()
    r = compare(x.a, x.b, x.B, x.domain)
    print(json.dumps(r, indent=1))
    if x.save:
        d = ROOT / "results/comparisons"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{x.a}__vs__{x.b}{'__' + x.domain.replace(',', '+') if x.domain else ''}.json").write_text(json.dumps(r, indent=2))
