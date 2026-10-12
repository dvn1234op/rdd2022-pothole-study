"""Canonical evaluation for every model in this project (identical code path for all models).

    python evaluation/evaluate.py <experiment_name> [--splits val test] [--data rdd_v1]

1. Loads experiments/<name>/weights/best.pt (chosen on VAL during training; the test set is never used for selection).
2. Predicts on the pre-resized images with conf=0.001, NMS IoU=0.7, max_det=300, imgsz=640, FP16.
3. COCO evaluation (pycocotools) against data/splits/rdd_v1/coco_<split>.json (GT at the 640-px scale):
   overall mAP50 / mAP50:95, per-class AP50 / AP50:95, size-stratified AP50 (small<32², medium<96², large),
   per-domain AP50 (overall + per class).
4. Precision / recall / F1 at IoU 0.5 with a per-class confidence threshold chosen to maximise F1 on VAL and then
   applied unchanged to TEST.
5. Image-level bootstrap (1000 resamples, seed 0) 95% CIs for test mAP50 and per-class AP50; per-image match data
   is saved so that paired bootstraps between models can be run (evaluation/compare.py).
6. Model complexity: params, GFLOPs@640, FP16 batch-1 latency/FPS, peak inference memory, checkpoint size.
Outputs: results/<name>/{metrics.json, preds_<split>.json, matches_<split>.npz}
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLASSES = ["D00", "D10", "D20", "D40"]
SIZES = {"small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}


def load_model(weights):
    from ultralytics import RTDETR, YOLO
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    is_detr = "RTDETR" in type(ck.get("model", ck.get("ema"))).__name__
    return (RTDETR if is_detr else YOLO)(str(weights))


def predict(model, gt, img_dir):
    dets = []
    files = [img_dir / im["file_name"] for im in gt["images"]]
    ids = {im["file_name"]: im["id"] for im in gt["images"]}
    for k in range(0, len(files), 256):
        for r in model.predict([str(f) for f in files[k:k + 256]], conf=0.001, iou=0.7, max_det=300, imgsz=640,
                               half=True, verbose=False, batch=16, stream=True, device=0):
            iid = ids[Path(r.path).name]
            b = r.boxes
            for (x1, y1, x2, y2), s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.int().tolist()):
                dets.append({"image_id": iid, "category_id": c, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    return dets


def coco_eval(gt_path, dets, img_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(str(gt_path))
        D = G.loadRes(dets) if dets else None
        E = COCOeval(G, D, "bbox")
        if img_ids is not None:
            E.params.imgIds = img_ids
        E.params.areaRng = [[0, 1e10]] + [list(v) for v in SIZES.values()]
        E.params.areaRngLbl = ["all"] + list(SIZES)
        E.evaluate(); E.accumulate()
    p = E.eval["precision"]  # [T, R, K, A, M]  T: iou .50:.05:.95, M: maxDets [1, 10, 100]
    keep = None if img_ids is None else set(img_ids)
    n_gt = {k: sum(1 for a in G.dataset["annotations"] if a["category_id"] == k and (keep is None or a["image_id"] in keep)) for k in range(4)}

    def ap(t=None, k=None, a=0):
        x = p[:, :, :, a, -1] if t is None else p[t, :, :, a, -1][None]
        x = x[:, :, [k]] if k is not None else x
        valid = x > -1
        if not valid.any():
            return None
        # mean over classes that have GT (pycocotools marks classes w/o GT as -1)
        per_k = [x[..., j][x[..., j] > -1].mean() for j in range(x.shape[-1]) if (x[..., j] > -1).any()]
        return float(np.mean(per_k))

    out = {"mAP50": ap(0), "mAP50_95": ap(), "n_gt": n_gt, "per_class": {}, "size_AP50": {}, "size_AP50_95": {}}
    for k, c in enumerate(CLASSES):
        out["per_class"][c] = {"AP50": ap(0, k), "AP50_95": ap(None, k),
                               **{f"AP50_{s}": ap(0, k, i + 1) for i, s in enumerate(SIZES)}}
    for i, s in enumerate(SIZES):
        out["size_AP50"][s] = ap(0, None, i + 1)
        out["size_AP50_95"][s] = ap(None, None, i + 1)
    return out


# ---------------- own IoU-0.5 matcher (COCO semantics, area 'all', maxDets 100) for P/R/F1 and bootstrap -------------
def iou_matrix(a, b):
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter)


def match(gt, dets, thr=0.5, max_det=100):
    """Returns per class: arrays (image_idx, score, tp) for dets and n_gt per image."""
    img_index = {im["id"]: i for i, im in enumerate(gt["images"])}
    n_img = len(img_index)
    G = {}
    for a in gt["annotations"]:
        G.setdefault((a["image_id"], a["category_id"]), []).append(a["bbox"])
    D = {}
    for d in dets:
        D.setdefault((d["image_id"], d["category_id"]), []).append((d["score"], d["bbox"]))
    res = {}
    for k in range(4):
        ngt = np.zeros(n_img, int)
        rows = []
        for (iid, c), g in G.items():
            if c == k:
                ngt[img_index[iid]] = len(g)
        for (iid, c), dl in D.items():
            if c != k:
                continue
            dl = sorted(dl, key=lambda x: -x[0])[:max_det]
            s = np.array([x[0] for x in dl]); bb = np.array([x[1] for x in dl])
            tp = np.zeros(len(dl), bool)
            g = G.get((iid, k))
            if g:
                ious = iou_matrix(bb, np.array(g))
                used = np.zeros(len(g), bool)
                for di in range(len(dl)):
                    cand = np.where(~used & (ious[di] >= thr))[0]
                    if len(cand):
                        j = cand[np.argmax(ious[di, cand])]
                        used[j] = True; tp[di] = True
            rows.append(np.stack([np.full(len(dl), img_index[iid]), s, tp], 1))
        r = np.concatenate(rows) if rows else np.zeros((0, 3))
        res[CLASSES[k]] = {"img": r[:, 0].astype(int), "score": r[:, 1], "tp": r[:, 2].astype(bool), "ngt": ngt}
    return res


def ap_from(score, tp, n_gt, w=None):
    """COCO 101-point interpolated AP. w = per-detection multiplicity (bootstrap)."""
    if n_gt <= 0:
        return np.nan
    o = np.argsort(-score, kind="mergesort")
    w = np.ones(len(score)) if w is None else w
    tpc = np.cumsum(tp[o] * w[o]); fpc = np.cumsum((~tp[o]) * w[o])
    rec = tpc / n_gt; prec = tpc / np.maximum(tpc + fpc, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    q = np.zeros(101)
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    ok = idx < len(prec)
    q[ok] = prec[idx[ok]]
    return float(q.mean())


def prf_at(m, thr):
    keep = m["score"] >= thr
    tp = int(m["tp"][keep].sum()); fp = int(keep.sum() - tp); n = int(m["ngt"].sum())
    p = tp / max(tp + fp, 1); r = tp / max(n, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def best_f1_threshold(m):
    o = np.argsort(-m["score"]); s = m["score"][o]; tp = m["tp"][o]
    tpc = np.cumsum(tp); fpc = np.cumsum(~tp); n = m["ngt"].sum()
    f1 = 2 * tpc / np.maximum(tpc + fpc + n, 1)
    i = int(np.argmax(f1)) if len(f1) else 0
    return float(s[i]) if len(s) else 0.5


def bootstrap(mt, n_img, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {c: [] for c in CLASSES}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
        for c in CLASSES:
            m = mt[c]
            out[c].append(ap_from(m["score"], m["tp"], (m["ngt"] * cnt).sum(), cnt[m["img"]].astype(float)))
    arr = np.array([out[c] for c in CLASSES])
    ci = {c: [float(np.percentile(arr[k], 2.5)), float(np.percentile(arr[k], 97.5))] for k, c in enumerate(CLASSES)}
    mean = arr.mean(0)
    ci["mAP50"] = [float(np.percentile(mean, 2.5)), float(np.percentile(mean, 97.5))]
    return ci


def profile(model, weights):
    net = model.model.cuda().eval().half()
    params = sum(p.numel() for p in net.parameters())
    gflops = None
    with contextlib.suppress(Exception):
        from ultralytics.utils.torch_utils import get_flops
        gflops = float(get_flops(net.float(), 640)); net.half()
    x = torch.zeros(1, 3, 640, 640, device="cuda", dtype=torch.half)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(30):
            net(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(200):
            net(x)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 200
    return {"params_M": params / 1e6, "GFLOPs_640": gflops, "latency_ms_bs1_fp16_forward_only": dt * 1e3,
            "fps_bs1_fp16_forward_only": 1 / dt, "peak_infer_mem_MB_bs1": torch.cuda.max_memory_allocated() / 2 ** 20,
            "checkpoint_MB": Path(weights).stat().st_size / 2 ** 20, "gpu": torch.cuda.get_device_name(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--data", default="rdd_v1"); ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--weights", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    weights = Path(a.weights) if a.weights else ROOT / "experiments" / a.name / "weights/best.pt"
    out_dir = ROOT / "results" / (a.name + a.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data/splits" / a.data
    # models trained outside Ultralytics (RF-DETR) provide preds_<split>.json themselves; then no model is loaded
    model = load_model(weights) if weights.exists() else None
    M = {"experiment": a.name, "weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
         "eval_settings": {"conf": 0.001, "nms_iou": 0.7, "max_det_predict": 300, "coco_maxDets": 100, "imgsz": 640, "half": True}}
    matches = {}
    for split in a.splits:
        gt_path = data / f"coco_{split}.json"
        gt = json.loads(gt_path.read_text())
        pred_file = out_dir / f"preds_{split}.json"
        if pred_file.exists():
            dets = json.loads(pred_file.read_text())
        else:
            dets = predict(model, gt, data / "images" / split)
            pred_file.write_text(json.dumps(dets))
        R = coco_eval(gt_path, dets)
        domains = sorted({im["subset"] for im in gt["images"]})
        R["per_domain"] = {}
        for dname in domains:
            ids = [im["id"] for im in gt["images"] if im["subset"] == dname]
            r = coco_eval(gt_path, dets, ids)
            R["per_domain"][dname] = {"mAP50": r["mAP50"], "mAP50_95": r["mAP50_95"], "n_images": len(ids), "n_gt": r["n_gt"],
                                      **{f"{c}_AP50": r["per_class"][c]["AP50"] for c in CLASSES}}
        mt = match(gt, dets)
        matches[split] = mt
        own = {c: ap_from(mt[c]["score"], mt[c]["tp"], mt[c]["ngt"].sum()) for c in CLASSES}
        R["sanity_own_AP50_vs_coco"] = {c: [own[c], R["per_class"][c]["AP50"]] for c in CLASSES}
        np.savez_compressed(out_dir / f"matches_{split}.npz", **{f"{c}_{k}": v for c in CLASSES for k, v in mt[c].items()})
        M[split] = R
    if "val" in matches and "test" in matches:
        thr = {c: best_f1_threshold(matches["val"][c]) for c in CLASSES}
        M["test"]["PRF1_at_val_thresholds"] = {"thresholds_from_val": thr}
        for c in CLASSES:
            p, r, f = prf_at(matches["test"][c], thr[c])
            M["test"]["PRF1_at_val_thresholds"][c] = {"precision": p, "recall": r, "F1": f}
        for key in ("precision", "recall", "F1"):
            M["test"]["PRF1_at_val_thresholds"][f"macro_{key}"] = float(np.mean([M["test"]["PRF1_at_val_thresholds"][c][key] for c in CLASSES]))
        n_img = len(json.loads((data / "coco_test.json").read_text())["images"])
        M["test"]["bootstrap95_AP50"] = bootstrap(matches["test"], n_img)
    cx = out_dir / "complexity.json"  # written by external pipelines (RF-DETR)
    if model is not None:
        with contextlib.suppress(Exception):
            M["complexity"] = profile(model, weights)
    elif cx.exists():
        M["complexity"] = json.loads(cx.read_text())
    (out_dir / "metrics.json").write_text(json.dumps(M, indent=2))
    t = M.get("test", M.get(a.splits[-1]))
    print(json.dumps({"mAP50": t["mAP50"], "mAP50_95": t["mAP50_95"], **{c: t["per_class"][c]["AP50"] for c in CLASSES},
                      "size_AP50": t["size_AP50"], "sanity": t["sanity_own_AP50_vs_coco"]}, indent=1))


if __name__ == "__main__":
    main()
