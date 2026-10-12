"""Resolution hypothesis test (inference only): does running a trained detector on native-resolution TILES of the
high-resolution domain (Norway, ~4000x2040 px) recover small damage that is destroyed by the 640-px resize?

    python evaluation/tiled_inference.py <experiment_name> [--domain Norway] [--tile 1280] [--overlap 0.25]

For each test image of the domain: predictions on (a) the whole image at 640 (as in evaluate.py) and (b) native
tiles of `tile` px each resized to 640, are merged (class-wise NMS 0.6, top 300) and mapped back to the 640-scale
frame of the standard ground truth, so metrics and size buckets are identical to evaluate.py.
Output: results/<name>/tiled_<domain>.json with untiled vs tiled metrics on the same images.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.evaluate import coco_eval, load_model  # noqa: E402


def tiles(W, H, t, ov):
    step = int(t * (1 - ov))
    xs = list(range(0, max(W - t, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - t, 0) + 1, step)) or [0]
    if xs[-1] + t < W: xs.append(W - t)
    if ys[-1] + t < H: ys.append(H - t)
    return [(x, y, min(x + t, W), min(y + t, H)) for y in ys for x in xs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name"); p.add_argument("--domain", default="Norway")
    p.add_argument("--tile", type=int, default=1280); p.add_argument("--overlap", type=float, default=0.25)
    a = p.parse_args()
    gt_path = ROOT / "data/splits/rdd_v1/coco_test.json"
    gt = json.loads(gt_path.read_text())
    paths = pd.read_csv(ROOT / "data/processed/images_all.csv").set_index("image_id").path
    ims = [im for im in gt["images"] if im["subset"] == a.domain]
    ids = [im["id"] for im in ims]
    model = load_model(ROOT / "experiments" / a.name / "weights/best.pt")
    idset = set(ids)
    base = [d for d in json.loads((ROOT / "results" / a.name / "preds_test.json").read_text()) if d["image_id"] in idset]
    by_img = {}
    for d in base:
        by_img.setdefault(d["image_id"], []).append(d)
    tiled = []
    for im in ims:
        img = cv2.imread(paths[im["file_name"][:-4]])
        H, W = img.shape[:2]
        s = 640 / max(H, W)
        boxes, scores, cls = [], [], []
        crops = [(x1, y1, img[y1:y2, x1:x2]) for x1, y1, x2, y2 in tiles(W, H, a.tile, a.overlap)]
        res = model.predict([c for _, _, c in crops], conf=0.001, iou=0.7, max_det=300, imgsz=640, half=True, verbose=False, device=0)
        for (x0, y0, _), r in zip(crops, res):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy() + np.array([x0, y0, x0, y0])
            boxes.append(xyxy); scores.append(b.conf.cpu().numpy()); cls.append(b.cls.cpu().numpy())
        own = by_img.get(im["id"], [])
        if own:
            bb = np.array([d["bbox"] for d in own]) / s
            boxes.append(np.c_[bb[:, :2], bb[:, :2] + bb[:, 2:]]); scores.append(np.array([d["score"] for d in own])); cls.append(np.array([d["category_id"] for d in own]))
        B = torch.tensor(np.concatenate(boxes), dtype=torch.float32); S = torch.tensor(np.concatenate(scores), dtype=torch.float32); K = torch.tensor(np.concatenate(cls))
        keep = torchvision.ops.batched_nms(B, S, K, 0.6)[:300]
        for i in keep.tolist():
            x1, y1, x2, y2 = (B[i] * s).tolist()
            tiled.append({"image_id": im["id"], "category_id": int(K[i]), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(S[i])})
    out = {"experiment": a.name, "domain": a.domain, "n_images": len(ims), "tile": a.tile, "overlap": a.overlap,
           "untiled": coco_eval(gt_path, base, ids), "tiled_plus_full": coco_eval(gt_path, tiled, ids)}
    (ROOT / "results" / a.name / f"tiled_{a.domain}.json").write_text(json.dumps(out, indent=2))
    for k in ("untiled", "tiled_plus_full"):
        r = out[k]
        print(k, f"mAP50 {100*r['mAP50']:.1f} mAP50:95 {100*r['mAP50_95']:.1f}",
              {c: round(100 * r['per_class'][c]['AP50'], 1) for c in r['per_class']},
              {s: round(100 * (v or 0), 1) for s, v in r['size_AP50'].items()})


if __name__ == "__main__":
    main()
