"""Export split_v1 to YOLO format, images pre-resized so the long side is 640 px (cv2.INTER_AREA).

Every detector in this project is trained/evaluated at imgsz=640, which already resizes the long side to 640,
so pre-resizing only removes repeated decoding of the ~4000-px Norway JPEGs; it does not change model inputs
beyond interpolation details. Labels are normalized, so they are resolution-independent.

Writes data/splits/rdd_v1/{images,labels}/{train,val,test}/  and  data/splits/rdd_v1/coco_{val,test}.json
(COCO ground truth at the 640-scale, used by evaluation/coco_eval.py), plus dataset yamls in configs/data/.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
DST = ROOT / "data/splits/rdd_v1"
CLASSES = ["D00", "D10", "D20", "D40"]
CID = {c: i for i, c in enumerate(CLASSES)}


def resize_one(args):
    src, dst = args
    if os.path.exists(dst):
        return
    im = cv2.imread(src)
    h, w = im.shape[:2]
    r = 640 / max(h, w)
    if r < 1:
        im = cv2.resize(im, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    img = pd.read_csv(P / "images_all.csv")
    ann = pd.read_csv(P / "annotations_all.csv")
    ann = ann[ann.label.isin(CLASSES) & (ann.x2 > ann.x1) & (ann.y2 > ann.y1)]
    spl = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    img = img.merge(spl[["image_id", "split"]], on="image_id")
    jobs = []
    for s in ("train", "val", "test"):
        (DST / "images" / s).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / s).mkdir(parents=True, exist_ok=True)
    by_img = {k: g for k, g in ann.groupby("image_id")}
    coco = {s: {"images": [], "annotations": [], "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)]} for s in ("val", "test")}
    aid = 0
    for idx, r in enumerate(img.itertuples()):
        jobs.append((r.path, str(DST / "images" / r.split / f"{r.image_id}.jpg")))
        g = by_img.get(r.image_id)
        lines = []
        sc = min(1.0, 640 / max(r.width, r.height))
        W, H = round(r.width * sc), round(r.height * sc)
        if r.split in coco:
            coco[r.split]["images"].append({"id": idx, "file_name": f"{r.image_id}.jpg", "width": W, "height": H, "subset": r.subset})
        if g is not None:
            for a in g.itertuples():
                x1, y1 = max(0, a.x1), max(0, a.y1)
                x2, y2 = min(r.width, a.x2), min(r.height, a.y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"{CID[a.label]} {(x1 + x2) / 2 / r.width:.6f} {(y1 + y2) / 2 / r.height:.6f} {(x2 - x1) / r.width:.6f} {(y2 - y1) / r.height:.6f}")
                if r.split in coco:
                    bw, bh = (x2 - x1) * sc, (y2 - y1) * sc
                    coco[r.split]["annotations"].append({"id": aid, "image_id": idx, "category_id": CID[a.label],
                                                         "bbox": [x1 * sc, y1 * sc, bw, bh], "area": bw * bh, "iscrowd": 0})
                    aid += 1
        (DST / "labels" / r.split / f"{r.image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        list(ex.map(resize_one, jobs, chunksize=64))
    for s, c in coco.items():
        (DST / f"coco_{s}.json").write_text(json.dumps(c))
    cfg = ROOT / "configs/data"
    cfg.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (cfg / "rdd_v1.yaml").write_text(f"path: {DST}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    print("done", img.split.value_counts().to_dict())


if __name__ == "__main__":
    main()
