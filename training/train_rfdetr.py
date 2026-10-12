"""RF-DETR baseline (Robinson et al., ICLR 2026) on split_v1, trained with its own pipeline and recommended
recipe, then scored by OUR evaluator (evaluation/evaluate.py reads the predictions this script writes).

    /home/divon/rfdetr_env/bin/python training/train_rfdetr.py configs/experiments/B05_rfdetr_small_s0.yaml

Same data and split as every other model: the pre-resized split_v1 images are exposed in the COCO layout RF-DETR
expects (symlinks; train/valid/test). Resolution 640. Differences vs the Ultralytics protocol (documented, not
removable without modifying the benchmark model): RF-DETR's own augmentation (multi-scale, no mosaic), AdamW
lr 1e-4 with layer-wise decay, EMA; effective batch 16 (its recommended value).
Writes experiments/<name>/ (RF-DETR output), results/<name>/preds_{val,test}.json and complexity.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/splits/rdd_v1"
COCO_DIR = ROOT / "data/splits/rdd_v1_coco"
CLASSES = ["D00", "D10", "D20", "D40"]


def build_coco_layout():
    """train/valid/test folders with symlinked images + _annotations.coco.json (ids 0..3)."""
    for split, dst in (("train", "train"), ("val", "valid"), ("test", "test")):
        d = COCO_DIR / dst
        if (d / "_annotations.coco.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        if split in ("val", "test"):
            coco = json.loads((BASE / f"coco_{split}.json").read_text())
        else:
            coco = {"images": [], "annotations": [], "categories": [{"id": i, "name": c, "supercategory": "damage"} for i, c in enumerate(CLASSES)]}
            aid = 0
            for k, img in enumerate(sorted((BASE / "images/train").glob("*.jpg"))):
                with Image.open(img) as im:
                    W, H = im.size
                coco["images"].append({"id": k, "file_name": img.name, "width": W, "height": H})
                for line in (BASE / "labels/train" / f"{img.stem}.txt").read_text().splitlines():
                    if not line.strip():
                        continue
                    c, x, y, w, h = map(float, line.split())
                    coco["annotations"].append({"id": aid, "image_id": k, "category_id": int(c), "bbox": [(x - w / 2) * W, (y - h / 2) * H, w * W, h * H],
                                                "area": w * W * h * H, "iscrowd": 0})
                    aid += 1
        for c in coco["categories"]:
            c.setdefault("supercategory", "damage")
        for im in coco["images"]:
            link = d / im["file_name"]
            if not link.exists():
                os.symlink(BASE / "images" / split / im["file_name"], link)
        (d / "_annotations.coco.json").write_text(json.dumps(coco))


def predict(model, split, out):
    gt = json.loads((BASE / f"coco_{split}.json").read_text())
    dets = []
    for k in range(0, len(gt["images"]), 16):
        ims = gt["images"][k:k + 16]
        batch = [Image.open(BASE / "images" / split / im["file_name"]).convert("RGB") for im in ims]
        res = model.predict(batch, threshold=0.001)
        res = res if isinstance(res, list) else [res]
        for im, r in zip(ims, res):
            order = r.confidence.argsort()[::-1][:300]
            for (x1, y1, x2, y2), s, c in zip(r.xyxy[order], r.confidence[order], r.class_id[order]):
                dets.append({"image_id": im["id"], "category_id": int(c), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    out.write_text(json.dumps(dets))


def main(cfg_path):
    exp = yaml.safe_load(Path(cfg_path).read_text())
    name, o = exp["name"], exp.get("overrides", {})
    run_dir = ROOT / "experiments" / name
    res_dir = ROOT / "results" / name
    run_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(exp, sort_keys=False))
    build_coco_layout()
    import rfdetr
    torch.manual_seed(exp.get("seed", 0))
    Model = getattr(rfdetr, exp["model"].split(":", 1)[1])
    ckpt = run_dir / "checkpoint_best_total.pth"
    if not ckpt.exists():
        model = Model(resolution=o.get("resolution", 640))
        model.train(dataset_dir=str(COCO_DIR), output_dir=str(run_dir), epochs=o.get("epochs", 20), batch_size=o.get("batch_size", 4),
                    grad_accum_steps=o.get("grad_accum_steps", 4), lr=o.get("lr", 1e-4), num_workers=o.get("num_workers", 6),
                    seed=exp.get("seed", 0), checkpoint_interval=o.get("epochs", 20))
    best = next((p for p in (run_dir / "checkpoint_best_total.pth", run_dir / "checkpoint_best_ema.pth", run_dir / "checkpoint_best_regular.pth") if p.exists()), None)
    model = Model(resolution=o.get("resolution", 640), pretrain_weights=str(best))
    for split in ("val", "test"):
        if not (res_dir / f"preds_{split}.json").exists():
            predict(model, split, res_dir / f"preds_{split}.json")
    net = model.model.model.cuda().eval()
    params = sum(p.numel() for p in net.parameters())
    x = torch.zeros(1, 3, o.get("resolution", 640), o.get("resolution", 640), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(30): net(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(200): net(x)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 200
    (res_dir / "complexity.json").write_text(json.dumps({"params_M": params / 1e6, "GFLOPs_640": None, "latency_ms_bs1_fp16_forward_only": dt * 1e3,
                                                         "fps_bs1_fp16_forward_only": 1 / dt, "checkpoint_MB": best.stat().st_size / 2 ** 20,
                                                         "gpu": torch.cuda.get_device_name(0), "note": "autocast fp16; GFLOPs not computed"}))
    print("checkpoint used:", best)


if __name__ == "__main__":
    main(sys.argv[1])
