"""Render a random sample of inserted synthetic D40 boxes (synthetic crop | source crop) for manual inspection.
Sheets: visualizations/synthetic_audit/manual_sheet_{k}.jpg ; index in results/synthetic_audit/manual_sample.csv
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/RDD2022/RDD2022_all_countries"
N, PER, T = 60, 20, 200

df = pd.read_csv(ROOT / "results/synthetic_audit/synthetic_boxes.csv")
s = df[df.status == "ok"].sample(N, random_state=7).reset_index(drop=True)
s.index.name = "sample_idx"
s.to_csv(ROOT / "results/synthetic_audit/manual_sample.csv")
for k in range(N // PER):
    tiles = []
    for i in range(k * PER, (k + 1) * PER):
        r = s.loc[i]
        syn = cv2.imread(str(ROOT / "data/synthetic_yolo_format/images" / f"{r.stem}.jpg"))
        sub = r.source.rsplit("_", 1)[0]
        src = cv2.imread(str(RAW / sub / sub / "train/images" / f"{r.source}.jpg"))
        src = cv2.resize(src, (syn.shape[1], syn.shape[0]))
        h, w = syn.shape[:2]
        bw, bh = max(r.bw * w, 24), max(r.bh * h, 24)
        x1, y1 = int(max(0, r.xc * w - bw)), int(max(0, r.yc * h - bh))
        x2, y2 = int(min(w, r.xc * w + bw)), int(min(h, r.yc * h + bh))
        pair = []
        for im in (syn, src):
            im = im.copy()
            cv2.rectangle(im, (int((r.xc - r.bw / 2) * w), int((r.yc - r.bh / 2) * h)), (int((r.xc + r.bw / 2) * w), int((r.yc + r.bh / 2) * h)), (0, 0, 255), 1)
            pair.append(cv2.resize(im[y1:y2, x1:x2], (T, T)))
        ctx = cv2.resize(syn, (T, T))
        cv2.rectangle(ctx, (int((r.xc - r.bw / 2) * T), int((r.yc - r.bh / 2) * T)), (int((r.xc + r.bw / 2) * T), int((r.yc + r.bh / 2) * T)), (0, 0, 255), 2)
        tile = np.hstack([ctx] + pair)
        cv2.putText(tile, str(i), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(tile)
    rows = [np.hstack(tiles[j:j + 2]) for j in range(0, PER, 2)]
    cv2.imwrite(str(ROOT / f"visualizations/synthetic_audit/manual_sheet_{k}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
