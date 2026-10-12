"""Parse the raw RDD2022 Pascal-VOC annotations into one flat table.

Output: data/processed/annotations_all.csv  (one row per box, ALL raw label names kept)
        data/processed/images_all.csv       (one row per image, incl. images with no boxes)

Only the official *train* folders carry annotations; the official test folders are unlabeled
(CRDDC2022 kept test labels private), so they cannot be used for evaluation.
"""
import csv
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/RDD2022/RDD2022_all_countries"
OUT = ROOT / "data/processed"

SUBSETS = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]


def parse_one(xml_path):
    tree = ET.parse(xml_path)
    r = tree.getroot()
    size = r.find("size")
    w = int(float(size.find("width").text)) if size is not None else -1
    h = int(float(size.find("height").text)) if size is not None else -1
    boxes = []
    for obj in r.findall("object"):
        name = obj.find("name").text.strip()
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((name, x1, y1, x2, y2))
    return w, h, boxes


def job(args):
    subset, xml_path = args
    stem = Path(xml_path).stem
    img_path = RAW / subset / subset / "train/images" / f"{stem}.jpg"
    w, h, boxes = parse_one(xml_path)
    if w <= 0 or h <= 0:  # fall back to the image header
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
    return subset, stem, str(img_path), w, h, boxes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = []
    for s in SUBSETS:
        xml_dir = RAW / s / s / "train/annotations/xmls"
        tasks += [(s, str(p)) for p in sorted(xml_dir.glob("*.xml"))]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        results = list(ex.map(job, tasks, chunksize=256))
    with open(OUT / "images_all.csv", "w", newline="") as fi, open(OUT / "annotations_all.csv", "w", newline="") as fa:
        wi, wa = csv.writer(fi), csv.writer(fa)
        wi.writerow(["image_id", "subset", "path", "width", "height", "n_boxes_raw"])
        wa.writerow(["image_id", "subset", "label", "x1", "y1", "x2", "y2", "img_w", "img_h"])
        for subset, stem, path, w, h, boxes in results:
            wi.writerow([stem, subset, path, w, h, len(boxes)])
            for name, x1, y1, x2, y2 in boxes:
                wa.writerow([stem, subset, name, x1, y1, x2, y2, w, h])
    print(f"images: {len(results)}  boxes: {sum(len(r[5]) for r in results)}")


if __name__ == "__main__":
    main()
