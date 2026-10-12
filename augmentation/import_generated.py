"""Turn the output of colab_generate_potholes.py into a training set + FAIR experiment configs.

    python augmentation/import_generated.py <gen_out_dir> [--tag G2] [--models B04_yolo26s P5_A4full]

<gen_out_dir> = the --out folder copied back from Drive (images/, labels/, manifest.csv, summary.json).
Writes
  data/splits/train_<tag>.txt            real train images + accepted synthetic images
  configs/data/rdd_v1_<tag>.yaml         same val/test as every other run (test is never touched)
  configs/experiments/<tag>_<model>_s0.yaml for each model: identical recipe to its real-data run, with epochs scaled
                                         so the number of optimisation steps stays equal (more images per epoch).
Every model gets the same synthetic set: the generated data is evaluated as a data method, never given to one
model only. Before training, audit a random sample by eye (audit/sheet_*.jpg) and record the pothole-present rate.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/splits/rdd_v1"
# model key -> real-data config it copies (recipe, init, batch) ; epochs are rescaled
SOURCES = {"B04_yolo26s": "B04_yolo26s_s0", "B01_yolov8s": "B01_yolov8s_s0", "P5_A4full": "P5_A4full_comp_s0",
           "A4": "A4_s0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir")
    ap.add_argument("--tag", default="G2")
    ap.add_argument("--models", nargs="+", default=["B04_yolo26s", "P5_A4full"])
    a = ap.parse_args()
    gen = Path(a.gen_dir)
    man = pd.read_csv(gen / "manifest.csv")
    acc = man[man.accepted.astype(bool)]
    split = pd.read_csv(ROOT / "data/splits/split_v1.csv").set_index("image_id")
    assert (split.loc[acc.source].split == "train").all(), "synthetic image built from a non-train source"

    dst = ROOT / "data/splits" / f"gen_{a.tag}"
    for s in ("images/train", "labels/train"):
        (dst / s).mkdir(parents=True, exist_ok=True)
    for name in acc.name:
        shutil.copy2(gen / "images" / f"{name}.jpg", dst / "images/train" / f"{name}.jpg")
        shutil.copy2(gen / "labels" / f"{name}.txt", dst / "labels/train" / f"{name}.txt")

    real = sorted(str(p) for p in (BASE / "images/train").glob("*.jpg"))
    syn = sorted(str(p) for p in (dst / "images/train").glob("*.jpg"))
    lst = ROOT / "data/splits" / f"train_{a.tag}.txt"
    lst.write_text("\n".join(real + syn) + "\n")
    data_yaml = ROOT / "configs/data" / f"rdd_v1_{a.tag}.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(BASE), "train": str(lst), "val": "images/val", "test": "images/test",
                                         "names": {0: "D00", 1: "D10", 2: "D20", 3: "D40"}}, sort_keys=False))

    for m in a.models:
        src = yaml.safe_load((ROOT / "configs/experiments" / f"{SOURCES[m]}.yaml").read_text())
        base_ep = src.get("overrides", {}).get("epochs", 20)
        base_n = 33595 if "comp" in SOURCES[m] else len(real)  # images per epoch of the source run
        ep = max(1, round(base_ep * base_n / len(real + syn)))
        cfg = {**src, "name": f"{a.tag}_{m}_s0", "data": f"configs/data/rdd_v1_{a.tag}.yaml",
               "description": f"{SOURCES[m]} recipe + {len(syn)} generated pothole images ({a.tag}); real data only "
                              f"otherwise (no RFS); epochs scaled {base_ep}->{ep} to keep optimisation steps equal"}
        cfg["overrides"] = {**src.get("overrides", {}), "epochs": ep}
        cfg["overrides"]["close_mosaic"] = max(1, round(5 * ep / 20))
        (ROOT / "configs/experiments" / f"{cfg['name']}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print("wrote", cfg["name"], "epochs", ep)
    summ = json.loads((gen / "summary.json").read_text()) if (gen / "summary.json").exists() else {}
    print(f"{len(syn)} synthetic + {len(real)} real train images -> {lst}\n", json.dumps(summ.get("by_domain", {})))


if __name__ == "__main__":
    main()
