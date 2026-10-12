"""Pack the TRAIN-split images/labels needed by colab_generate_potholes.py into one zip for Colab/Kaggle.

    python augmentation/pack_for_generation.py [--targets Czech Norway United_States China_MotorBike] [--out gen_pkg.zip]

Contents (only split_v1 TRAIN images; val/test are never packed, so generated images cannot leak test scenes):
  * target images  : train images of the D40-poor domains we want to add potholes to
  * donor images   : every train image (any domain) that contains at least one real D40 box
  * labels/<stem>.txt for all of the above, and split_v1.csv
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/splits/rdd_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["Czech", "Norway", "United_States", "China_MotorBike"])
    ap.add_argument("--out", default=str(ROOT / "colab/gen_pkg.zip"))
    a = ap.parse_args()
    split = pd.read_csv(ROOT / "data/splits/split_v1.csv")
    train = split[split.split == "train"]
    keep = set(train[train.subset.isin(a.targets)].image_id) | set(train[train.has_D40].image_id)
    n = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:  # jpgs are already compressed
        z.write(ROOT / "data/splits/split_v1.csv", "gen_pkg/split_v1.csv")
        for stem in sorted(keep):
            img, lab = BASE / "images/train" / f"{stem}.jpg", BASE / "labels/train" / f"{stem}.txt"
            if not img.exists():
                continue
            z.write(img, f"gen_pkg/images/{stem}.jpg")
            if lab.exists():
                z.write(lab, f"gen_pkg/labels/{stem}.txt")
            n += 1
    print(f"packed {n} train images ({len(set(train[train.subset.isin(a.targets)].image_id))} target-domain, "
          f"{int(train.has_D40.sum())} D40 donors) -> {a.out}")


if __name__ == "__main__":
    main()
