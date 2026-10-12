"""Build colab/btp_colab.zip (pre-resized split_v1 images+labels, configs, pretrained weights, colab_train.py)
and colab/BTP_colab_baselines.ipynb. Upload the zip to Google Drive under MyDrive/BTP/ and open the notebook in Colab."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab"
RUNS = ["B02_yolo11s_s0", "B04_yolo26s_s0"]


def build_zip():
    OUT.mkdir(exist_ok=True)
    z = OUT / "btp_colab.zip"
    files = [p for p in (ROOT / "data/splits/rdd_v1").rglob("*") if p.is_file() and p.suffix in (".jpg", ".txt")]
    files += [ROOT / "configs/protocol.yaml", ROOT / "training/colab_train.py"]
    files += [ROOT / f"configs/experiments/{r}.yaml" for r in RUNS]
    files += [ROOT / "checkpoints/pretrained/yolo11s.pt", ROOT / "checkpoints/pretrained/yolo26s.pt"]
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED, allowZip64=True) as f:
        for p in files:
            f.write(p, p.relative_to(ROOT).as_posix())
    print(z, f"{z.stat().st_size / 2**30:.2f} GB, {len(files)} files")


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def build_notebook():
    cells = [
        cell("# BTP pilot — Colab baselines (YOLO11s, YOLO26s)\n"
             "Same split, same protocol as the laptop runs. **Runtime → Change runtime type → T4 GPU.**\n\n"
             "1. Upload `btp_colab.zip` to `MyDrive/BTP/`.\n2. Set `RUN` below, then *Runtime → Run all*.\n"
             "3. If Colab disconnects, reconnect and *Run all* again — training resumes from the last epoch.\n"
             "4. When finished, download `MyDrive/BTP/runs/<RUN>/` (at least `weights/best.pt`, `results.csv`) and give it to the laptop for evaluation.", "markdown"),
        cell("RUN = 'B02_yolo11s_s0'   # second session: 'B04_yolo26s_s0'"),
        cell("!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
             "from google.colab import drive\ndrive.mount('/content/drive')"),
        cell("import os\nif not os.path.exists('/content/btp/data/splits/rdd_v1/images/test'):\n"
             "    !cp /content/drive/MyDrive/BTP/btp_colab.zip /content/ && mkdir -p /content/btp && unzip -q /content/btp_colab.zip -d /content/btp && rm /content/btp_colab.zip\n"
             "!ls /content/btp/data/splits/rdd_v1/images/train | wc -l   # expect 26868"),
        cell("!pip -q install ultralytics==8.4.171"),
        cell("!mkdir -p /content/drive/MyDrive/BTP/runs\n"
             "!cd /content/btp && python training/colab_train.py $RUN /content/drive/MyDrive/BTP/runs"),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    (OUT / "BTP_colab_baselines.ipynb").write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    build_notebook()
    build_zip()
