"""Train one standard Ultralytics baseline on Google Colab under the SAME protocol as training/train.py.

    python colab_train.py <experiment_name> <drive_runs_dir>

Expects the package from scripts/make_colab_package.py unzipped at /content/btp (data at /content/btp/data/splits/rdd_v1).
Runs are written to <drive_runs_dir>/<experiment_name>/ on Google Drive and auto-resume from last.pt after a
disconnect. Only `workers` differs from the laptop (Colab free has 2 CPU cores) — this changes data-loading
throughput, not the training recipe. Evaluation is NOT done here: download weights/best.pt and run
evaluation/evaluate.py on the laptop so all models are scored and timed on the same hardware.
"""
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

PKG = Path("/content/btp")


def main(name, runs_dir):
    exp = yaml.safe_load((PKG / f"configs/experiments/{name}.yaml").read_text())
    proto = yaml.safe_load((PKG / "configs/protocol.yaml").read_text())
    args = {**proto["common"], **proto["yolo"], **exp.get("overrides", {})}
    data_yaml = PKG / "rdd_v1_colab.yaml"
    data_yaml.write_text(f"path: {PKG}/data/splits/rdd_v1\ntrain: images/train\nval: images/val\ntest: images/test\n"
                         "names:\n  0: D00\n  1: D10\n  2: D20\n  3: D40\n")
    args.update(data=str(data_yaml), seed=exp.get("seed", 0), project=runs_dir, name=name, exist_ok=True, workers=2)
    run = Path(runs_dir) / name
    last = run / "weights/last.pt"
    if (run / "weights/best.pt").exists() and (run / "results.csv").exists() and \
            len((run / "results.csv").read_text().strip().splitlines()) - 1 >= args["epochs"]:
        print("already finished:", run)
        return
    if last.exists():
        print("resuming from", last)
        YOLO(str(last)).train(resume=True)
    else:
        run.mkdir(parents=True, exist_ok=True)
        (run / "resolved_config.yaml").write_text(yaml.safe_dump({"experiment": exp, "train_args": args, "hardware": "colab"}, sort_keys=False))
        YOLO(str(PKG / "checkpoints/pretrained" / exp["model"])).train(**args)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
