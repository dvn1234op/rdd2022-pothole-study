"""Single entry point for every training run in this project.

    python training/train.py configs/experiments/<name>.yaml

The experiment yaml holds: name, model, data, seed and optional `overrides` / `init`.
Everything else comes from configs/protocol.yaml (the fixed training protocol shared by ALL runs).
model:
  * an Ultralytics checkpoint or yaml (yolov8s.pt, yolo11s.pt, rtdetr-l.pt, ...)
  * custom:<VARIANT>   -> models/multires.py ablation variant (A0..A7), trained with MultiResTrainer
Outputs go to experiments/<name>/ (Ultralytics run dir) and the resolved config is saved alongside.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cfg(path):
    exp = yaml.safe_load(Path(path).read_text())
    proto = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())
    family = "rtdetr" if "rtdetr" in str(exp["model"]) else "yolo"
    args = {**proto["common"], **proto.get(family, {}), **exp.get("overrides", {})}
    args.update(data=str(ROOT / exp["data"]), seed=exp.get("seed", 0), project=str(ROOT / "experiments"), name=exp["name"], exist_ok=True)
    return exp, args


def main(path):
    exp, args = load_cfg(path)
    run_dir = ROOT / "experiments" / exp["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump({"experiment": exp, "train_args": args}, sort_keys=False))
    model = str(exp["model"])
    if model.startswith("custom:"):
        from models.multires import MultiResTrainer
        MultiResTrainer.variant = model.split(":", 1)[1]
        MultiResTrainer.init_weights = str(ROOT / exp.get("init", "checkpoints/pretrained/yolov8s.pt"))
        MultiResTrainer.init_scope = exp.get("init_scope", "backbone")
        trainer = MultiResTrainer(overrides={**args, "model": "yolov8s.yaml"})
        trainer.train()
    else:
        from ultralytics import RTDETR, YOLO
        w = ROOT / "checkpoints/pretrained" / model if (ROOT / "checkpoints/pretrained" / model).exists() else model
        m = (RTDETR if "rtdetr" in model else YOLO)(str(w))
        if "init" in exp:  # yaml architecture + transfer of every pretrained tensor that has a counterpart
            n = load_p2_shifted(m.model, ROOT / exp["init"]) if "-p2" in model else None
            if n is None:
                m.load(str(ROOT / exp["init"]))
            else:
                m.ckpt = {"model": m.model}  # train() seeds from self.model only when ckpt is set (else rebuilds)
                print(f"[train] {model}: transferred {n} tensors from {exp['init']} (P2 index remap)")
        m.train(**args)


def load_p2_shifted(model, weights):
    """Load a P3-P5 Ultralytics checkpoint (e.g. yolo26s.pt) into its -p2 variant. The -p2 head inserts a 3-layer P2
    top-down block + its 3-layer bottom-up return right after the P3 output, so the P3->P4->P5 bottom-up layers shift
    by +6 and Detect level j (P3/P4/P5) becomes level j+1 (level 0 = P2, no pretrained counterpart). Layers before
    the insertion are identical. Only shape-matching tensors are copied. Returns the number of tensors loaded."""
    import torch
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    src_model = (ck.get("ema") or ck["model"]).float()
    src, dst = src_model.state_dict(), model.state_dict()
    n_src = len(src_model.model)  # layers incl. Detect
    p3_out = src_model.model[-1].f[0]  # Detect input of P3 in the source (e.g. 16)
    shift = len(model.model) - n_src  # 6
    lmap = {i: (i if i <= p3_out else i + shift) for i in range(n_src)}
    det = n_src - 1
    keep = {}
    for k, v in src.items():
        parts = k.split(".")
        if parts[0] != "model" or not parts[1].isdigit():
            continue
        i = int(parts[1])
        rest = parts[2:]
        if i == det and len(rest) > 1 and rest[1].isdigit() and rest[0] in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
            rest = [rest[0], str(int(rest[1]) + 1), *rest[2:]]
        key = ".".join(["model", str(lmap[i]), *rest])
        if key in dst and dst[key].shape == v.shape:
            keep[key] = v
    model.load_state_dict(keep, strict=False)
    return len(keep)


if __name__ == "__main__":
    main(sys.argv[1])
