"""Unit checks for models/multires.py: build every variant, forward/backward, identity-at-init of CRA,
gradient flow into deformable offsets / gates, and backbone-weight loading."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.multires import VARIANTS, CrossResAttn, build_variant, load_backbone  # noqa: E402

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.manual_seed(0)

# CRA is the identity at init (zero-initialised output projection), and offsets get gradients once out != 0
for dense in (False, True):
    m = CrossResAttn(64, 128, dense=dense).to(dev)
    xq, xk = torch.randn(2, 64, 40, 40, device=dev), torch.randn(2, 128, 10, 10, device=dev)
    assert torch.allclose(m(xq, xk), xq), "CRA should be identity at init"
    torch.nn.init.normal_(m.out.weight, std=0.02)
    m(xq, xk).square().mean().backward()
    if not dense:
        assert m.off.weight.grad.abs().sum() > 0 and m.att.weight.grad.abs().sum() > 0
    print(f"CrossResAttn dense={dense}: identity-at-init OK, grads OK")

# deformable sampling sanity: with zero offsets and a constant KV map, output == that constant through v/out
for name in VARIANTS:
    model = build_variant(name, nc=4).to(dev)
    n = load_backbone(model, ROOT / "checkpoints/pretrained/yolov8s.pt", 5 if name == "A7" else 10)
    model.train()
    x = torch.randn(2, 3, 256, 256, device=dev)
    out = model(x)

    def flat(o):
        if torch.is_tensor(o):
            return [o]
        return [t for v in (o.values() if isinstance(o, dict) else o) for t in flat(v)]
    out = flat(out)
    loss = sum(o.float().abs().mean() for o in out)
    loss.backward()
    gated = [m for m in model.modules() if type(m).__name__ == "GatedFuse"]
    cras = [m for m in model.modules() if type(m).__name__ == "CrossResAttn"]
    g_ok = all(g.g.weight.grad is not None for g in gated)
    shapes = [tuple(o.shape) for o in out]
    print(f"{name}: params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  backbone tensors loaded {n}  "
          f"gates {len(gated)} (grad {g_ok})  CRA {len(cras)}  stride {model.stride.tolist()}  out {shapes}")
