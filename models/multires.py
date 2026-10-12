"""Multi-resolution ablation family (A0-A7) built on the YOLOv8s backbone + PAN neck + YOLOv8 Detect head.

Why this design and not a from-scratch dual-patch ViT (patch 8 / patch 16) + RT-DETR decoder?
  * A dual-patch ViT trained from scratch on ~27k images, under a 6 GB / 20-epoch budget, would confound
    "multi-resolution" with "no ImageNet/COCO pretraining" and "transformer vs CNN" - the ablation would not be able
    to attribute anything to the multi-resolution mechanism. A patch-8 global-attention branch at 640 px also has
    6,400 tokens per image, which does not fit the hardware.
  * The research hypothesis is about the REPRESENTATION (preserve high-resolution detail + inject low-resolution
    context). So we hold the backbone (COCO-pretrained YOLOv8s layers 0-9), loss, head, data and training budget
    fixed, and vary only the multi-resolution components. All ablation variants (incl. A0) load ONLY backbone
    weights; neck/head start from random init, so no variant gets a pretrained-neck advantage.

Components
  * High-resolution branch = an extra stride-4 (P2) pathway in the neck + a P2 detection level
    (the original proposal's "patch-8" fine branch corresponds to stride 8 = P3, which YOLOv8 ALREADY has;
    the only way to add finer resolution than the baseline is stride 4).
  * Low-resolution branch  = the stride-16/32 (P4/P5) context pathway.
  * CrossResAttn (deformable): each query location of feature map Q predicts, per head, K sampling offsets and
    weights into feature map KV (another resolution) - Deformable-DETR-style sampling (Zhu et al., ICLR 2021),
    applied across two pyramid levels. Bidirectional: fine<-coarse (context into P2/P3) and coarse<-fine
    (detail into P5). Output is added residually through a zero-initialised projection.
  * DenseCrossAttn: same interface but full softmax attention over all KV positions (for A5).
  * GatedFuse (adaptive fusion gate): F = g*H + (1-g)*P(L), g = sigmoid(conv1x1([H, P(L)])) per location/channel,
    replacing the concatenation in the top-down path where an up-sampled coarse map meets a fine skip connection.

Variants
  A0  yolov8s (P3-P5)                                   baseline (backbone-only init)
  A1  yolov8s-p2 (P2-P5)                                multi-resolution representation only
  A2  A1 + deformable cross-resolution attention
  A3  A1 + adaptive fusion gates
  A4  A1 + deformable CRA + gates                       full model
  A5  A4 with dense cross-attention instead of deformable
  A6  A0 + deformable CRA + gates                       full model without the high-resolution (P2) branch
  A7  P2/P3-only network (backbone truncated at P3) + gate   full model without the low-resolution branch
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2f, Concat
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK


# ----------------------------------------------------------------------------------------------------------------------
class CrossResAttn(nn.Module):
    """Deformable cross-resolution attention: queries from x_q sample K points per head from x_kv."""

    def __init__(self, c_q, c_kv, heads=4, points=4, dense=False):
        super().__init__()
        self.h, self.k, self.dense = heads, points, dense
        self.d = max(16, c_q // heads)
        inner = self.h * self.d
        self.q_norm = nn.GroupNorm(1, c_q)
        self.v = nn.Conv2d(c_kv, inner, 1)
        if dense:
            self.q = nn.Conv2d(c_q, inner, 1)
            self.kproj = nn.Conv2d(c_kv, inner, 1)
        else:
            self.off = nn.Conv2d(c_q, self.h * self.k * 2, 1)
            self.att = nn.Conv2d(c_q, self.h * self.k, 1)
            nn.init.zeros_(self.off.weight)
            # initial offsets on a small ring (like Deformable DETR) so the K points start distinct
            ang = torch.arange(self.h * self.k, dtype=torch.float32) * (2 * math.pi / (self.h * self.k))
            ring = torch.stack([ang.cos(), ang.sin()], 1) * (1 + torch.arange(self.k).repeat(self.h)[:, None] % 2)
            self.off.bias.data = ring.reshape(-1)
            nn.init.zeros_(self.att.weight); nn.init.zeros_(self.att.bias)
        self.out = nn.Conv2d(inner, c_q, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # identity at init

    def forward(self, x_q, x_kv):
        B, C, H, W = x_q.shape
        Hk, Wk = x_kv.shape[-2:]
        q = self.q_norm(x_q)
        v = self.v(x_kv)
        if self.dense:
            qq = self.q(q).flatten(2).reshape(B, self.h, self.d, H * W).transpose(2, 3)
            kk = self.kproj(x_kv).flatten(2).reshape(B, self.h, self.d, Hk * Wk).transpose(2, 3)
            vv = v.flatten(2).reshape(B, self.h, self.d, Hk * Wk).transpose(2, 3)
            o = F.scaled_dot_product_attention(qq, kk, vv)  # B,h,HW,d
            o = o.transpose(2, 3).reshape(B, self.h * self.d, H, W)
        else:
            off = self.off(q).reshape(B, self.h, self.k, 2, H, W)
            att = self.att(q).reshape(B, self.h, self.k, H, W).softmax(2)
            ys = (torch.arange(H, device=x_q.device, dtype=x_q.dtype) + 0.5) / H * 2 - 1
            xs = (torch.arange(W, device=x_q.device, dtype=x_q.dtype) + 0.5) / W * 2 - 1
            ref = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), 0)            # 2,H,W  (x, y) in [-1,1]
            scale = torch.tensor([2.0 / Wk, 2.0 / Hk], device=x_q.device, dtype=x_q.dtype).view(1, 1, 1, 2, 1, 1)
            loc = ref.view(1, 1, 1, 2, H, W) + off * scale                         # offsets in KV-pixel units
            loc = loc.permute(0, 1, 4, 5, 2, 3).reshape(B * self.h, H, W * self.k, 2)
            vv = v.reshape(B * self.h, self.d, Hk, Wk)
            s = F.grid_sample(vv, loc.to(vv.dtype), mode="bilinear", padding_mode="zeros", align_corners=False)
            s = s.reshape(B, self.h, self.d, H, W, self.k)
            o = (s * att.permute(0, 1, 3, 4, 2).unsqueeze(2)).sum(-1).reshape(B, self.h * self.d, H, W)
        return x_q + self.out(o)


class CRABlock(nn.Module):
    """Wraps an existing neck layer: y = CRA(layer(x), kv). Takes inputs [x, kv]."""

    def __init__(self, layer, c_q, c_kv, dense=False):
        super().__init__()
        self.layer = layer
        self.cra = CrossResAttn(c_q, c_kv, dense=dense)

    def forward(self, xs):
        return self.cra(self.layer(xs[0]), xs[1])


class GatedFuse(nn.Module):
    """Adaptive fusion gate. Inputs [L_up (coarse, upsampled), H (fine skip)] -> c_h channels."""

    def __init__(self, c_l, c_h):
        super().__init__()
        self.p = nn.Sequential(nn.Conv2d(c_l, c_h, 1, bias=False), nn.BatchNorm2d(c_h))
        self.g = nn.Conv2d(2 * c_h, c_h, 1)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)  # g = 0.5 at init

    def forward(self, xs):
        low, high = xs
        low = self.p(low)
        g = torch.sigmoid(self.g(torch.cat([high, low], 1)))
        return g * high + (1 - g) * low


# ----------------------------------------------------------------------------------------------------------------------
BACKBONE = [
    [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, "C2f", [128, True]],
    [-1, 1, "Conv", [256, 3, 2]], [-1, 6, "C2f", [256, True]], [-1, 1, "Conv", [512, 3, 2]],
    [-1, 6, "C2f", [512, True]], [-1, 1, "Conv", [1024, 3, 2]], [-1, 3, "C2f", [1024, True]], [-1, 1, "SPPF", [1024, 5]],
]
HEAD_P3 = [  # yolov8 PAN
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 6], 1, "Concat", [1]], [-1, 3, "C2f", [512]],          # 10-12
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],          # 13-15 P3
    [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]], [-1, 3, "C2f", [512]],                         # 16-18 P4
    [-1, 1, "Conv", [512, 3, 2]], [[-1, 9], 1, "Concat", [1]], [-1, 3, "C2f", [1024]],                         # 19-21 P5
    [[15, 18, 21], 1, "Detect", ["nc"]],
]
HEAD_P2 = [  # yolov8-p2 PAN
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 6], 1, "Concat", [1]], [-1, 3, "C2f", [512]],          # 10-12
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],          # 13-15
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 2], 1, "Concat", [1]], [-1, 3, "C2f", [128]],          # 16-18 P2
    [-1, 1, "Conv", [128, 3, 2]], [[-1, 15], 1, "Concat", [1]], [-1, 3, "C2f", [256]],                         # 19-21 P3
    [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]], [-1, 3, "C2f", [512]],                         # 22-24 P4
    [-1, 1, "Conv", [512, 3, 2]], [[-1, 9], 1, "Concat", [1]], [-1, 3, "C2f", [1024]],                         # 25-27 P5
    [[18, 21, 24, 27], 1, "Detect", ["nc"]],
]
BACKBONE_TO_P3 = BACKBONE[:5]
HEAD_HIRES_ONLY = [  # backbone truncated at P3 (layer 4); P2 + P3 detection only
    [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 2], 1, "Concat", [1]], [-1, 3, "C2f", [128]],          # 5-7 P2
    [-1, 1, "Conv", [128, 3, 2]], [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],                          # 8-10 P3
    [[7, 10], 1, "Detect", ["nc"]],
]

# variant -> (backbone, head, gate concat layers (top-down), CRA specs [(layer, kv_layer)], dense)
VARIANTS = {
    "A0": (BACKBONE, HEAD_P3, [], [], False),
    "A1": (BACKBONE, HEAD_P2, [], [], False),
    "A2": (BACKBONE, HEAD_P2, [], [(18, 9), (27, 18)], False),
    "A3": (BACKBONE, HEAD_P2, [11, 14, 17], [], False),
    "A4": (BACKBONE, HEAD_P2, [11, 14, 17], [(18, 9), (27, 18)], False),
    "A5": (BACKBONE, HEAD_P2, [11, 14, 17], [(18, 9), (27, 18)], True),
    "A6": (BACKBONE, HEAD_P3, [11, 14], [(15, 9), (21, 15)], False),
    "A7": (BACKBONE_TO_P3, HEAD_HIRES_ONLY, [6], [], False),
}


def variant_yaml(name, nc):
    bb, hd, *_ = VARIANTS[name]
    return {"nc": nc, "scales": {"s": [0.33, 0.50, 1024]}, "scale": "s", "backbone": copy.deepcopy(bb), "head": copy.deepcopy(hd)}


def _out_ch(m):
    if isinstance(m, C2f):
        return m.cv2.conv.out_channels
    if hasattr(m, "conv"):
        return m.conv.out_channels
    if hasattr(m, "cv2"):
        return m.cv2.conv.out_channels
    raise ValueError(type(m))


def build_variant(name, nc=4, verbose=False):
    _, _, gates, cras, dense = VARIANTS[name]
    model = DetectionModel(variant_yaml(name, nc), ch=3, nc=nc, verbose=verbose)
    seq = model.model
    ch = {}  # output channels per layer index
    for m in seq:
        try:
            ch[m.i] = _out_ch(m)
        except (ValueError, AttributeError):  # Upsample / Concat / Detect carry no own channel count
            pass
    for i in gates:  # Concat([up(coarse), fine]) followed by C2f -> GatedFuse + C2f(c_fine -> same out)
        cat = seq[i]
        assert isinstance(cat, Concat)
        up_src, skip = cat.f
        c_l = ch[i - 2]  # layer i-1 is nn.Upsample of layer i-2 (keeps its channels)
        c_h = ch[skip]
        g = GatedFuse(c_l, c_h)
        g.i, g.f, g.type, g.np = cat.i, cat.f, "GatedFuse", sum(p.numel() for p in g.parameters())
        seq[i] = g
        old = seq[i + 1]
        new = C2f(c_h, _out_ch(old), n=len(old.m), shortcut=False)
        new.i, new.f, new.type, new.np = old.i, old.f, old.type, sum(p.numel() for p in new.parameters())
        seq[i + 1] = new
    for li, kv in cras:
        old = seq[li]
        blk = CRABlock(old, _out_ch(old), ch[kv], dense=dense)
        blk.i, blk.f, blk.type, blk.np = old.i, [old.f, kv], "CRABlock", sum(p.numel() for p in blk.parameters())
        seq[li] = blk
        if kv not in model.save:
            model.save.append(kv)
    model.save = sorted(set(model.save))
    model.variant = name
    return model


def load_backbone(model, weights, n_layers=10):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    src = (ck.get("ema") or ck["model"]).float().state_dict()
    dst = model.state_dict()
    keep = {k: v for k, v in src.items() if k.split(".")[1].isdigit() and int(k.split(".")[1]) < n_layers
            and k in dst and dst[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return len(keep)


# yolov8s layer index -> variant layer index for the neck/head (backbone 0-9 is identical in every variant but A7).
# HEAD_P2 inserts the P2 top-down/bottom-up block (16-21), so yolov8s P4/P5 bottom-up layers 16-21 move to 22-27 and
# Detect level j (P3/P4/P5) becomes level j+1 (P2 is level 0 and has no pretrained counterpart).
FULL_INIT_MAP = {
    "HEAD_P3": ({i: i for i in range(10, 23)}, 0),
    "HEAD_P2": ({**{i: i for i in range(10, 16)}, **{i: i + 6 for i in range(16, 22)}, 22: 28}, 1),
}


def load_full(model, weights, variant):
    """Transfer every COCO yolov8s tensor whose layer has a counterpart in `variant` and whose shape matches
    (backbone + PAN neck + Detect box/cls stems). New modules (P2 path, GatedFuse, CRA) and shape-changed layers
    (C2f after a gate, the nc-specific cls output) keep their own init. Returns the number of tensors loaded."""
    _, hd, _, cras, _ = VARIANTS[variant]
    if hd is HEAD_HIRES_ONLY:
        raise ValueError("full init is not defined for A7 (truncated backbone)")
    lmap, lvl_shift = FULL_INIT_MAP["HEAD_P2" if hd is HEAD_P2 else "HEAD_P3"]
    lmap = {**{i: i for i in range(10)}, **lmap}
    detect_src = max(lmap)
    wrapped = {li for li, _ in cras}  # CRABlock keeps the original layer under `.layer`
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    src = (ck.get("ema") or ck["model"]).float().state_dict()
    dst = model.state_dict()
    keep = {}
    for k, v in src.items():
        parts = k.split(".")  # model.<idx>.<rest...>
        if not parts[1].isdigit() or int(parts[1]) not in lmap:
            continue
        i = int(parts[1])
        j, rest = lmap[i], parts[2:]
        if i == detect_src and rest[0] in ("cv2", "cv3"):  # per-level branches: shift level index
            rest = [rest[0], str(int(rest[1]) + lvl_shift), *rest[2:]]
        key = ".".join(["model", str(j), *(["layer"] if j in wrapped else []), *rest])
        if key in dst and dst[key].shape == v.shape:
            keep[key] = v
    model.load_state_dict(keep, strict=False)
    return len(keep)


class MultiResTrainer(DetectionTrainer):
    variant = "A0"
    init_weights = None
    init_scope = "backbone"  # "backbone" (ablation protocol: neck/head from scratch) or "full" (all matching layers)

    def get_model(self, cfg=None, weights=None, verbose=True):
        m = build_variant(self.variant, nc=self.data["nc"], verbose=verbose and RANK == -1)
        n_bb = 5 if self.variant == "A7" else 10
        if self.init_weights and self.init_scope == "full":
            n = load_full(m, self.init_weights, self.variant)
            print(f"[MultiResTrainer] variant {self.variant}: loaded {n} tensors (full init, all matching layers) from {self.init_weights}")
        elif self.init_weights:
            n = load_backbone(m, self.init_weights, n_bb)
            print(f"[MultiResTrainer] variant {self.variant}: loaded {n} backbone tensors (layers 0-{n_bb - 1}) from {self.init_weights}")
        return m
