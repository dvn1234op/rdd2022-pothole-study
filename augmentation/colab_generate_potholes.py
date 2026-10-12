"""Generate realistic synthetic potholes (D40) for D40-poor domains: road-aware, perspective-aware,
diffusion-harmonised copy-paste of REAL potholes, with automatic quality filters. Built for Colab / Kaggle (T4).

Why not plain text-prompt inpainting (the previous synthetic set)?  RESEARCH_AUDIT.md §3: 0/60 audited boxes contained
a visible pothole, 38 % were off-road, and inpainting sometimes erased real cracks/signs. This script fixes each cause:
  1. WHERE   road mask from SegFormer (Cityscapes "road") -> a box is accepted only if >= --min_road of it is road;
             no overlap with any existing labelled damage (we never cover/erase a labelled crack).
  2. SIZE    geometry comes from a real D40 box at a similar vertical position (perspective: far = small, near = big).
  3. WHAT    the appearance is a REAL pothole crop (train split, any domain) -> a pothole is guaranteed to be there.
  4. BLEND   Poisson blending, then Stable-Diffusion inpainting at low strength inside a small mask only, to
             harmonise texture/lighting with the target road. Pixels outside the mask are copied back unchanged.
  5. CHECK   CLIP zero-shot filter (pothole vs crack / smooth road / shadow / manhole / patch) on the box crop;
             optional detector filter. Everything (accepted + rejected, scores) goes to manifest.csv.
--mode text  reproduces the old method (pure text inpainting, strength 1.0) under the same road/size rules, for the
             ablation "real-pothole harmonised paste" vs "text-only diffusion".

Colab setup (GPU runtime):
    !pip -q install diffusers transformers accelerate
    from google.colab import drive; drive.mount('/content/drive')
    !unzip -q /content/drive/MyDrive/btp/gen_pkg.zip -d /content          # made by augmentation/pack_for_generation.py
    !python colab_generate_potholes.py --pkg /content/gen_pkg --out /content/drive/MyDrive/btp/gen_v2 --n 3000

Re-running with the same --out RESUMES (images already in manifest.csv are skipped) - safe across disconnects.
Outputs in --out:  images/syn_*.jpg, labels/syn_*.txt (source labels + new D40 lines, YOLO format),
manifest.csv, audit/sheet_*.jpg (before/after crops for manual checking), summary.json.
"""
import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

D40 = 3
CLIP_PROMPTS = ["a photo of a pothole in an asphalt road", "a photo of a crack in an asphalt road",
                "a photo of a smooth asphalt road", "a photo of a shadow on a road",
                "a photo of a manhole cover on a road", "a photo of a patched repair on an asphalt road"]


# ---------------------------------------------------------------------------------------------------------------- io
def read_labels(p):
    if not p.exists() or p.stat().st_size == 0:  # negatives have an empty label file
        return np.zeros((0, 5))
    a = np.loadtxt(p, ndmin=2)
    return a.reshape(-1, 5) if a.size else np.zeros((0, 5))


def xywhn_to_xyxy(b, w, h):
    cx, cy, bw, bh = b
    return np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


# ------------------------------------------------------------------------------------------------------------ models
class RoadSeg:
    def __init__(self, name, dev):
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        self.proc = SegformerImageProcessor.from_pretrained(name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(name).to(dev).eval().half()
        self.road = [i for i, n in self.model.config.id2label.items() if n.lower() == "road"][0]
        self.dev = dev

    @torch.no_grad()
    def __call__(self, img):  # img: HxWx3 RGB uint8 -> bool mask HxW
        x = self.proc(images=img, return_tensors="pt")["pixel_values"].to(self.dev).half()
        logits = self.model(pixel_values=x).logits
        seg = self.proc.post_process_semantic_segmentation(SimpleNamespace(logits=logits.float()),
                                                           target_sizes=[img.shape[:2]])[0]
        return (seg == self.road).cpu().numpy()


class ClipFilter:
    def __init__(self, name, dev):
        from transformers import CLIPModel, CLIPProcessor
        self.proc = CLIPProcessor.from_pretrained(name)
        self.model = CLIPModel.from_pretrained(name).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, crop):  # RGB uint8 -> probability of the pothole prompt
        x = self.proc(text=CLIP_PROMPTS, images=Image.fromarray(crop), return_tensors="pt", padding=True).to(self.dev)
        return float(self.model(**x).logits_per_image.softmax(-1)[0, 0])


class Harmoniser:
    def __init__(self, name, dev):
        from diffusers import StableDiffusionInpaintPipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(name, torch_dtype=torch.float16,
                                                                   safety_checker=None).to(dev)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.enable_attention_slicing()  # fits next to SegFormer + CLIP on a 6 GB laptop GPU

    def __call__(self, img, mask, prompt, neg, strength, steps, seed):
        """Inpaint a square window around the mask at 512 px; return full image with ONLY masked pixels changed."""
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        side = int(min(h, w, max(256, 3 * max(np.ptp(ys), np.ptp(xs)))))  # square window -> no aspect distortion
        cy, cx = int(ys.mean()), int(xs.mean())
        y0, x0 = np.clip(cy - side // 2, 0, max(0, h - side)), np.clip(cx - side // 2, 0, max(0, w - side))
        y1, x1 = min(h, y0 + side), min(w, x0 + side)
        win, mwin = img[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        g = torch.Generator("cuda").manual_seed(seed)
        out = self.pipe(prompt=prompt, negative_prompt=neg, image=Image.fromarray(win).resize((512, 512)),
                        mask_image=Image.fromarray((mwin * 255).astype(np.uint8)).resize((512, 512)),
                        strength=strength, num_inference_steps=steps, guidance_scale=7.5, generator=g,
                        height=512, width=512).images[0]
        out = np.array(out.resize((x1 - x0, y1 - y0), Image.BICUBIC))
        soft = cv2.GaussianBlur(mwin.astype(np.float32), (0, 0), 2.0)[..., None]  # feathered seam
        res = img.copy()
        res[y0:y1, x0:x1] = (soft * out + (1 - soft) * win).astype(np.uint8)
        return res


# --------------------------------------------------------------------------------------------------------- placement
def donor_bank(pkg, split, min_px, min_contrast=0.0):
    """Every real D40 box in train with both sides >= min_px (at the stored 640-long-side scale) whose interior is
    visibly different from its own road: |mean L(pothole) - mean L(surrounding road)| >= min_contrast (LAB L, 0-255).
    Low-contrast donors (puddle-flat or blurry potholes) produced the 'invisible' grey discs in the first test."""
    rows = []
    for stem in split[(split.split == "train") & split.has_D40].image_id:
        ip = pkg / "images" / f"{stem}.jpg"
        if not ip.exists():
            continue
        labs = [l for l in read_labels(pkg / "labels" / f"{stem}.txt") if int(l[0]) == D40]
        if not labs:
            continue
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        h, w = img.shape
        for c, *b in labs:
            x0, y0, x1, y1 = xywhn_to_xyxy(b, w, h)
            if min(x1 - x0, y1 - y0) < min_px:
                continue
            box = np.array([x0, y0, x1, y1])
            inner = ellipse_mask(img.shape, box, 0.8)
            ring = ellipse_mask(img.shape, box, 1.6) & ~ellipse_mask(img.shape, box, 1.1)
            if inner.sum() < 10 or ring.sum() < 10:
                continue
            contrast = float(img[ring].mean() - img[inner].mean())  # > 0: pothole darker than its road
            if contrast >= min_contrast:  # signed: light/dusty 'potholes' became bright blobs on dark asphalt
                rows.append(dict(stem=stem, cx=b[0], cy=b[1], bw=b[2], bh=b[3], W=w, H=h, contrast=round(contrast, 1)))
    return pd.DataFrame(rows)


def propose(rng, img_shape, road, boxes, donors, a, geom=None):
    """Pick a real donor (appearance) and a box on the road. None if no valid spot.
    geom given (real D40 boxes of the TARGET domain): width and vertical position come from a real pothole of the
    same domain, so synthetic sizes follow that domain's camera/resolution (Norway: median 10 px at 640) - the G2 set
    took sizes from the donor's own (mostly Indian/Japanese, large) potholes and was 4x too large for Norway."""
    h, w = img_shape[:2]
    for _ in range(a.tries):
        d = donors.iloc[rng.integers(len(donors))]
        g = d if geom is None or not len(geom) else geom.iloc[rng.integers(len(geom))]
        bw_px = g.bw * w * rng.uniform(0.9, 1.1)                  # size relative to image width
        bh_px = bw_px * (d.bh * d.H) / (d.bw * d.W)               # keep the appearance donor's aspect ratio
        cy = np.clip(g.cy + rng.normal(0, 0.03), 0.05, 0.95) * h
        if not (a.min_px <= bw_px <= 0.45 * w and a.min_px <= bh_px <= 0.45 * h):
            continue
        row = road[int(np.clip(cy, 0, h - 1))]
        xs = np.nonzero(row)[0]
        if len(xs) < bw_px:
            continue
        cx = rng.choice(xs)
        box = np.array([cx - bw_px / 2, cy - bh_px / 2, cx + bw_px / 2, cy + bh_px / 2])
        if box[0] < 3 or box[1] < 3 or box[2] > w - 4 or box[3] > h - 4:
            continue
        x0, y0, x1, y1 = box.astype(int)
        if road[y0:y1, x0:x1].mean() < a.min_road:
            continue
        if any(iou(box, b) > 0.0 for b in boxes):  # never touch an existing labelled damage
            continue
        return d, box
    return None


def paste(img, donor_img, d, box, method="alpha"):
    """Insert the real donor pothole (with a margin of its own road) into img at box.
    alpha  : colour-match the donor's road ring to the target's road ring (per-channel mean/std), then feathered alpha
             blend - keeps the pothole's dark interior and texture (default).
    poisson: cv2.seamlessClone - seamless, but matches gradients only and tends to wash potholes out to grey blobs."""
    H, W = donor_img.shape[:2]
    dx0, dy0, dx1, dy1 = xywhn_to_xyxy((d.cx, d.cy, d.bw, d.bh), W, H)
    mx, my = 0.15 * (dx1 - dx0), 0.15 * (dy1 - dy0)
    sx0, sy0 = int(max(0, dx0 - mx)), int(max(0, dy0 - my))
    sx1, sy1 = int(min(W, dx1 + mx)), int(min(H, dy1 + my))
    src = donor_img[sy0:sy1, sx0:sx1]
    scale = (box[2] - box[0]) / max(1, dx1 - dx0)
    if scale > 1.5:  # strong upscaling of a small donor gives a blurry, unrealistic pothole
        return None
    src = cv2.resize(src, (max(4, int(src.shape[1] * scale)), max(4, int(src.shape[0] * scale))), cv2.INTER_AREA)
    m = np.zeros(src.shape[:2], np.uint8)
    cv2.ellipse(m, (src.shape[1] // 2, src.shape[0] // 2), (max(1, src.shape[1] // 2 - 1), max(1, src.shape[0] // 2 - 1)),
                0, 0, 360, 255, -1)
    center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
    h, w = img.shape[:2]  # the patch must lie fully inside the image
    px0, py0 = center[0] - src.shape[1] // 2, center[1] - src.shape[0] // 2
    if px0 < 1 or py0 < 1 or px0 + src.shape[1] >= w - 1 or py0 + src.shape[0] >= h - 1:
        return None
    if method == "poisson":
        return cv2.seamlessClone(src, img, m, center, cv2.NORMAL_CLONE)
    dst = img[py0:py0 + src.shape[0], px0:px0 + src.shape[1]].astype(np.float32)
    inner = np.zeros_like(m)  # the pothole itself (box without the 15 % margin) -> excluded from the road statistics
    ix, iy = int(mx * scale), int(my * scale)
    inner[iy:src.shape[0] - iy, ix:src.shape[1] - ix] = 1
    ring = (m > 0) & (inner == 0)
    if ring.sum() < 20:
        ring = np.ones_like(m, bool)
    # colour transfer in LAB: lightness gets mean + (bounded) contrast matched, colour channels are only shifted,
    # so a grey Indian road never turns into rainbow noise on a snowy Norwegian one
    s = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    t = cv2.cvtColor(dst.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    mu_s, sd_s, mu_t, sd_t = s[ring].mean(0), s[ring].std(0) + 1e-3, t[ring].mean(0), t[ring].std(0) + 1e-3
    s[..., 0] = (s[..., 0] - mu_s[0]) * np.clip(sd_t[0] / sd_s[0], 0.6, 1.4) + mu_t[0]
    s[..., 1:] = s[..., 1:] - mu_s[1:] + mu_t[1:]
    s = cv2.cvtColor(np.clip(s, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    feather = max(2.0, 0.12 * min(src.shape[:2]))  # alpha ramps 0 -> 1 over `feather` px inside the ellipse edge
    alpha = np.clip(cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5) / feather, 0, 1)[..., None]
    out = img.copy()
    out[py0:py0 + src.shape[0], px0:px0 + src.shape[1]] = (alpha * s + (1 - alpha) * dst).astype(np.uint8)
    return out


def ellipse_mask(shape, box, grow):
    m = np.zeros(shape[:2], np.uint8)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ax, ay = (box[2] - box[0]) / 2 * grow, (box[3] - box[1]) / 2 * grow
    cv2.ellipse(m, (int(cx), int(cy)), (max(2, int(ax)), max(2, int(ay))), 0, 0, 360, 1, -1)
    return m.astype(bool)


def crop(img, box, margin=0.5, min_side=0):
    h, w = img.shape[:2]
    bw, bh = max(box[2] - box[0], min_side / (1 + 2 * margin)), max(box[3] - box[1], min_side / (1 + 2 * margin))
    box = [(box[0] + box[2]) / 2 - bw / 2, (box[1] + box[3]) / 2 - bh / 2, (box[0] + box[2]) / 2 + bw / 2, (box[1] + box[3]) / 2 + bh / 2]
    x0, y0 = int(max(0, box[0] - margin * bw)), int(max(0, box[1] - margin * bh))
    x1, y1 = int(min(w, box[2] + margin * bw)), int(min(h, box[3] + margin * bh))
    return img[y0:y1, x0:x1]


def box_change(res, img, box):
    """Mean |pixel change| (0-255) inside box - the audit's visibility measure."""
    x0, y0, x1, y1 = [int(v) for v in box]
    return float(np.abs(res[y0:y1, x0:x1].astype(int) - img[y0:y1, x0:x1].astype(int)).mean())


def colour_shift(res, box):
    """LAB chroma distance between the inserted pothole and the road ring around it (catches blue/green blobs)."""
    lab = cv2.cvtColor(res, cv2.COLOR_RGB2LAB).astype(np.float32)
    inner = ellipse_mask(res.shape, box, 0.8)
    ring = ellipse_mask(res.shape, box, 1.8) & ~ellipse_mask(res.shape, box, 1.2)
    return float(np.linalg.norm(lab[inner][:, 1:].mean(0) - lab[ring][:, 1:].mean(0)))


def detected(det, res, box, conf):
    r = det.predict(res, conf=conf, verbose=False)[0]
    pb = r.boxes.xyxy.cpu().numpy()[r.boxes.cls.cpu().numpy() == D40]
    return any(iou(box, q) >= 0.3 for q in pb)


# --------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="folder with images/, labels/, split_v1.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3000, help="number of ACCEPTED synthetic images to produce")
    ap.add_argument("--targets", nargs="+", default=["Czech", "Norway", "United_States"])  # China_MotorBike: close-up
    # coarse asphalt no donor matches, and its D40 AP is already 0.94
    ap.add_argument("--mode", choices=["paste", "text"], default="paste")
    ap.add_argument("--blend", choices=["alpha", "poisson"], default="alpha", help="paste mode: how the donor is inserted")
    ap.add_argument("--attempts", type=int, default=4, help="donor/placement tries per target image")
    ap.add_argument("--harm_region", choices=["ring", "full"], default="ring",
                    help="paste mode: SD repaints only the seam ring (keeps real pothole texture) or the whole ellipse")
    ap.add_argument("--donor_min_px", type=float, default=20, help="min donor pothole side in px (sharper texture)")
    ap.add_argument("--donor_contrast", type=float, default=10.0, help="min L(road)-L(pothole) of a donor (LAB, 0-255): pothole must be darker")
    ap.add_argument("--clip_margin", type=float, default=1.0, help="context around the box shown to CLIP (x box size)")
    ap.add_argument("--strength", type=float, default=0.3, help="SD strength for harmonisation (paste mode)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--geom", choices=["domain", "donor"], default="domain",
                    help="box size/height from real D40 of the target domain (default) or from the donor itself (G2)")
    ap.add_argument("--min_px", type=float, default=5, help="min pothole side in px at 640 scale")
    ap.add_argument("--min_road", type=float, default=0.9)
    ap.add_argument("--clip_min", type=float, default=0.45, help="min CLIP pothole probability (6-way softmax)")
    ap.add_argument("--pre_clip", type=float, default=0.3, help="CLIP pre-check on the paste, before diffusion")
    ap.add_argument("--max_dab", type=float, default=10.0, help="max LAB chroma shift pothole vs road ring")
    ap.add_argument("--min_change", type=float, default=12.0, help="min mean |pixel change| inside a new box (0-255)")
    ap.add_argument("--detector", default="", help="optional YOLO best.pt trained on TRAIN only; extra filter")
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sd", default="runwayml/stable-diffusion-inpainting")  # cached locally; on Colab use stable-diffusion-v1-5/stable-diffusion-inpainting
    ap.add_argument("--seg", default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch32")
    a = ap.parse_args()

    pkg, out = Path(a.pkg), Path(a.out)
    for s in ("images", "labels", "audit"):
        (out / s).mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    split = pd.read_csv(pkg / "split_v1.csv")
    assert not split[split.image_id.isin([p.stem for p in (pkg / "images").glob("*.jpg")]) & (split.split != "train")].shape[0], \
        "package contains non-train images - refusing (test leakage)"
    targets = split[(split.split == "train") & split.subset.isin(a.targets)].image_id.tolist()
    targets = [t for t in targets if (pkg / "images" / f"{t}.jpg").exists()]
    donors = donor_bank(pkg, split, max(a.min_px, a.donor_min_px), a.donor_contrast)
    print(f"{len(targets)} target images in {a.targets}; {len(donors)} real D40 donor boxes")
    geom_bank = {}
    for stem in split[(split.split == "train") & split.has_D40 & split.subset.isin(a.targets)].image_id:
        for c, *b in read_labels(pkg / "labels" / f"{stem}.txt"):
            if int(c) == D40:
                geom_bank.setdefault(stem.rsplit("_", 1)[0], []).append(dict(cx=b[0], cy=b[1], bw=b[2], bh=b[3]))
    geom_bank = {k: pd.DataFrame(v) for k, v in geom_bank.items()}
    print("same-domain D40 geometry boxes:", {k: len(v) for k, v in geom_bank.items()})

    man_p = out / "manifest.csv"
    man = pd.read_csv(man_p) if man_p.exists() else pd.DataFrame()
    done_src = set(man.source) if len(man) else set()
    n_acc = int(man.accepted.sum()) if len(man) else 0
    seg, clip = RoadSeg(a.seg, dev), ClipFilter(a.clip, dev)
    harm = Harmoniser(a.sd, dev)
    det = None
    if a.detector:
        from ultralytics import YOLO
        det = YOLO(a.detector)

    rng = np.random.default_rng(a.seed + len(man))
    random.Random(a.seed).shuffle(targets)
    rows, sheet, t0 = [], [], time.time()
    for stem in targets:
        if n_acc >= a.n:
            break
        if stem in done_src:
            continue
        img = np.array(Image.open(pkg / "images" / f"{stem}.jpg").convert("RGB"))
        h, w = img.shape[:2]
        labs = read_labels(pkg / "labels" / f"{stem}.txt")
        boxes = [xywhn_to_xyxy(b[1:], w, h) for b in labs]
        road = seg(img)
        info = dict(source=stem, domain=stem.rsplit("_", 1)[0], road_frac=float(road.mean()))
        prompt, neg = "a pothole in an asphalt road, photo", "car, person, text, watermark, painting"
        tried, last = 0, {}
        for k in range(a.attempts):  # up to `attempts` donor/placement pairs per image, first one that passes wins
            p = propose(rng, img.shape, road, boxes, donors, a, geom_bank.get(info["domain"]) if a.geom == "domain" else None)
            if p is None:
                last = dict(reason="no valid road placement")
                break
            d, box = p
            tried += 1
            if a.mode == "paste":
                donor = np.array(Image.open(pkg / "images" / f"{d.stem}.jpg").convert("RGB"))
                pasted = paste(img, donor, d, box, a.blend)
                if pasted is None:
                    last = dict(reason="paste out of bounds")
                    continue
                # cheap pre-check before spending a diffusion call
                # size-aware thresholds: CLIP cannot judge a 10 px pothole and feathering lowers the mean change of
                # small boxes, so fixed thresholds silently removed exactly the tiny potholes the target domains need
                f = min(1.0, np.sqrt((box[2] - box[0]) * (box[3] - box[1])) / 24.0)
                min_change, pre_clip, clip_min = a.min_change * np.sqrt(f), a.pre_clip * f, a.clip_min * f
                if box_change(pasted, img, box) < min_change or clip(crop(pasted, box, a.clip_margin, 64)) < pre_clip:
                    last = dict(reason="pre-check")
                    continue
                hmask = ellipse_mask(img.shape, box, 1.25)
                if a.harm_region == "ring":  # repaint only the seam; the real pothole interior keeps its texture
                    hmask &= ~ellipse_mask(img.shape, box, 0.7)
                res = harm(pasted, hmask, prompt, neg, a.strength, a.steps, a.seed + len(man) + len(rows) + k)
            else:  # text-only diffusion, the old method under the same placement rules
                res = harm(img, ellipse_mask(img.shape, box, 1.0), prompt, neg, 1.0, a.steps, a.seed + len(man) + len(rows) + k)
            vis, sc, dab = box_change(res, img, box), clip(crop(res, box, a.clip_margin, 64)), colour_shift(res, box)
            last = dict(box=box, donor=d.stem, vis=vis, clip=sc, dab=dab, res=res)
            if a.mode != "paste":
                min_change, clip_min = a.min_change, a.clip_min
            if vis < min_change:
                last["reason"] = "invisible"
            elif sc < clip_min:
                last["reason"] = "clip"
            elif dab > a.max_dab:
                last["reason"] = "colour"
            elif det is not None and not detected(det, res, box, a.det_conf):
                last["reason"] = "detector"
            else:
                last["reason"] = ""
                break
        ok = last.get("reason", "x") == ""
        name = f"syn_{a.mode}_{stem}"
        if ok:
            res, box = last["res"], last["box"]
            Image.fromarray(res).save(out / "images" / f"{name}.jpg", quality=95)
            lines = [" ".join(f"{v:.6f}" if i else str(int(v)) for i, v in enumerate(l)) for l in labs]
            lines.append(f"{D40} {(box[0] + box[2]) / 2 / w:.6f} {(box[1] + box[3]) / 2 / h:.6f} {(box[2] - box[0]) / w:.6f} {(box[3] - box[1]) / h:.6f}")
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
            n_acc += 1
            if len(sheet) < 48:
                sheet.append(np.hstack([cv2.resize(crop(img, box, 1.0), (160, 160)), cv2.resize(crop(res, box, 1.0), (160, 160))]))
        changed_out = (float((np.abs(last["res"].astype(int) - img.astype(int)).sum(-1) > 0)[~ellipse_mask(img.shape, last["box"], 1.6)].mean())
                       if "res" in last else None)
        rows.append({**info, "accepted": ok, "reason": last.get("reason", ""), "attempts": tried, "name": name if ok else "",
                     "boxes": json.dumps([[round(float(v), 1) for v in last["box"]]]) if "box" in last else "",
                     "donors": last.get("donor", ""), "clip": round(last.get("clip", float("nan")), 3),
                     "visible_change": round(last.get("vis", float("nan")), 1), "colour_shift": round(last.get("dab", float("nan")), 1),
                     "changed_outside_mask": changed_out, "mode": a.mode})
        if len(rows) % 25 == 0:  # checkpoint for resume
            man = pd.concat([man, pd.DataFrame(rows)], ignore_index=True)
            man.to_csv(man_p, index=False)
            rows = []
            print(f"{n_acc}/{a.n} accepted | {len(man)} tried | {(time.time() - t0) / 60:.1f} min")
    man = pd.concat([man, pd.DataFrame(rows)], ignore_index=True)
    man.to_csv(man_p, index=False)
    for i in range(0, len(sheet), 12):  # before | after pairs for manual audit
        cv2.imwrite(str(out / "audit" / f"sheet_{i // 12}.jpg"),
                    cv2.cvtColor(np.vstack(sheet[i:i + 12]), cv2.COLOR_RGB2BGR))
    summ = dict(mode=a.mode, accepted=int(man.accepted.sum()), tried=len(man),
                acceptance_rate=round(float(man.accepted.mean()), 3),
                by_domain=man[man.accepted].domain.value_counts().to_dict(),
                rejected=man[~man.accepted].reason.fillna("").value_counts().to_dict(), args=vars(a))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
