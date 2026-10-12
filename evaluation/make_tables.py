"""Aggregate results/<run>/metrics.json into the markdown tables used by RESULTS.md.

    python evaluation/make_tables.py            -> results/tables.md  (+ results/master_table.csv)

Seed groups (<prefix>_s0/_s1/_s2 or A?_s?) are reported as mean ± sample std when >1 seed exists; the number of
seeds is shown in the table. All values are TEST-split values in percent.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C = ["D00", "D10", "D20", "D40"]
DOMS = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]


def load_all():
    runs = {}
    for f in sorted((ROOT / "results").glob("*/metrics.json")):
        if "codecheck" in f.parent.name:
            continue
        m = json.loads(f.read_text())
        if "test" in m:
            runs[f.parent.name] = m
    return runs


def group(runs):
    g = defaultdict(list)
    for name, m in runs.items():
        g[re.sub(r"_s\d+$", "", name)].append(m)
    return g


def fmt(vals, scale=100, nd=1):
    v = np.array([x for x in vals if x is not None], float) * scale
    if len(v) == 0:
        return "—"
    return f"{v.mean():.{nd}f}" if len(v) == 1 else f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}"


def row(ms, fn, **k):
    return fmt([fn(m) for m in ms], **k)


def main():
    runs = load_all()
    G = group(runs)
    out = []
    out.append("### Master table (test split, %, mean ± std over seeds where n>1)\n")
    out.append("| Model | n seeds | D00 AP50 | D10 AP50 | D20 AP50 | D40 AP50 | D40 AP50:95 | mAP50 | mAP50:95 | mAP50 95% CI (seed 0) | Params (M) | GFLOPs | FPS bs1 fp16 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|")
    csv = ["model,n,D00,D10,D20,D40,D40_5095,mAP50,mAP5095,params_M,GFLOPs,fps"]
    for g, ms in G.items():
        t = lambda m: m["test"]
        ci = ms[0]["test"].get("bootstrap95_AP50", {}).get("mAP50")
        cx = ms[0].get("complexity", {})
        cells = [row(ms, lambda m, c=c: t(m)["per_class"][c]["AP50"]) for c in C]
        out.append(f"| {g} | {len(ms)} | " + " | ".join(cells) + f" | {row(ms, lambda m: t(m)['per_class']['D40']['AP50_95'])}"
                   f" | {row(ms, lambda m: t(m)['mAP50'])} | {row(ms, lambda m: t(m)['mAP50_95'])}"
                   f" | {f'[{100*ci[0]:.1f}, {100*ci[1]:.1f}]' if ci else '—'}"
                   f" | {cx.get('params_M', float('nan')):.2f} | {cx.get('GFLOPs_640') or float('nan'):.1f} | {cx.get('fps_bs1_fp16_forward_only', float('nan')):.0f} |")
        csv.append(",".join([g, str(len(ms))] + [f"{np.mean([t(m)['per_class'][c]['AP50'] for m in ms]):.4f}" for c in C]
                            + [f"{np.mean([t(m)['per_class']['D40']['AP50_95'] for m in ms]):.4f}", f"{np.mean([t(m)['mAP50'] for m in ms]):.4f}",
                               f"{np.mean([t(m)['mAP50_95'] for m in ms]):.4f}", f"{cx.get('params_M', 0):.3f}", f"{cx.get('GFLOPs_640') or 0:.2f}",
                               f"{cx.get('fps_bs1_fp16_forward_only', 0):.1f}"]))

    out.append("\n### Size-stratified AP50 (test, 640-px input scale; small<32², medium<96², large≥96²)\n")
    out.append("| Model | Small AP50 | Medium AP50 | Large AP50 | Small AP50:95 | D40 small AP50 | D40 medium AP50 | D40 large AP50 |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for g, ms in G.items():
        t = lambda m: m["test"]
        out.append(f"| {g} | " + " | ".join(row(ms, lambda m, s=s: t(m)["size_AP50"][s]) for s in ("small", "medium", "large"))
                   + f" | {row(ms, lambda m: t(m)['size_AP50_95']['small'])} | "
                   + " | ".join(row(ms, lambda m, s=s: t(m)["per_class"]["D40"][f"AP50_{s}"]) for s in ("small", "medium", "large")) + " |")

    out.append("\n### Per-domain mAP50 (test; all 4 classes)\n")
    out.append("| Model | " + " | ".join(DOMS) + " |")
    out.append("|---|" + "---:|" * len(DOMS))
    for g, ms in G.items():
        out.append(f"| {g} | " + " | ".join(row(ms, lambda m, d=d: m["test"]["per_domain"][d]["mAP50"]) for d in DOMS) + " |")
    out.append("\n### Per-domain D40 AP50 (test) — n D40 test boxes in header\n")
    any_m = next(iter(runs.values()))
    hdr = [f"{d} (n={any_m['test']['per_domain'][d]['n_gt'].get('3', any_m['test']['per_domain'][d]['n_gt'].get(3, 0))})" for d in DOMS]
    out.append("| Model | " + " | ".join(hdr) + " |")
    out.append("|---|" + "---:|" * len(DOMS))
    for g, ms in G.items():
        out.append(f"| {g} | " + " | ".join(row(ms, lambda m, d=d: m["test"]["per_domain"][d]["D40_AP50"]) for d in DOMS) + " |")

    out.append("\n### Precision / recall / F1 (test, IoU 0.5, per-class thresholds chosen on val)\n")
    out.append("| Model | " + " | ".join(f"{c} P / R / F1" for c in C) + " | macro F1 |")
    out.append("|---|" + "---|" * len(C) + "---:|")
    for g, ms in G.items():
        p = lambda m: m["test"]["PRF1_at_val_thresholds"]
        cells = [f"{row(ms, lambda m, c=c: p(m)[c]['precision'])} / {row(ms, lambda m, c=c: p(m)[c]['recall'])} / {row(ms, lambda m, c=c: p(m)[c]['F1'])}" for c in C]
        out.append(f"| {g} | " + " | ".join(cells) + f" | {row(ms, lambda m: p(m)['macro_F1'])} |")

    out.append("\n### Bootstrap 95% CIs, seed-0 run (test, AP50 %)\n")
    out.append("| Run | " + " | ".join(C) + " | mAP50 |")
    out.append("|---|" + "---|" * 5)
    for name, m in runs.items():
        b = m["test"].get("bootstrap95_AP50")
        if b and name.endswith("_s0"):
            out.append(f"| {name} | " + " | ".join(f"[{100*b[c][0]:.1f}, {100*b[c][1]:.1f}]" for c in C + ["mAP50"]) + " |")
    (ROOT / "results/tables.md").write_text("\n".join(out) + "\n")
    (ROOT / "results/master_table.csv").write_text("\n".join(csv) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
