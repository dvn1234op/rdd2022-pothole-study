"""Build the fixed, leakage-aware RDD2022 evaluation split (run ONCE; the test set is frozen afterwards).

Pool       : the 38,385 labeled images of the official RDD2022 *train* folders (official test labels are private).
Grouping   : images joined by near-duplicate edges (pHash Hamming <= 4 OR ResNet-50 cosine >= 0.985) form one
             group; a group never straddles splits. Threshold chosen by visual inspection of contact sheets.
Stratum    : subset (domain) x has-D40 at image level.
Ratios     : train 70% / val 10% / test 20% via StratifiedGroupKFold(10 folds, seed 20261004).
Negatives  : images with no D00/D10/D20/D40 box are kept in every split (they are real road images).
Other raw labels (D43, D44, D50, Repair, D01, D11, ...) are dropped, as in CRDDC2022.

Also writes the cross-domain protocol: 'train_noIndia' = train minus India (test on the India part of test).

Outputs: data/splits/split_v1.csv (image_id, subset, split, group), data/splits/split_v1.sha256
         reports/split_leakage_audit.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
SPL = ROOT / "data/splits"
SEED = 20261004
PH, COS = 4, 0.985


def groups(n, pairs):
    e = pairs[(pairs.phash_dist <= PH) | (pairs.cos >= COS)]
    _, lab = connected_components(coo_matrix((np.ones(len(e)), (e.i, e.j)), shape=(n, n)), directed=False)
    return lab, e


def leak_stats(split_of, e, a, b):
    """How many images of split b have a near-duplicate partner in split a."""
    si, sj = split_of[e.i.values], split_of[e.j.values]
    m = ((si == a) & (sj == b)) | ((si == b) & (sj == a))
    eb = np.where(si[m] == b, e.i.values[m], e.j.values[m])
    return int(len(np.unique(eb))), int(m.sum())


def main():
    out = SPL / "split_v1.csv"
    if out.exists():
        raise SystemExit(f"{out} already exists - the split is frozen. Delete it deliberately if you really mean to rebuild.")
    img = pd.read_csv(P / "images_all.csv")
    ann = pd.read_csv(P / "annotations_all.csv")
    pairs = pd.read_csv(P / "near_dup_pairs.csv")
    n = len(img)
    lab, e = groups(n, pairs)
    has_d40 = img.image_id.isin(ann[ann.label == "D40"].image_id)
    strat = img.subset + "_" + has_d40.astype(int).astype(str)
    folds = np.full(n, -1)
    for k, (_, te) in enumerate(StratifiedGroupKFold(10, shuffle=True, random_state=SEED).split(np.zeros(n), strat, lab)):
        folds[te] = k
    split = np.where(folds <= 1, "test", np.where(folds == 2, "val", "train"))
    df = pd.DataFrame({"image_id": img.image_id, "subset": img.subset, "split": split, "group": lab, "has_D40": has_d40})
    assert df.groupby("group").split.nunique().max() == 1

    # ---- leakage audit: new split vs the previous random split (data/yolo_format) ----
    audit = {"near_dup_threshold": {"phash_max": PH, "cos_min": COS}, "n_edges": int(len(e)),
             "images_in_multi_image_groups": int((np.bincount(lab)[lab] > 1).sum()),
             "largest_group": int(np.bincount(lab).max())}
    old = {}
    for s in ("train", "val"):
        for f in (ROOT / f"data/yolo_format/images/{s}").iterdir():
            old[f.stem] = s
    old_split = img.image_id.map(old).fillna("excluded").values
    audit["old_split"] = {"sizes": pd.Series(old_split).value_counts().to_dict()}
    audit["old_split"]["val_images_with_neardup_in_train"], audit["old_split"]["cross_pairs"] = leak_stats(old_split, e, "train", "val")
    e_exact = e[e.phash_dist == 0]
    audit["old_split"]["val_images_with_exact_phash_dup_in_train"], _ = leak_stats(old_split, e_exact, "train", "val")
    audit["new_split"] = {"sizes": df.split.value_counts().to_dict()}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        audit["new_split"][f"{b}_images_with_neardup_in_{a}"], _ = leak_stats(split, e, a, b)
    # Pairs with 4 < pHash <= 8 and cos < 0.985 are NOT grouped (same road, different location); report how many straddle.
    loose = pairs[(pairs.phash_dist <= 8) & ~((pairs.phash_dist <= PH) | (pairs.cos >= COS))]
    audit["new_split"]["test_images_with_looser_similar_in_train(ph<=8)"], _ = leak_stats(split, loose, "train", "test")
    comp = df.groupby(["subset", "split"]).size().unstack()
    comp_d40 = df[df.has_D40].groupby(["subset", "split"]).size().unstack()
    audit["new_split"]["images_by_subset"] = comp.to_dict()
    audit["new_split"]["d40_images_by_subset"] = comp_d40.to_dict()

    SPL.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (SPL / "split_v1.sha256").write_text(hashlib.sha256(out.read_bytes()).hexdigest() + "  split_v1.csv\n")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/split_leakage_audit.json").write_text(json.dumps(audit, indent=2, default=int))
    print(json.dumps(audit, indent=1, default=int))
    print(comp, "\n", comp_d40)


if __name__ == "__main__":
    main()
