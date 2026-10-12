"""Near-duplicate / same-scene detection over all 38,385 labeled RDD2022 images.

Two independent signals:
  1. 64-bit perceptual hash (pHash) -> Hamming distance
  2. ImageNet ResNet-50 global-average-pooled embedding -> cosine similarity (GPU)

Output: data/processed/near_dup_pairs.csv  (pairs with phash_dist <= 10 OR cos >= 0.95)
        data/processed/embeddings_r50.npy, data/processed/phash.npy (cached)
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
PHASH_MAX, COS_MIN = 10, 0.95


def phash_one(path):
    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("L"))), 16)


class DS(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def main():
    img = pd.read_csv(P / "images_all.csv")
    paths = img.path.tolist()
    if not (P / "phash.npy").exists():
        with ProcessPoolExecutor(os.cpu_count()) as ex:
            h = list(ex.map(phash_one, paths, chunksize=128))
        np.save(P / "phash.npy", np.array(h, dtype=np.uint64))
    ph = np.load(P / "phash.npy")
    if not (P / "embeddings_r50.npy").exists():
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        m.fc = torch.nn.Identity()
        m = m.cuda().eval().half()
        dl = torch.utils.data.DataLoader(DS(paths), batch_size=128, num_workers=12)
        out = []
        with torch.no_grad():
            for x in dl:
                out.append(torch.nn.functional.normalize(m(x.cuda().half()).float(), dim=1).cpu())
        np.save(P / "embeddings_r50.npy", torch.cat(out).numpy().astype(np.float16))
    E = torch.from_numpy(np.load(P / "embeddings_r50.npy").astype(np.float32)).cuda()

    # pHash Hamming distance, all pairs, on GPU via bit unpacking
    bits = torch.from_numpy(np.unpackbits(ph.view(np.uint8).reshape(-1, 8), axis=1).astype(np.float16)).cuda()
    n = len(paths)
    rows = []
    B = 2048
    for s in range(0, n, B):
        b = bits[s:s + B]
        ham = (b @ (1 - bits).T + (1 - b) @ bits.T).float()      # Hamming distance
        cos = E[s:s + B] @ E.T
        mask = (ham <= PHASH_MAX) | (cos >= COS_MIN)
        idx = torch.nonzero(mask)
        idx = idx[idx[:, 0] + s < idx[:, 1]]                     # i < j, no self pairs
        for i, j in idx.tolist():
            rows.append((s + i, j, int(ham[i, j]), float(cos[i, j])))
    df = pd.DataFrame(rows, columns=["i", "j", "phash_dist", "cos"])
    df["id_i"], df["id_j"] = img.image_id.values[df.i], img.image_id.values[df.j]
    df["subset_i"], df["subset_j"] = img.subset.values[df.i], img.subset.values[df.j]
    df.to_csv(P / "near_dup_pairs.csv", index=False)
    print(f"pairs: {len(df)}  exact pHash: {(df.phash_dist == 0).sum()}  phash<=6: {(df.phash_dist <= 6).sum()}  cos>=0.97: {(df.cos >= 0.97).sum()}")

    # how similar are neighbouring frame IDs (sequence structure) vs random pairs, per subset?
    rng = np.random.default_rng(0)
    print("\nconsecutive-ID vs random-pair similarity (median cosine, % pairs with cos>=0.9):")
    for sub, g in img.reset_index().groupby("subset"):
        g = g.sort_values("image_id")
        a, b = g["index"].values[:-1], g["index"].values[1:]
        c_adj = (E[a] * E[b]).sum(1).cpu().numpy()
        r = rng.choice(g["index"].values, size=(len(a), 2))
        c_rnd = (E[r[:, 0]] * E[r[:, 1]]).sum(1).cpu().numpy()
        print(f"  {sub:16s} adj median {np.median(c_adj):.3f} (>=0.9: {100*(c_adj>=0.9).mean():5.1f}%)  random median {np.median(c_rnd):.3f} (>=0.9: {100*(c_rnd>=0.9).mean():5.1f}%)")


if __name__ == "__main__":
    main()
