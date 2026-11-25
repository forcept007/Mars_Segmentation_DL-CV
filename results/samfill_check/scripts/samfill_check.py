"""Offline check for SAM-guided gap filling (beta = 0, ground truth only).

Hide square patches of labeled pixels, fill each SAM segment's hidden pixels with the
majority ground-truth class of its visible labeled pixels (if purity >= tau), and compare
with the hidden labels. Also reports how much of the real gaps would get filled.
"""
import argparse, json, os, sys, time, zlib
from multiprocessing import Pool
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import maximum_filter

C = 9
TAUS = [0.6, 0.7, 0.8, 0.9, 1.0]
FILTERS = {"sammix": (0.001, 0.10), "all": (0.0, 1.0)}   # area fraction range, same iou cut as SAM-Mix
MIN_IOU = 0.88

def id_map(masks, h, w, lo, hi):
    out = np.zeros((h, w), dtype=np.int32)
    kept = [m for m in masks if lo <= m["area"] / (h * w) <= hi and m["predicted_iou"] > MIN_IOU]
    kept.sort(key=lambda m: m["area"], reverse=True)   # smaller masks overwrite larger ones
    for k, m in enumerate(kept, start=1):
        out[mask_utils.decode(m["segmentation"]).astype(bool)] = k
    return out, len(kept)

def hidden_mask(h, w, patch, frac, seed):
    rng = np.random.default_rng(seed)
    gh, gw = -(-h // patch), -(-w // patch)
    cells = rng.random((gh, gw)) < frac
    return np.kron(cells, np.ones((patch, patch), dtype=bool))[:h, :w]

def boundary_mask(y, d):
    # labeled pixels within d px of a different class or of an unlabeled pixel (where real gaps tend to be)
    edge = np.zeros(y.shape, dtype=bool)
    edge[:, 1:] |= y[:, 1:] != y[:, :-1]
    edge[:, :-1] |= y[:, 1:] != y[:, :-1]
    edge[1:] |= y[1:] != y[:-1]
    edge[:-1] |= y[1:] != y[:-1]
    return maximum_filter(edge, size=2 * d + 1)

def process(args):
    img_id, root, sam_root, patch, frac, n_min, hide = args
    raw = np.array(Image.open(os.path.join(root, "labels", img_id + ".png"))).astype(np.int64)
    y = np.where(raw == 0, 255, raw - 1)
    h, w = y.shape
    with open(os.path.join(sam_root, img_id + ".json")) as f:
        masks = json.load(f)["masks"]
    if hide == "patch":
        hid = hidden_mask(h, w, patch, frac, zlib.crc32(img_id.encode())) & (y != 255)
    else:
        hid = boundary_mask(y, patch) & (y != 255)
    vis = (y != 255) & ~hid
    gap = y == 255
    res = {}
    for name, (lo, hi) in FILTERS.items():
        ids, n = id_map(masks, h, w, lo, hi)
        if n == 0:
            continue
        counts = np.bincount(ids[vis] * C + y[vis], minlength=(n + 1) * C).reshape(n + 1, C)
        tot = counts.sum(1)
        purity = counts.max(1) / np.maximum(tot, 1)
        cls = counts.argmax(1)
        ok_base = (tot >= n_min)
        ok_base[0] = False   # pixels outside every segment are never filled
        r = {"hidden_total": int(hid.sum()), "gap_total": int(gap.sum()), "gap_in_seg": int((gap & (ids > 0)).sum())}
        for tau in TAUS:
            ok = ok_base & (purity >= tau)
            fh = hid & ok[ids]
            conf = np.bincount(y[fh] * C + cls[ids[fh]], minlength=C * C).reshape(C, C)
            fg = gap & ok[ids]
            r[str(tau)] = {"conf": conf.tolist(), "gap_filled": int(fg.sum()),
                           "gap_fill_cls": np.bincount(cls[ids[fg]], minlength=C).tolist()}
        res[name] = r
    return res

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--sam_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--patch", type=int, default=48)
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--n_min", type=int, default=20)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--hide", choices=["patch", "boundary"], default="patch")
    a = p.parse_args()
    with open(os.path.join(a.root, "split", "train.json")) as f:
        ids = json.load(f)[:a.limit]
    ids = [i for i in ids if os.path.exists(os.path.join(a.sam_root, i + ".json"))]
    print(f"{len(ids)} images with SAM masks", flush=True)
    agg = {}
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(process, [(i, a.root, a.sam_root, a.patch, a.frac, a.n_min, a.hide) for i in ids], chunksize=4), 1):
            for name, r in res.items():
                g = agg.setdefault(name, {"hidden_total": 0, "gap_total": 0, "gap_in_seg": 0, "images": 0})
                g["images"] += 1
                for key in ("hidden_total", "gap_total", "gap_in_seg"):
                    g[key] += r[key]
                for tau in TAUS:
                    t = g.setdefault(str(tau), {"conf": np.zeros((C, C), np.int64), "gap_filled": 0, "gap_fill_cls": np.zeros(C, np.int64)})
                    t["conf"] += np.array(r[str(tau)]["conf"])
                    t["gap_filled"] += r[str(tau)]["gap_filled"]
                    t["gap_fill_cls"] += np.array(r[str(tau)]["gap_fill_cls"])
            if k % 100 == 0:
                print(f"{k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    for g in agg.values():
        for tau in TAUS:
            g[str(tau)]["conf"] = g[str(tau)]["conf"].tolist()
            g[str(tau)]["gap_fill_cls"] = g[str(tau)]["gap_fill_cls"].tolist()
    with open(a.out, "w") as f:
        json.dump(agg, f)
    print("done", a.out, f"{time.time() - t0:.0f}s")
