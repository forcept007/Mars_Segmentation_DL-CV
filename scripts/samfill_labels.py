"""Write SAM gap-filled training labels (proposed method 3).

For each train image, every SAM segment votes with the ground-truth classes of its labeled pixels.
If one class has at least --rho of the votes (and the segment has at least --n_min labeled pixels),
the segment's unlabeled pixels get that class. Labeled pixels are never changed. Output PNGs use the
same raw format as labels/ (0 = unlabeled, 1-9 = classes), so data.load_label reads them as is.

python scripts/samfill_labels.py --root S5Mars_data --sam_root ~/data/sam_masks --rho 0.8
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

C = 9
MIN_IOU = 0.88   # same predicted_iou cut as SAM-Mix, but no area filter

def id_map(masks, h, w):
    # 0 = no segment. Smaller masks are drawn last so they win where masks overlap.
    out = np.zeros((h, w), dtype=np.int32)
    kept = sorted((m for m in masks if m["predicted_iou"] > MIN_IOU), key=lambda m: m["area"], reverse=True)
    for k, m in enumerate(kept, start=1):
        out[mask_utils.decode(m["segmentation"]).astype(bool)] = k
    return out, len(kept)

def fill(raw, ids, n, rho, n_min):
    # raw: (H,W) label PNG values, 0 = unlabeled, 1-9 = classes. Returns the filled raw label.
    lab = raw > 0
    counts = np.bincount(ids[lab] * C + (raw[lab] - 1), minlength=(n + 1) * C).reshape(n + 1, C)
    tot = counts.sum(1)
    ok = (tot >= n_min) & (counts.max(1) >= rho * np.maximum(tot, 1))
    ok[0] = False
    cls = counts.argmax(1) + 1
    gap = ~lab & ok[ids]
    out = raw.copy()
    out[gap] = cls[ids[gap]]
    return out

def process(args):
    img_id, root, sam_root, out_dir, rho, n_min = args
    raw = np.array(Image.open(os.path.join(root, "labels", img_id + ".png")))
    out_path = os.path.join(out_dir, img_id + ".png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sam_path = os.path.join(sam_root, img_id + ".json")
    if os.path.exists(sam_path):
        with open(sam_path) as f:
            ids, n = id_map(json.load(f)["masks"], *raw.shape)
        out = fill(raw, ids, n, rho, n_min) if n else raw
    else:
        out = raw   # no SAM masks for this image, keep the original label
    Image.fromarray(out.astype(np.uint8)).save(out_path)
    before, after = raw > 0, out > 0
    return (os.path.exists(sam_path), int(raw.size), int(before.sum()),
            np.bincount(out[after & ~before] - 1, minlength=C))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--sam_root", required=True)
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--n_min", type=int, default=20)
    p.add_argument("--out_name", default=None, help="folder under --root, default labels_samfill_rho<rho>")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    sam_root = os.path.expanduser(a.sam_root)
    out_name = a.out_name or f"labels_samfill_rho{a.rho:g}"
    out_dir = os.path.join(a.root, out_name)
    with open(os.path.join(a.root, "split", "train.json")) as f:
        ids = json.load(f)[:a.limit]

    t0 = time.time()
    n_sam, total, labeled, added = 0, 0, 0, np.zeros(C, np.int64)
    with Pool(a.workers) as pool:
        jobs = [(i, a.root, sam_root, out_dir, a.rho, a.n_min) for i in ids]
        for k, (has_sam, t, l, cls) in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            n_sam += has_sam; total += t; labeled += l; added += cls
            if k % 500 == 0:
                print(f"{k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)

    names = ["sky", "ridge", "soil", "sand", "bedrock", "rock", "rover", "trace", "hole"]
    stats = {"rho": a.rho, "n_min": a.n_min, "images": len(ids), "images_with_sam": n_sam,
             "labeled_before": labeled / total, "labeled_after": (labeled + int(added.sum())) / total,
             "gap_filled": int(added.sum()) / max(total - labeled, 1),
             "filled_class_share": dict(zip(names, (added / max(added.sum(), 1)).round(4).tolist()))}
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"wrote {out_dir} in {time.time() - t0:.0f}s")
