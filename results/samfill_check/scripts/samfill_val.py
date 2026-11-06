"""Compare SAM gap filling with the SSL teacher on held-out boundary pixels of the val set.

The teacher never trained on val labels, so this is a fair comparison. For each val image:
hide labeled pixels within d px of a class boundary, then on those pixels compare
  teacher  : argmax of the teacher prediction
  sam_b0   : SAM segment vote from the visible labels only (beta = 0)
  sam_b1   : SAM segment vote from visible labels + teacher probabilities (beta = 1)
  hybrid_* : SAM vote where the segment passes tau, teacher everywhere else
Run from the repo root so models.py can be imported.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import maximum_filter

sys.path.insert(0, os.getcwd())
from models import build_model

C = 9
TAUS = [0.6, 0.7, 0.8, 0.9, 1.0]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
MIN_IOU = 0.88
CROP = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def center(a):
    h, w = a.shape[:2]
    t, l = (h - CROP) // 2, (w - CROP) // 2
    return a[t:t + CROP, l:l + CROP]

def boundary_mask(y, d):
    edge = np.zeros(y.shape, dtype=bool)
    edge[:, 1:] |= y[:, 1:] != y[:, :-1]
    edge[:, :-1] |= y[:, 1:] != y[:, :-1]
    edge[1:] |= y[1:] != y[:-1]
    edge[:-1] |= y[1:] != y[:-1]
    return maximum_filter(edge, size=2 * d + 1)

def id_map(masks, h, w):
    out = np.zeros((h, w), dtype=np.int32)
    kept = [m for m in masks if m["predicted_iou"] > MIN_IOU]
    kept.sort(key=lambda m: m["area"], reverse=True)
    for k, m in enumerate(kept, start=1):
        out[mask_utils.decode(m["segmentation"]).astype(bool)] = k
    return out, len(kept)

def conf(t, p):
    return np.bincount(t * C + p, minlength=C * C).reshape(C, C)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--sam_out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n_min", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    dev = "cuda"
    with open(os.path.join(a.root, "split", "val.json")) as f:
        ids = json.load(f)[:a.limit]

    # 1. SAM masks for val (same generator settings as scripts/precompute_sam_masks.py)
    todo = [i for i in ids if not os.path.exists(os.path.join(a.sam_out, i + ".json"))]
    if todo:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry["vit_h"](checkpoint=a.sam_ckpt).to(dev)
        gen = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")
        t0 = time.time()
        for k, i in enumerate(todo, 1):
            img = np.array(Image.open(os.path.join(a.root, "images", i + ".jpg")).convert("RGB"))
            path = os.path.join(a.sam_out, i + ".json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"id": i, "model": "vit_h", "masks": gen.generate(img)}, f)
            if k % 20 == 0:
                print(f"sam {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        del sam, gen
        torch.cuda.empty_cache()

    # 2. Teacher
    ckpt = torch.load(a.teacher, map_location=dev)
    model = build_model({**ckpt.get("model_cfg", {}), "pretrained": False}).to(dev).eval()
    model.load_state_dict(ckpt["model"])

    keys = ["teacher_all"] + [f"{m}_b{b}_{t}" for b in BETAS for t in TAUS for m in ("sam", "teacher_same", "hybrid")]
    cms = {k: np.zeros((C, C), np.int64) for k in keys}
    hidden_total = 0
    t0 = time.time()
    for k, i in enumerate(ids, 1):
        img = np.array(Image.open(os.path.join(a.root, "images", i + ".jpg")).convert("RGB"))
        raw = np.array(Image.open(os.path.join(a.root, "labels", i + ".png"))).astype(np.int64)
        y = center(np.where(raw == 0, 255, raw - 1))
        x = (center(img).astype(np.float32) / 255 - MEAN) / STD
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(torch.from_numpy(x).permute(2, 0, 1)[None].to(dev))
        prob = F.softmax(logits.float(), dim=1)[0].permute(1, 2, 0).cpu().numpy()   # (H, W, C)
        tpred = prob.argmax(-1)

        with open(os.path.join(a.sam_out, i + ".json")) as f:
            masks = json.load(f)["masks"]
        h, w = raw.shape
        ids_full, n = id_map(masks, h, w)
        seg = center(ids_full)

        hid = boundary_mask(y, a.d) & (y != 255)
        vis = (y != 255) & ~hid
        hidden_total += int(hid.sum())
        cms["teacher_all"] += conf(y[hid], tpred[hid])

        counts = np.bincount(seg[vis] * C + y[vis], minlength=(n + 1) * C).reshape(n + 1, C).astype(np.float64)
        nvis = counts.sum(1)
        psum = np.zeros((n + 1, C))
        np.add.at(psum, seg.ravel(), prob.reshape(-1, C))
        size = np.bincount(seg.ravel(), minlength=n + 1).astype(np.float64)
        for b in BETAS:
            # q_S = (1 - beta) * labeled-pixel histogram + beta * mean teacher prob over the segment
            q = (1 - b) * counts / np.maximum(nvis, 1)[:, None] + b * psum / np.maximum(size, 1)[:, None]
            score, cls = q.max(1), q.argmax(1)
            base = (nvis >= a.n_min) if b < 1 else (size >= a.n_min)
            base[0] = False
            for t in TAUS:
                ok = base & (score >= t - 1e-9)
                fh = hid & ok[seg]
                cms[f"sam_b{b}_{t}"] += conf(y[fh], cls[seg[fh]])
                cms[f"teacher_same_b{b}_{t}"] += conf(y[fh], tpred[fh])
                hyb = np.where(ok[seg], cls[seg], tpred)
                cms[f"hybrid_b{b}_{t}"] += conf(y[hid], hyb[hid])
        if k % 20 == 0:
            print(f"eval {k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)

    json.dump({"hidden_total": hidden_total, "images": len(ids), "cms": {k: v.tolist() for k, v in cms.items()}}, open(a.out, "w"))
    print("done", a.out)
