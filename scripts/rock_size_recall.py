"""MS-64: rock recall by ground-truth rock instance size.

Each connected component (8-connectivity) of the ground-truth "rock" mask is one instance. For every model we
measure, per size bin (area in pixels at evaluation resolution):
  recall        = rock pixels predicted as rock / all rock pixels in the bin (pooled over instances)
  rock->bedrock = rock pixels predicted as bedrock / all rock pixels in the bin
  detected      = share of instances where at least half of the pixels are predicted as rock

Usage (from the repo root):
    python scripts/rock_size_recall.py \
        --models plain=runs/dinov2_vits14_short/best.pth detail=runs/dinov2_vits14_detail_short/best.pth \
        --split test --out-dir results/ms64_size_recall
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import ndimage

from data import CLASSES

ROCK, BEDROCK = CLASSES.index("rock"), CLASSES.index("bedrock")
STRUCT = np.ones((3, 3), dtype=int)  # 8-connectivity


def rock_components(gt, pred):
    """One (area, pixels predicted rock, pixels predicted bedrock) tuple per connected ground-truth rock component."""
    labels, n = ndimage.label(gt == ROCK, structure=STRUCT)  # 255 (ignore) is never rock, so it is excluded
    if n == 0:
        return []
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    hit = np.bincount(labels[pred == ROCK], minlength=n + 1)[1:]
    bed = np.bincount(labels[pred == BEDROCK], minlength=n + 1)[1:]
    return list(zip(area.tolist(), hit.tolist(), bed.tolist()))


def bin_edges(edges):
    bounds = [1] + list(edges) + [float("inf")]
    return list(zip(bounds[:-1], bounds[1:]))


def bin_label(lo, hi):
    return f"{lo}+" if hi == float("inf") else f"{lo}-{hi - 1}"


def summarize(rows, edges):
    """rows: list of (image id, area, hit, bed). Returns one dict per size bin."""
    area = np.array([r[1] for r in rows], dtype=np.int64)
    hit = np.array([r[2] for r in rows], dtype=np.int64)
    bed = np.array([r[3] for r in rows], dtype=np.int64)
    out = []
    for lo, hi in bin_edges(edges):
        sel = (area >= lo) & (area < hi)
        n = int(sel.sum())
        total = int(area[sel].sum())
        out.append({
            "bin": bin_label(lo, hi),
            "instances": n,
            "rock_pixels": total,
            "recall": 100 * hit[sel].sum() / total if total else float("nan"),
            "rock_to_bedrock": 100 * bed[sel].sum() / total if total else float("nan"),
            "detected": 100 * float(np.mean(hit[sel] / area[sel] >= 0.5)) if n else float("nan"),
        })
    return out


def collect(path, loader, device):
    import torch
    from models import build_model

    ckpt = torch.load(path, map_location=device)
    model = build_model({**ckpt.get("model_cfg", {}), "pretrained": False}).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    rows = []
    with torch.no_grad():
        for x, y, ids in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
            for p, t, sample_id in zip(pred, y.numpy(), ids):
                rows += [(sample_id, *c) for c in rock_components(t, p)]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--edges", type=int, nargs="+", default=[100, 500, 2000, 10000],
                        help="bin edges in pixels; default gives 1-99, 100-499, 500-1999, 2000-9999, 10000+")
    parser.add_argument("--out-dir", default="results/ms64_size_recall")
    args = parser.parse_args()

    import torch
    from data import get_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    _, val_loader, test_loader = get_dataloaders()
    loader = val_loader if args.split == "val" else test_loader

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = {}
    for item in args.models:
        name, path = item.split("=", 1)
        rows = collect(path, loader, device)
        with open(os.path.join(args.out_dir, f"instances_{name}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "area", "pixels_pred_rock", "pixels_pred_bedrock"])
            writer.writerows(rows)
        summaries[name] = summarize(rows, args.edges)
        areas = np.array([r[1] for r in rows])
        print(f"\n{name}: {len(rows)} rock instances, area percentiles 10/25/50/75/90 = "
              f"{np.percentile(areas, [10, 25, 50, 75, 90]).astype(int).tolist()}")
        print(f"{'bin':<12}{'inst':>7}{'recall':>9}{'->bedrock':>11}{'detected':>10}")
        for s in summaries[name]:
            print(f"{s['bin']:<12}{s['instances']:>7}{s['recall']:>9.1f}{s['rock_to_bedrock']:>11.1f}{s['detected']:>10.1f}")

    with open(os.path.join(args.out_dir, "rock_size_recall.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "bin", "instances", "rock_pixels", "recall", "rock_to_bedrock", "detected"])
        for name, rows in summaries.items():
            for s in rows:
                writer.writerow([name, s["bin"], s["instances"], s["rock_pixels"], round(s["recall"], 2),
                                 round(s["rock_to_bedrock"], 2), round(s["detected"], 2)])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = list(summaries)
        bins = [s["bin"] for s in summaries[names[0]]]
        counts = [s["instances"] for s in summaries[names[0]]]
        width = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for i, name in enumerate(names):
            ax.bar(np.arange(len(bins)) + i * width, [s["recall"] for s in summaries[name]], width, label=name)
        ax.set_xticks(np.arange(len(bins)) + width * (len(names) - 1) / 2)
        ax.set_xticklabels([f"{b}\n(n={c})" for b, c in zip(bins, counts)])
        ax.set_xlabel("ground-truth rock instance size (pixels)")
        ax.set_ylabel("rock recall (%)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "rock_size_recall.png"), dpi=200)
    except Exception as e:  # the CSVs are the result; the figure is a convenience
        print(f"figure skipped: {e}")
    print(f"\nsaved to {args.out_dir}")


if __name__ == "__main__":
    main()
