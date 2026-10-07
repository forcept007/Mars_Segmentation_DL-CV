import argparse
import csv
import json
import os
import numpy as np
import torch

from data import CLASSES, NUM_CLASSES, IGNORE_INDEX, get_dataloaders
from models import build_model

def confusion_matrix(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    valid = target != ignore_index
    idx = target[valid] * num_classes + pred[valid]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)

def percent(num, den):
    return float(100 * num / den) if den else float("nan")

def rock_metrics(cm, class_names=CLASSES):
    rock, bedrock = class_names.index("rock"), class_names.index("bedrock")
    tp = cm[rock, rock]
    fn = cm[rock].sum() - tp
    fp = cm[:, rock].sum() - tp
    precision, recall = percent(tp, tp + fp), percent(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rock_to_bedrock": percent(cm[rock, bedrock], cm[rock].sum()),
        "bedrock_to_rock": percent(cm[bedrock, rock], cm[bedrock].sum()),
        "fn_predicted_bedrock": percent(cm[rock, bedrock], fn),
        "fp_from_bedrock": percent(cm[bedrock, rock], fp),
    }

def compute_metrics(cm, class_names=CLASSES):
    counts = cm.double()
    tp = counts.diag()
    iou = tp / (counts.sum(dim=1) + counts.sum(dim=0) - tp)
    acc = tp / counts.sum(dim=1)
    metrics = {
        "mIoU": 100 * iou.nanmean().item(),
        "mAcc": 100 * acc.nanmean().item(),
        "aAcc": 100 * (tp.sum() / counts.sum()).item(),
        "IoU": {c: 100 * v for c, v in zip(class_names, iou.tolist())},
        "Acc": {c: 100 * v for c, v in zip(class_names, acc.tolist())},
        "confusion_matrix": cm.long().tolist(),
    }
    if "rock" in class_names and "bedrock" in class_names:
        metrics["rock"] = rock_metrics(cm.long().numpy(), class_names)
    return metrics

@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    model.eval()
    cms = {}
    for x, y, ids in loader:
        pred = model(x.to(device)).argmax(dim=1)
        y = y.to(device)
        for p, t, sample_id in zip(pred, y, ids):
            subset = sample_id.split("/")[0]
            if subset not in cms:
                cms[subset] = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
            cms[subset] += confusion_matrix(p, t)
    model.train(was_training)
    metrics = compute_metrics(sum(cms.values()).cpu())
    metrics["subsets"] = {name: compute_metrics(cm.cpu()) for name, cm in sorted(cms.items())}
    return metrics

def print_metrics(metrics):
    subsets = list(metrics.get("subsets", {}))
    print(f"{'class':<10}{'IoU':>8}{'Acc':>8}" + "".join(f"{'IoU ' + s:>10}" for s in subsets))
    for c in metrics["IoU"]:
        print(f"{c:<10}{metrics['IoU'][c]:>8.2f}{metrics['Acc'][c]:>8.2f}"
              + "".join(f"{metrics['subsets'][s]['IoU'][c]:>10.2f}" for s in subsets))
    print(f"{'mean':<10}{metrics['mIoU']:>8.2f}{metrics['mAcc']:>8.2f}"
          + "".join(f"{metrics['subsets'][s]['mIoU']:>10.2f}" for s in subsets))
    print(f"pixel accuracy {metrics['aAcc']:.2f}")
    if "rock" in metrics:
        print("rock: " + ", ".join(f"{k} {v:.2f}" for k, v in metrics["rock"].items()))

def save_results(metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(out_dir, "confusion.npy"), np.array(metrics["confusion_matrix"]))
    subsets = list(metrics.get("subsets", {}))
    with open(os.path.join(out_dir, "per_class.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "IoU", "Acc"] + [f"{m}_{s}" for s in subsets for m in ("IoU", "Acc")])
        for c in metrics["IoU"]:
            writer.writerow([c, round(metrics["IoU"][c], 2), round(metrics["Acc"][c], 2)]
                            + [round(metrics["subsets"][s][m][c], 2) for s in subsets for m in ("IoU", "Acc")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model({**ckpt.get("model_cfg", {}), "pretrained": False}).to(device)
    model.load_state_dict(ckpt["model"])
    _, val_loader, test_loader = get_dataloaders()
    metrics = evaluate(model, val_loader if args.split == "val" else test_loader, device)
    print_metrics(metrics)

    run = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
    out_dir = args.out_dir or os.path.join("results", run, args.split)
    save_results(metrics, out_dir)
    print(f"saved {out_dir}/metrics.json, per_class.csv, confusion.npy")
