import argparse
import json
import torch

from data import NUM_CLASSES, IGNORE_INDEX, get_dataloaders
from model import DeepLabV3Plus

CLASSES = ["sky", "ridge", "soil", "sand", "bedrock", "rock", "rover", "trace", "hole"]

def confusion_matrix(pred, target, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    valid = target != ignore_index
    idx = target[valid] * num_classes + pred[valid]
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)

def compute_metrics(cm):
    cm = cm.double()
    tp = cm.diag()
    iou = tp / (cm.sum(dim=1) + cm.sum(dim=0) - tp)
    acc = tp / cm.sum(dim=1)
    return {
        "mIoU": 100 * iou.nanmean().item(),
        "mAcc": 100 * acc.nanmean().item(),
        "aAcc": 100 * (tp.sum() / cm.sum()).item(),
        "IoU": {c: 100 * v for c, v in zip(CLASSES, iou.tolist())},
        "Acc": {c: 100 * v for c, v in zip(CLASSES, acc.tolist())},
        "confusion_matrix": cm.long().tolist(),
    }

@torch.no_grad()
def evaluate(model, loader, device):
    was_training = model.training
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        cm += confusion_matrix(model(x).argmax(dim=1), y)
    model.train(was_training)
    return compute_metrics(cm.cpu())

def print_metrics(metrics):
    print(f"{'class':<10}{'IoU':>8}{'Acc':>8}")
    for c in CLASSES:
        print(f"{c:<10}{metrics['IoU'][c]:>8.2f}{metrics['Acc'][c]:>8.2f}")
    print(f"{'mean':<10}{metrics['mIoU']:>8.2f}{metrics['mAcc']:>8.2f}")
    print(f"pixel accuracy {metrics['aAcc']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    model = DeepLabV3Plus(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    _, val_loader, test_loader = get_dataloaders()
    metrics = evaluate(model, val_loader if args.split == "val" else test_loader, device)
    print_metrics(metrics)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(metrics, f, indent=2)
