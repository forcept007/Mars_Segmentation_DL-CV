import os
import sys
import argparse
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data import CLASSES, NUM_CLASSES, get_dataloaders, norm_stats
from evaluate import confusion_matrix, compute_metrics
from models import build_model

# Add a constant to the rock logit before argmax. If rock IoU goes up, the model is too cautious about rock.
BIASES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
ROCK = CLASSES.index("rock")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model({**ckpt.get("model_cfg", {}), "pretrained": False}).to(device).eval()
    model.load_state_dict(ckpt["model"])
    _, val_loader, test_loader = get_dataloaders(**norm_stats(ckpt.get("data_cfg", {})))
    loader = val_loader if args.split == "val" else test_loader

    cms = {b: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device) for b in BIASES}
    with torch.no_grad():
        for x, y, _ in loader:
            logits = model(x.to(device))
            y = y.to(device)
            for b in BIASES:
                shifted = logits.clone()
                shifted[:, ROCK] += b
                cms[b] += confusion_matrix(shifted.argmax(dim=1), y)

    print(f"{'bias':>5}{'mIoU':>8}{'rock IoU':>10}{'precision':>11}{'recall':>8}")
    for b, cm in cms.items():
        m = compute_metrics(cm.cpu())
        print(f"{b:>5}{m['mIoU']:>8.2f}{m['IoU']['rock']:>10.2f}{m['rock']['precision']:>11.2f}{m['rock']['recall']:>8.2f}")
