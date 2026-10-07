import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data import CLASSES, NUM_CLASSES, IGNORE_INDEX, S5MarsDataset, train_transform
from check_labels import LabelCounts

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#edc948", "#59a14f", "#b07aa1", "#ff9da7", "#9c755f", "#76b7b2"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def most_rock_pixels(split="train"):
    loader = DataLoader(LabelCounts(split), batch_size=16, num_workers=4)
    rock = torch.cat([counts[:, CLASSES.index("rock")] for counts in loader])
    return rock.argmax().item()

def overlay(ax, image, label, title):
    ax.imshow(image)
    ax.imshow(np.ma.masked_equal(label, IGNORE_INDEX), cmap=ListedColormap(PALETTE),
              vmin=0, vmax=NUM_CLASSES - 1, alpha=0.5, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")

if __name__ == "__main__":
    torch.manual_seed(0)
    full = S5MarsDataset("train")
    augmented = S5MarsDataset("train", train_transform())
    picks = torch.randperm(len(full))[:7].tolist() + [most_rock_pixels()]

    fig, axes = plt.subplots(4, 4, figsize=(16, 17))
    for k, idx in enumerate(picks):
        image, label, name = full[idx]
        overlay(axes.flat[2 * k], image.permute(1, 2, 0).numpy(), label.numpy(),
                name + ("  (most rock pixels)" if k == len(picks) - 1 else ""))
        x, y, _ = augmented[idx]
        overlay(axes.flat[2 * k + 1], (x * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy(), y.numpy(),
                "training view (scale, crop, flip)")
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(PALETTE, CLASSES)],
               loc="lower center", ncol=NUM_CLASSES, fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    os.makedirs(os.path.join(REPO, "figures"), exist_ok=True)
    fig.savefig(os.path.join(REPO, "figures", "batch_overlay.jpg"), dpi=80)
    print("saved figures/batch_overlay.jpg")
