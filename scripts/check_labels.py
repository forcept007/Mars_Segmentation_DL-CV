import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data import DATA_ROOT, CLASSES, NUM_CLASSES, IGNORE_INDEX, load_label

SPLITS = ["train", "val", "test"]

class LabelCounts(Dataset):
    def __init__(self, split, root=DATA_ROOT):
        with open(os.path.join(root, "split", split + ".json")) as f:
            self.ids = json.load(f)
        self.root = root

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        label = load_label(os.path.join(self.root, "labels", self.ids[idx] + ".png"))
        return torch.bincount(label.flatten(), minlength=256)

def count_pixels(split, num_workers=4):
    loader = DataLoader(LabelCounts(split), batch_size=16, num_workers=num_workers)
    return sum(counts.sum(dim=0) for counts in loader)

def check_split(split, counts):
    labeled = counts[:NUM_CLASSES].sum().item()
    labeled_pct = 100 * labeled / counts.sum().item()
    shares = 100 * counts[:NUM_CLASSES].double() / labeled
    unexpected = counts.sum().item() - labeled - counts[IGNORE_INDEX].item()

    print(f"\n{split}: {labeled_pct:.2f}% of pixels labeled")
    for name, share in zip(CLASSES, shares.tolist()):
        print(f"  {name:<8}{share:6.2f}%")

    assert unexpected == 0, f"{split}: {unexpected} pixels outside 0-8 and {IGNORE_INDEX}"
    assert 46 <= labeled_pct <= 52, f"{split}: labeled fraction {labeled_pct:.2f}% is not near 49%"
    assert CLASSES[shares.argmax()] == "bedrock", f"{split}: largest class is {CLASSES[shares.argmax()]}, not bedrock"
    rock = shares[CLASSES.index("rock")].item()
    assert 2 <= rock <= 5, f"{split}: rock is {rock:.2f}% of labeled pixels, expected about 3%"
    return shares.numpy()

def plot_shares(shares, path):
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(shares)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (split, values) in enumerate(shares.items()):
        ax.bar(x + (i - (len(shares) - 1) / 2) * width, values, width, label=split)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("% of labeled pixels")
    ax.set_title("S5Mars class distribution after remapping")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)

if __name__ == "__main__":
    shares = {split: check_split(split, count_pixels(split)) for split in SPLITS}
    os.makedirs(os.path.join(REPO, "figures"), exist_ok=True)
    plot_shares(shares, os.path.join(REPO, "figures", "class_hist.png"))
    print("\nall label checks passed; saved figures/class_hist.png")
