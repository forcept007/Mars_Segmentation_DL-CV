import os
import sys
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data import S5MarsDataset, train_transform
from train_ssl import augin
from visualize_batch import MEAN, STD

if __name__ == "__main__":
    torch.manual_seed(0)
    dataset = S5MarsDataset("train", train_transform())
    picks = torch.randperm(len(dataset))[:8].tolist()
    fig, axs = plt.subplots(2, 4, figsize=(16,8))
    images = []
    for k, idx in enumerate(picks):
        image, label, name = dataset[idx]
        images.append(image)

    x = torch.stack(images, dim=0)
    # Apply augin
    x_strong = augin(x, 'sigma')

    for j in range(4):
        weak_im = (x[j] * STD) + MEAN
        axs[0,j].imshow(weak_im.clamp(0, 1).permute(1, 2, 0))
        axs[0,j].set_title('Weak Augmentation')
        axs[0,j].axis('off')
        strong_im = (x_strong[j] * STD) + MEAN
        axs[1,j].imshow(strong_im.clamp(0, 1).permute(1, 2, 0))
        axs[1,j].set_title(f'Strong Augmentation: x{x_strong[j].std()/x[j].std():.2f}')
        axs[1,j].axis('off')

    os.makedirs(os.path.join(REPO, "figures"), exist_ok=True)
    fig.savefig(os.path.join(REPO, "figures", "augin_sigma.jpg"), dpi=80)
    print("saved figures/augin_sigma.jpg")
