import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2 #https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
from torchvision import tv_tensors # debug with claude

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class S5MarsDataset(Dataset):
    def __init__(self, split_file, transform=None):
        with open(split_file) as f:
            self.ids = json.load(f)  # e.g. "easy/0271MR..._DXXX"
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join("S5Mars_data/images", self.ids[idx] + ".jpg")
        mask_path = os.path.join("S5Mars_data/labels", self.ids[idx] + ".png")

        # debug with claude
        image = tv_tensors.Image(decode_image(img_path, mode=ImageReadMode.RGB))           # (3,H,W) uint8
        label = tv_tensors.Mask(torch.from_numpy(np.array(Image.open(mask_path))).long())  # (H,W) 0–8, 255

        if self.transform:
            image, label = self.transform(image, label)  # same crop/flip applied to both
        return image, label

def get_dataloaders(batch: int = 32):
    train_transform = v2.Compose([
        v2.RandomCrop(512),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet stats
    ])
    test_transform = v2.Compose([
        v2.CenterCrop(1024),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet stats
    ])

    training_data = S5MarsDataset("S5Mars_data/split/train.json", train_transform)
    val_data = S5MarsDataset("S5Mars_data/split/val.json", test_transform)
    test_data = S5MarsDataset("S5Mars_data/split/test.json", test_transform)

    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader