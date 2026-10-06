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

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S5Mars_data")
NUM_CLASSES = 9      # sky, ridge, soil, sand, bedrock, rock, rover, trace, hole (S5Mars_data/Label.txt)
IGNORE_INDEX = 255   # unlabeled pixels, skipped by the loss and metrics

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class S5MarsDataset(Dataset):
    def __init__(self, split, transform=None, root=DATA_ROOT):
        with open(os.path.join(root, "split", split + ".json")) as f:
            self.ids = json.load(f)  # e.g. "easy/0271MR..._DXXX"
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.ids[idx] + ".jpg")
        mask_path = os.path.join(self.root, "labels", self.ids[idx] + ".png")

        # debug with claude
        image = tv_tensors.Image(decode_image(img_path, mode=ImageReadMode.RGB))  # (3,H,W) uint8
        raw = torch.from_numpy(np.array(Image.open(mask_path))).long()            # (H,W) 0 = unlabeled, 1–9 = classes
        label = tv_tensors.Mask(torch.where(raw == 0, IGNORE_INDEX, raw - 1))     # (H,W) 0–8 = train ids, 255 = ignore

        if self.transform:
            image, label = self.transform(image, label)  # same crop/flip applied to both
        return image, label

def get_dataloaders(batch: int = 8, eval_batch: int = 4, num_workers: int = 4):
    # Paper Sec. V-B: 512x512 training crops (weak augs: resize, crop, flip), test images center-cropped to 1024x1024
    train_transform = v2.Compose([
        v2.RandomResize(576, 2400),  # short side is 1152–1200 px, so this rescales ~0.5x–2x (UniMatch/AugSeg range)
        v2.RandomCrop(512, pad_if_needed=True, fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE_INDEX}),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet stats
    ])
    test_transform = v2.Compose([
        v2.CenterCrop(1024),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet stats
    ])

    training_data = S5MarsDataset("train", train_transform)
    val_data = S5MarsDataset("val", test_transform)
    test_data = S5MarsDataset("test", test_transform)

    pin = torch.cuda.is_available()
    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=pin)
    val_dataloader = DataLoader(val_data, batch_size=eval_batch, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_dataloader = DataLoader(test_data, batch_size=eval_batch, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_dataloader, val_dataloader, test_dataloader
