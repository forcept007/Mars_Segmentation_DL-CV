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

DATA_ROOT = os.environ.get("S5MARS_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "S5Mars_data"))
CLASSES = ["sky", "ridge", "soil", "sand", "bedrock", "rock", "rover", "trace", "hole"]
NUM_CLASSES = len(CLASSES)
IGNORE_INDEX = 255

def load_label(mask_path):
    raw = torch.from_numpy(np.array(Image.open(mask_path))).long()
    return torch.where(raw == 0, IGNORE_INDEX, raw - 1)

class RandomScale(v2.Transform):
    def __init__(self, scale_range=(0.5, 2.0)):
        super().__init__()
        self.scale_range = scale_range

    def make_params(self, flat_inputs):
        h, w = v2.query_size(flat_inputs)
        s = torch.empty(1).uniform_(*self.scale_range).item()
        return {"size": [round(h * s), round(w * s)]}

    def transform(self, inpt, params):
        return self._call_kernel(v2.functional.resize, inpt, params["size"])

def train_transform(scale_range=(0.5, 2.0), crop=512):
    return v2.Compose([
        RandomScale(scale_range),
        v2.RandomCrop(crop, pad_if_needed=True, fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE_INDEX}),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet stats
    ])

def eval_transform(crop=1024):
    return v2.Compose([
        v2.CenterCrop(crop),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # ImageNet stats
    ])

# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class S5MarsDataset(Dataset):
    def __init__(self, split, transform=None, root=DATA_ROOT, limit=None):
        with open(os.path.join(root, "split", split + ".json")) as f:
            self.ids = json.load(f)[:limit]  # e.g. "easy/0271MR..._DXXX"
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.ids[idx] + ".jpg")
        mask_path = os.path.join(self.root, "labels", self.ids[idx] + ".png")

        # debug with claude
        image = tv_tensors.Image(decode_image(img_path, mode=ImageReadMode.RGB))           # (3,H,W) uint8
        label = tv_tensors.Mask(load_label(mask_path))     # (H,W) 0–8, 255

        if self.transform:
            image, label = self.transform(image, label)  # same crop/flip applied to both
        return image, label, self.ids[idx]

def get_dataloaders(batch: int = 8, eval_batch: int = 4, num_workers: int = 4, root=DATA_ROOT,
                    scale_range=(0.5, 2.0), pin_memory=False):
    training_data = S5MarsDataset("train", train_transform(scale_range), root)
    val_data = S5MarsDataset("val", eval_transform(), root)
    test_data = S5MarsDataset("test", eval_transform(), root)

    loader_args = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    train_dataloader = DataLoader(training_data, batch_size=batch, shuffle=True, drop_last=True, **loader_args)
    val_dataloader = DataLoader(val_data, batch_size=eval_batch, shuffle=False, **loader_args)
    test_dataloader = DataLoader(test_data, batch_size=eval_batch, shuffle=False, **loader_args)

    return train_dataloader, val_dataloader, test_dataloader
