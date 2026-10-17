import json
import random
import os
import torch

from pycocotools import mask as mask_utils
from torchvision import tv_tensors
from torchvision.io import decode_image, ImageReadMode
from data import DATA_ROOT, S5MarsDataset, load_label

def filter_masks(masks: list[dict], min_area: float = 0.001, max_area: float = 0.10, min_iou: float = 0.88):
    # Paper Sec. IV-B: keep SAM masks whose size is in a range and whose confidence is above a threshold.
    # masks: the "masks" list from one precomputed JSON file.
    # min_area / max_area: fraction of the image area (the paper gives no values; our choice).
    # Returns the qualifying masks; an empty list means the caller falls back to a random rectangle (CutMix).
    kept = []
    for m in masks:
        h, w = m['segmentation']['size']
        area_frac = m['area']/(h*w)
        # min_iou at 0.88 matches SAM
        if min_area <= area_frac <= max_area and m['predicted_iou'] > min_iou:
            kept.append(m)
    return kept

def load_sam_id_map(json_path: str, min_area: float = 0.001, max_area: float = 0.10, min_iou: float = 0.88):
    # Before Paper Sec. IV-B (randomly select one qualifying SAM mask for this image.)
    # This function returns an id_map of all masks in the original image. Then when the map is cropped
    # the same as the training image, all masks remain within the crop. This avoids picking one random
    # mask before the crop and then facing the possibility of the mask disappearing after the crop.
    with open(json_path) as f:
        record = json.load(f)
    kept = filter_masks(record['masks'], min_area, max_area, min_iou)
    if len(kept) == 0:
        return None
    h, w = kept[0]['segmentation']['size']
    id_map = torch.zeros((h, w), dtype=torch.int16)

    kept.sort(key=lambda a: a['area'], reverse=True)
    for k, m in enumerate(kept, start=1):
        decoded = mask_utils.decode(m['segmentation'])
        mask = torch.tensor(decoded, dtype=torch.bool)
        id_map[mask] = k

    return id_map

def pick_sam_mask(id_map: torch.Tensor, min_pixels: int):
    # Paper Sec. IV-B (randomly select one qualifying SAM mask for this image.)
    # After the crop: randomly choose one mask id that still has at least min_pixels pixels.
    # Returns a 0/1 uint8 mask with the same shape as id_map, all zeros if no id qualifies (caller uses CutMix).
    ids, counts = id_map.unique(sorted=True, return_counts=True)
    candidates = ids[(ids > 0) & (counts >= min_pixels)]
    
    if len(candidates) == 0:
        return torch.zeros(id_map.shape, dtype=torch.uint8)
    
    k = random.choice(candidates)
    
    return torch.where(id_map == k, 1, 0).to(torch.uint8)


class SamMixDataset(S5MarsDataset):
    def __init__(self, split, sam_root, transform=None, root=DATA_ROOT, limit=None, min_pixels=260):
        super().__init__(split, transform, root, limit)

        self.sam_root = sam_root
        self.min_pixels = min_pixels

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.ids[idx] + ".jpg")
        mask_path = os.path.join(self.root, "labels", self.ids[idx] + ".png")
        sam_path = os.path.join(self.sam_root, self.ids[idx] + ".json")

        image = tv_tensors.Image(decode_image(img_path, mode=ImageReadMode.RGB)) # (3,H,W) uint8
        label = tv_tensors.Mask(load_label(mask_path)) # (H,W) 0–8, 255

        id_map = load_sam_id_map(sam_path)
        if id_map is None:
            id_map = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.int16)
        sam = tv_tensors.Mask(id_map)

        if self.transform:
            image, label, sam = self.transform(image, label, sam)
        sam = pick_sam_mask(sam, self.min_pixels)

        return image, label, sam, self.ids[idx]
