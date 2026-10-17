import json
import random
import os
import torch
import torch.nn.functional as F
import math
import numpy as np

from pycocotools import mask as mask_utils
from torchvision import tv_tensors
from torchvision.transforms.functional import gaussian_blur
from torchvision.io import decode_image, ImageReadMode
from torch.utils.data import DataLoader
from data import DATA_ROOT, S5MarsDataset, load_label, train_transform

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
    # Before Paper Sec. IV-B: (randomly select one qualifying SAM mask for this image.)
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
    # Paper Sec. IV-B: (randomly select one qualifying SAM mask for this image.)
    # After the crop: randomly choose one mask id that still has at least min_pixels pixels.
    # Returns a 0/1 uint8 mask with the same shape as id_map, all zeros if no id qualifies (caller uses CutMix).
    ids, counts = id_map.unique(sorted=True, return_counts=True)
    candidates = ids[(ids > 0) & (counts >= min_pixels)]
    
    if len(candidates) == 0:
        return torch.zeros(id_map.shape, dtype=torch.uint8)
    
    k = random.choice(candidates)
    
    return torch.where(id_map == k, 1, 0).to(torch.uint8)

def mask_transform(mask: torch.Tensor, image: torch.Tensor, raw: torch.Tensor, label: torch.Tensor, 
                   max_angle: float = 30.0, scale_range: tuple[float, float] = (0.75, 1.25), flip_p: float = 0.5,
                   angle: torch.Tensor = None, flip: torch.Tensor = None,
                   scale: torch.Tensor = None):
    # Paper Sec. IV-B: rotate, flip and rescale the masked object around its own centroid (pasted at the same
    # position), applying one random transform per sample to the mask, image, teacher logits and label alike.
    # mask, label: (B, H, W). image: (B, 3, H, W). raw (teacher logits): (B, C, H, W).
    # angle (degrees), flip (bool), scale: optional (B,) tensors; drawn at random when None (tests pass them).
    B, H, W = mask.shape
    dev = mask.device

    if angle is None:
        angle = (torch.rand(B, device=dev) * 2 - 1) * max_angle
    if flip is None:
        flip = torch.rand(B, device=dev) <= flip_p
    if scale is None:
        scale = scale_range[0] + torch.rand(B, device=dev) * (scale_range[1] - scale_range[0])

    # Object centroid in normalized coordinates (-1 = left/top edge, +1 = right/bottom edge)
    xs = torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=dev)
    ys = torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=dev)
    m = mask.float()
    area = m.sum(dim=(1, 2)).clamp(min=1)
    cx = (m * xs[None, None, :]).sum(dim=(1, 2)) / area
    cy = (m * ys[None, :, None]).sum(dim=(1, 2)) / area

    # theta = [A | c - A c], A = S^-1 R(-a), S = diag(sx, scale), sx negative where flipped:
    # A = [[ cos(a)/sx,     sin(a)/sx    ],
    #      [-sin(a)/scale,  cos(a)/scale ]]
    a = torch.deg2rad(angle)
    sx = torch.where(flip, -scale, scale)
    row0 = torch.stack([torch.cos(a)/sx, torch.sin(a)/sx], dim=1)
    row1 = torch.stack([-torch.sin(a)/scale, torch.cos(a)/scale], dim=1)
    A = torch.stack([row0, row1], dim=1)
    c = torch.stack([cx, cy], dim=-1)
    t = c - (A @ c[..., None])[..., 0]
    theta = torch.cat([A, t[..., None]], dim=2)

    # 4. One grid, four tensors
    grid = F.affine_grid(theta, (B, 1, H, W), align_corners=False)
    mask = F.grid_sample(m[:,None], grid, mode='nearest', align_corners=False)[:,0]
    image = F.grid_sample(image, grid, mode='bilinear', align_corners=False)
    raw = F.grid_sample(raw, grid, mode='bilinear', align_corners=False)
    label = F.grid_sample(label[:,None].float(), grid, mode='nearest', align_corners=False)[:,0]
    return mask, image, raw, label

def mix_batch(x_strong: torch.Tensor, logits_t: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, **transform_args):
    # Paper Sec. IV-B: paste one transformed object from a partner sample into each sample, same position.
    # x_strong: (B, 3, H, W) strong view (after AugIN). logits_t: (B, C, H, W) teacher logits on the weak view.
    # label: (B, H, W) ground truth, 255 = unlabeled. mask: (B, H, W) 0/1 masks from SamMixDataset.
    # transform_args: passed on to mask_transform (max_angle, scale_range, flip_p, or fixed angle/flip/scale in tests).
    # Returns the mixed strong view, the mixed teacher logits and the mixed label (for the unsupervised loss).
    B = x_strong.shape[0]
    partner = torch.roll(torch.arange(B, device=x_strong.device), 1)   # sample i receives sample i-1's object

    mask_p, x_p, logits_p, label_p = mask_transform(mask[partner], x_strong[partner], logits_t[partner].float(), label[partner], **transform_args)

    paste = torch.where(mask_p >= 0.5, 1, 0)
    x_mix = torch.where(paste[:, None] == 1, x_p, x_strong)
    logits_mix = torch.where(paste[:, None] == 1, logits_p, logits_t.float())
    label_mix = torch.where(paste == 1, label_p, label).long()
    return x_mix, logits_mix, label_mix

def random_box(h: int, w: int, area_range=(0.02, 0.4), ratio_range=(0.3, 1 / 0.3)):
    # Paper Sec. IV-B: "If there is no qualified mask, a random rectangular mask will be directly generated" (CutMix).
    # Size and shape ranges follow UniMatch's obtain_cutmix_box. Returns an (h, w) uint8 mask, 1 inside the box.
    area = random.uniform(area_range[0], area_range[1]) * (h * w)
    ratio = random.uniform(ratio_range[0], ratio_range[1])
    box_h = min(round(math.sqrt(area*ratio)), h)
    box_w = min(round(box_h / ratio), w)
    top = random.randint(0, h - box_h)
    left = random.randint(0, w - box_w)
    mask = torch.zeros(h, w, dtype=torch.uint8)
    mask[top:top + box_h, left:left + box_w] = 1
    return mask

class SamMixDataset(S5MarsDataset):
    def __init__(self, split, sam_root, transform=None, root=DATA_ROOT, limit=None, min_pixels=260, sigma=2):
        super().__init__(split, transform, root, limit)

        self.sam_root = sam_root
        self.min_pixels = min_pixels
        self.sigma = sigma

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.ids[idx] + ".jpg")
        mask_path = os.path.join(self.root, "labels", self.ids[idx] + ".png")

        image = tv_tensors.Image(decode_image(img_path, mode=ImageReadMode.RGB)) # (3,H,W) uint8
        label = tv_tensors.Mask(load_label(mask_path)) # (H,W) 0–8, 255

        if self.sam_root is None:
            id_map = None
        else:
            sam_path = os.path.join(self.sam_root, self.ids[idx] + ".json")
            id_map = load_sam_id_map(sam_path)

        if id_map is None:
            id_map = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.int16)
        sam = tv_tensors.Mask(id_map)

        if self.transform:
            image, label, sam = self.transform(image, label, sam)
        sam = pick_sam_mask(sam, self.min_pixels)

        # Apply a Gaussian filter on the mask
        kernel = 2 * math.ceil(3 * self.sigma) + 1
        sam = gaussian_blur(sam[None].float(), kernel, self.sigma)[0]
        sam = torch.where(sam >= 0.5, 1, 0).to(torch.uint8)

        # Check if the mask is empty and if so, fall back to CutMix
        if sam.sum() == 0:
            sam = random_box(sam.shape[0], sam.shape[1])

        return image, label, sam, self.ids[idx]

def get_mix_loader(batch: int, num_workers: int, sam_root: str, root=DATA_ROOT, scale_range=(0.5, 2.0),
                   min_pixels: int = 260, sigma: float = 2, pin_memory: bool = False):
    # Train loader for SAM-Mix (sam_root = folder of precomputed masks) or CutMix (sam_root = None: every mask is a
    # random box). Same weak transform and loader settings as data.get_dataloaders; items are (image, label, mask, id).
    dataset = SamMixDataset("train", sam_root, train_transform(scale_range), root, min_pixels=min_pixels, sigma=sigma)
    return DataLoader(dataset, batch_size=batch, num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
