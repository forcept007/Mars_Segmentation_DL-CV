import json
import random

from pycocotools import mask as mask_utils

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

def load_sam_mask(json_path: str, min_area: float = 0.001, max_area: float = 0.10, min_iou: float = 0.88):
    # Paper Sec. IV-B: randomly select one qualifying SAM mask for this image.
    # Returns an H x W uint8 array (1 inside the mask), or None if no mask qualifies (caller uses CutMix).
    with open(json_path) as f:
        record = json.load(f)
    kept = filter_masks(record['masks'], min_area, max_area, min_iou)
    if len(kept) == 0:
        return None

    m = random.choice(kept)
    return mask_utils.decode(m['segmentation'])
