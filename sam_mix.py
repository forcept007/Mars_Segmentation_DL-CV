
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