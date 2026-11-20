import torch

from data import IGNORE_INDEX

# Lovász-softmax loss (Berman et al., CVPR 2018): a smooth version of 1 - IoU, averaged over classes.
# Reference code: https://github.com/bermanmaxim/LovaszSoftmax

def lovasz_grad(gt_sorted):
    # How much the IoU drops as each pixel (sorted by error) is counted wrong
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1 - intersection / union
    jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def lovasz_softmax(logits, labels, ignore_index=IGNORE_INDEX):
    # logits (B, C, H, W), labels (B, H, W). Uses only the classes present in the batch.
    probs = logits.softmax(dim=1).permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    labels = labels.reshape(-1)
    valid = labels != ignore_index
    probs, labels = probs[valid], labels[valid]
    losses = []
    for c in labels.unique():
        fg = (labels == c).float()
        errors = (fg - probs[:, c]).abs()
        errors, order = errors.sort(descending=True)
        losses.append(torch.dot(errors, lovasz_grad(fg[order])))
    return torch.stack(losses).mean()
