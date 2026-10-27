import os
import sys
import torch
from torchvision import tv_tensors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import IMAGENET_MEAN, IMAGENET_STD, eval_transform, norm_stats, train_transform

def sample(h=1200, w=1344):
    image = tv_tensors.Image(torch.randint(0, 256, (3, h, w), dtype=torch.uint8))
    label = tv_tensors.Mask(torch.randint(0, 9, (h, w)))
    return image, label

def test_norm_stats_defaults_to_imagenet():
    assert norm_stats({}) == {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    assert norm_stats({"mean": [0.6, 0.5, 0.4], "std": [0.1, 0.1, 0.1]}) == {"mean": (0.6, 0.5, 0.4),
                                                                           "std": (0.1, 0.1, 0.1)}

def test_eval_transform_crop_and_full_res():
    image, label = sample()
    x, y = eval_transform()(image, label)
    assert x.shape == (3, 1024, 1024) and y.shape == (1024, 1024)
    x, y = eval_transform(None)(image, label)
    assert x.shape == (3, 1200, 1344) and y.shape == (1200, 1344)
    assert torch.equal(y, label)

def test_custom_normalization():
    image = tv_tensors.Image(torch.full((3, 1100, 1100), 128, dtype=torch.uint8))
    label = tv_tensors.Mask(torch.zeros(1100, 1100, dtype=torch.long))
    mean, std = (0.6193, 0.5137, 0.3601), (0.1374, 0.1248, 0.1033)
    x, _ = eval_transform(1024, mean, std)(image, label)
    expected = (128 / 255 - torch.tensor(mean)) / torch.tensor(std)
    assert torch.allclose(x[:, 0, 0], expected, atol=1e-5)
    x, _ = train_transform((1.0, 1.0), mean=mean, std=std)(image, label)
    assert x.shape == (3, 512, 512) and torch.allclose(x[:, 0, 0], expected, atol=1e-5)

if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
