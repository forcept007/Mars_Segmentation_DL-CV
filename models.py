import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter

from data import NUM_CLASSES

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained", "resnet50.pth")
MODEL_DEFAULTS = {"encoder": "resnet50", "output_stride": 16, "pretrained": True,
                  "weights_path": WEIGHTS_PATH, "num_classes": NUM_CLASSES}

# https://github.com/LiheYoung/UniMatch/blob/main/model/backbone/resnet.py
def deep_stem_resnet50(replace_stride_with_dilation, weights_path=None):
    net = resnet50(replace_stride_with_dilation=replace_stride_with_dilation)
    net.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                              nn.BatchNorm2d(64),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(64, 64, 3, padding=1, bias=False),
                              nn.BatchNorm2d(64),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(64, 128, 3, padding=1, bias=False))
    net.bn1 = nn.BatchNorm2d(128)
    net.layer1[0].conv1 = nn.Conv2d(128, 64, 1, bias=False)
    net.layer1[0].downsample[0] = nn.Conv2d(128, 256, 1, bias=False)
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"{weights_path} not found; download ResNet-50 from "
                                    "https://github.com/LiheYoung/UniMatch#pretrained-backbone into pretrained/")
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return net

def build_encoder(name, output_stride, pretrained, weights_path):
    dilate = {16: [False, False, True], 8: [False, True, True]}[output_stride]
    if name == "resnet50":
        net = deep_stem_resnet50(dilate, weights_path if pretrained else None)
    elif name == "resnet50_torchvision":
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None, replace_stride_with_dilation=dilate)
    else:
        raise ValueError(f"unknown encoder {name}")
    return IntermediateLayerGetter(net, return_layers={"layer1": "low", "layer4": "high"}), 256, 2048

def build_model(cfg=None):
    cfg = {**MODEL_DEFAULTS, **(cfg or {})}
    encoder, low_channels, high_channels = build_encoder(cfg["encoder"], cfg["output_stride"],
                                                         cfg["pretrained"], cfg["weights_path"])
    dilations = {16: (6, 12, 18), 8: (12, 24, 36)}[cfg["output_stride"]]
    return DeepLabV3Plus(encoder, low_channels, high_channels, cfg["num_classes"], dilations)

def conv_bn_relu(in_ch, out_ch, kernel_size=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=dilation * (kernel_size // 2), dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True))

# https://github.com/LiheYoung/UniMatch/blob/main/model/semseg/deeplabv3plus.py
class DeepLabV3Plus(nn.Module):
    def __init__(self, encoder, low_channels, high_channels, num_classes=NUM_CLASSES, dilations=(6, 12, 18)):
        super().__init__()
        self.encoder = encoder
        self.aspp = ASPP(high_channels, 256, dilations)
        self.reduce = conv_bn_relu(low_channels, 48)
        self.fuse = nn.Sequential(conv_bn_relu(256 + 48, 256, 3), conv_bn_relu(256, 256, 3))
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.encoder(x)
        low = self.reduce(feats["low"])
        high = self.aspp(feats["high"])
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        out = self.classifier(self.fuse(torch.cat([low, high], dim=1)))
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

    def param_groups(self, head_lr_mult=10.0):
        head = [p for name, p in self.named_parameters() if not name.startswith("encoder.")]
        return [{"params": list(self.encoder.parameters()), "lr_mult": 1.0},
                {"params": head, "lr_mult": head_lr_mult}]

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn_relu(in_ch, out_ch)] +
                                      [conv_bn_relu(in_ch, out_ch, 3, d) for d in dilations])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_bn_relu(in_ch, out_ch))
        self.project = conv_bn_relu(out_ch * (len(dilations) + 2), out_ch)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        outs.append(F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False))
        return self.project(torch.cat(outs, dim=1))
