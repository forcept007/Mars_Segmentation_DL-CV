import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter

from data import NUM_CLASSES

# https://github.com/LiheYoung/UniMatch/blob/main/model/semseg/deeplabv3plus.py
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, dilations=(6, 12, 18)):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        self.backbone = IntermediateLayerGetter(resnet, return_layers={"layer1": "low", "layer4": "high"})

        self.aspp = ASPP(2048, 256, dilations)
        self.reduce = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.backbone(x)
        low = self.reduce(feats["low"])
        high = self.aspp(feats["high"])
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        out = self.classifier(self.fuse(torch.cat([low, high], dim=1)))
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

def conv_bn_relu(in_ch, out_ch, kernel_size=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size, padding=dilation * (kernel_size // 2), dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(inplace=True))

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
