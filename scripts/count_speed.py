import time
import torch
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.randn(1, 3, 1024, 1024, device=dev)
for enc in ["resnet50", "dinov2_vits14", "dinov2_vitb14"]:
    model = build_model({"encoder": enc, "pretrained": False}).to(dev).eval()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    encp = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
        for _ in range(5):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(20):
            model(x)
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t) / 20 * 1000
    print(f"{enc:15s} params {total:6.1f}M (encoder {encp:5.1f}M)  {ms:6.1f} ms / 1024 crop on {dev}")
    del model