import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model, WEIGHTS_PATH

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def test_output_shape_and_param_count():
    model = build_model({"pretrained": False}).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 512, 512))
    assert out.shape == (2, 9, 512, 512)
    print(f"parameters: {count_params(model) / 1e6:.2f}M (S5Mars Table XIII: 40.47M)")
    assert round(count_params(model) / 1e6, 2) == 40.47

def test_output_stride():
    x = torch.randn(1, 3, 512, 512)
    for stride in (16, 8):
        model = build_model({"pretrained": False, "output_stride": stride}).eval()
        with torch.no_grad():
            assert model.encoder(x)["high"].shape[-1] == 512 // stride
            assert model(x).shape == (1, 9, 512, 512)

def test_torchvision_encoder():
    model = build_model({"pretrained": False, "encoder": "resnet50_torchvision"})
    assert round(count_params(model) / 1e6, 2) == 40.35

def test_param_groups():
    model = build_model({"pretrained": False})
    encoder, head = model.param_groups(head_lr_mult=10.0)
    assert sum(p.numel() for p in encoder["params"]) + sum(p.numel() for p in head["params"]) == count_params(model)
    assert (encoder["lr_mult"], head["lr_mult"]) == (1.0, 10.0)

def test_pretrained_weights_loaded():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"skipped: {WEIGHTS_PATH} not downloaded")
        return
    model = build_model()
    saved = torch.load(WEIGHTS_PATH, map_location="cpu")
    loaded = model.encoder.state_dict()
    assert all(torch.equal(loaded[k], saved[k]) for k in loaded)

def test_runs_on_gpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    if device.type == "cpu":
        print("skipped: no GPU")
        return
    model = build_model({"pretrained": False}).to(device)
    loss = model(torch.randn(2, 3, 512, 512, device=device)).mean()
    loss.backward()
    assert model.classifier.weight.grad is not None

if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: passed")
