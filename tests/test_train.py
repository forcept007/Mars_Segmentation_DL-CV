import os
import sys
import math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from train import load_config, poly_lr

def test_full_config_matches_paper():
    cfg = load_config(os.path.join(REPO, "configs", "baseline_r50_full.yaml"))
    assert cfg["model"] == {"encoder": "resnet50", "output_stride": 16, "pretrained": True}
    assert cfg["data"]["batch_size"] == 8 and cfg["data"]["scale_range"] == [0.5, 2.0]
    assert (cfg["optim"]["lr"], cfg["optim"]["momentum"], cfg["optim"]["poly_power"]) == (0.01, 0.9, 0.9)
    assert cfg["schedule"]["max_iters"] == 150000 and cfg["schedule"]["eval_interval"] == 5000

def test_base_inheritance_and_overrides():
    cfg = load_config(os.path.join(REPO, "configs", "baseline_r50_short.yaml"),
                      ["schedule.eval_interval=1000", "data.root=/tmp/s5mars", "optim.lr=0.02"])
    assert cfg["schedule"]["max_iters"] == 40000 and cfg["schedule"]["eval_interval"] == 1000
    assert cfg["data"]["root"] == "/tmp/s5mars" and cfg["optim"]["lr"] == 0.02
    assert cfg["data"]["batch_size"] == 8 and "base" not in cfg

def test_poly_lr():
    assert poly_lr(0.01, 0, 150000) == 0.01
    assert math.isclose(poly_lr(0.01, 75000, 150000), 0.01 * 0.5 ** 0.9)
    assert poly_lr(0.01, 150000, 150000) == 0.0

if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: passed")
