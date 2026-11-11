import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from samfill_labels import fill
from train import load_config

def test_fill_only_changes_unlabeled_pixels_in_pure_segments():
    raw = np.zeros((4, 8), dtype=np.uint8)
    raw[:, :3] = 6           # segment 1: 12 labeled rock pixels, 4 gaps
    raw[:, 4:6] = 5          # segment 2: half bedrock, half rock, 4 gaps
    raw[:2, 6:8] = 6
    ids = np.zeros((4, 8), dtype=np.int32)
    ids[:, :4] = 1
    ids[:, 4:8] = 2
    out = fill(raw, ids, 2, rho=0.8, n_min=1)
    assert (out[:, 3] == 6).all()                   # pure segment, gaps get rock
    assert (out[2:, 6:8] == 0).all()                # mixed segment (50/50) stays unlabeled
    assert (out[raw > 0] == raw[raw > 0]).all()     # labeled pixels never change

def test_fill_needs_enough_labeled_pixels_and_skips_no_segment():
    raw = np.zeros((2, 4), dtype=np.uint8)
    raw[0, 0] = 3
    ids = np.zeros((2, 4), dtype=np.int32)
    ids[:, :2] = 1           # segment 1 has 1 labeled pixel
    assert (fill(raw, ids, 1, rho=0.8, n_min=2) == raw).all()
    out = fill(raw, ids, 1, rho=0.8, n_min=1)
    assert (out[:, :2] == 3).all() and (out[:, 2:] == 0).all()   # id 0 (no segment) is never filled

def test_samfill_configs():
    b1 = load_config(os.path.join(REPO, "configs", "baseline_r50_short_1xHeadLR.yaml"))
    cfg = load_config(os.path.join(REPO, "configs", "samfill_b1_short.yaml"))
    assert cfg["data"].pop("train_label_dir") == "labels_samfill_rho0.8"
    assert cfg == b1
    full = load_config(os.path.join(REPO, "configs", "samfill_b1_full.yaml"))
    assert full["data"].pop("train_label_dir") == "labels_samfill_rho0.8"
    assert full == load_config(os.path.join(REPO, "configs", "baseline_r50_full_1xHeadLR.yaml"))
    ssl = load_config(os.path.join(REPO, "configs", "samfill_ssl_short.yaml"))
    assert ssl["data"]["train_label_dir"] == "labels_samfill_rho0.8"
    assert ssl["ssl"]["unsup_pixels"] == "all_gt" and ssl["schedule"]["max_iters"] == 40000

if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
