import os
import sys
import math
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import CLASSES, NUM_CLASSES
from evaluate import confusion_matrix, compute_metrics, evaluate

def close(a, b):
    return math.isclose(a, b, abs_tol=1e-9)

def test_three_class_example():
    target = torch.tensor([0, 0, 1, 1, 2, 2, 255])
    pred = torch.tensor([0, 1, 1, 1, 2, 0, 2])
    cm = confusion_matrix(pred, target, num_classes=3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    m = compute_metrics(cm, class_names=["a", "b", "c"])
    assert close(m["IoU"]["a"], 100 / 3) and close(m["IoU"]["b"], 200 / 3) and close(m["IoU"]["c"], 50)
    assert close(m["Acc"]["a"], 50) and close(m["Acc"]["b"], 100) and close(m["Acc"]["c"], 50)
    assert close(m["mIoU"], 50) and close(m["mAcc"], 200 / 3) and close(m["aAcc"], 200 / 3)
    assert "rock" not in m

def test_rock_metrics():
    rock, bedrock, soil, sand = (CLASSES.index(c) for c in ("rock", "bedrock", "soil", "sand"))
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    cm[rock, rock], cm[rock, bedrock], cm[rock, soil] = 6, 3, 1
    cm[bedrock, bedrock], cm[bedrock, rock] = 18, 2
    cm[sand, rock] = 2
    r = compute_metrics(cm)["rock"]
    expected = {"precision": 60, "recall": 60, "f1": 60, "rock_to_bedrock": 30, "bedrock_to_rock": 10,
                "fn_predicted_bedrock": 75, "fp_from_bedrock": 50}
    assert all(close(r[k], v) for k, v in expected.items()), r

def test_evaluate_subsets_and_train_mode():
    target = torch.randint(0, NUM_CLASSES, (4, 8, 8))
    target[0, 0, 0] = 255
    onehot = nn.functional.one_hot(target.clamp(max=NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2).float()
    loader = [(onehot[:2], target[:2], ("easy/a", "hard/b")), (onehot[2:], target[2:], ("easy/c", "easy/d"))]
    model = nn.Identity().train()
    m = evaluate(model, loader, torch.device("cpu"))
    assert model.training
    assert close(m["mIoU"], 100)
    assert sum(map(sum, m["subsets"]["easy"]["confusion_matrix"])) == 3 * 64 - 1
    assert sum(map(sum, m["subsets"]["hard"]["confusion_matrix"])) == 64

if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(f"{name}: passed")
