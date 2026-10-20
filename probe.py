import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

from data import S5MarsDataset, CLASSES, NUM_CLASSES, IGNORE_INDEX, DATA_ROOT
from evaluate import confusion_matrix, compute_metrics
from models import build_encoder, MODEL_DEFAULTS

ROCK, BEDROCK = CLASSES.index("rock"), CLASSES.index("bedrock")


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device("mps" if torch.backends.mps.is_available() else device)


def probe_transform(size):
    return v2.Compose([
        v2.Resize((size, size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def subset_loader(split, n, size, seed, root, batch, workers):
    ds = S5MarsDataset(split, probe_transform(size), root)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch, shuffle=False, num_workers=workers)

class Extractor:

    def __init__(self, name, device, args):
        self.name, self.device = name, device
        self.is_vit = name.startswith("dinov2")
        if self.is_vit:
            if args.hub_dir:
                self.model = torch.hub.load(args.hub_dir, name, source="local")
            else:
                self.model = torch.hub.load("facebookresearch/dinov2", name)
        else:
            self.model, _, _ = build_encoder(name, args.output_stride, True, MODEL_DEFAULTS["weights_path"])
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x):
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            if self.is_vit:
                tokens = self.model.forward_features(x)["x_norm_patchtokens"]
                b, n, c = tokens.shape
                gh, gw = x.shape[-2] // 14, x.shape[-1] // 14
                return tokens.transpose(1, 2).reshape(b, c, gh, gw).float()
            return self.model(x)["high"].float()  # ResNet-50 stage 4


def grid_labels(y, grid):
    cls = y.clone()
    cls[cls == IGNORE_INDEX] = NUM_CLASSES
    onehot = F.one_hot(cls, NUM_CLASSES + 1).permute(0, 3, 1, 2).float()
    out = F.adaptive_avg_pool2d(onehot, grid).argmax(1)
    out[out == NUM_CLASSES] = IGNORE_INDEX
    return out.to(torch.uint8)


def extract(extractor, loader, device):
    feats, grids, labels = [], [], []
    for x, y, _ in loader:
        x = x.as_subclass(torch.Tensor).to(device)
        y = y.as_subclass(torch.Tensor).to(device).long()
        f = extractor(x)
        feats.append(f.half().cpu())
        grids.append(grid_labels(y, f.shape[-2:]).cpu())
        labels.append(y.to(torch.uint8).cpu())
    return torch.cat(feats), torch.cat(grids), torch.cat(labels)


def channel_stats(feats, device):
    c = feats.shape[1]
    s = torch.zeros(c, dtype=torch.float64, device=device)
    s2 = torch.zeros(c, dtype=torch.float64, device=device)
    count = 0
    for chunk in feats.split(32):
        chunk = chunk.to(device).double()
        s += chunk.sum((0, 2, 3))
        s2 += (chunk ** 2).sum((0, 2, 3))
        count += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]
    mean = s / count
    std = (s2 / count - mean ** 2).clamp(min=1e-12).sqrt()
    return mean.float().view(1, -1, 1, 1), std.float().view(1, -1, 1, 1)

def fit_linear(ftr, gtr, mean, std, device, epochs, lr, bs, seed):
    torch.manual_seed(seed)
    n = len(ftr)
    head = nn.Conv2d(ftr.shape[1], NUM_CLASSES, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    for ep in range(epochs):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            f = (ftr[idx].to(device).float() - mean) / std
            g = gtr[idx].to(device).long()
            loss = F.cross_entropy(head(f), g, ignore_index=IGNORE_INDEX)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep + 1:>3}/{epochs}  train loss {running / n:.4f}")
    return head


@torch.no_grad()
def eval_linear(head, fval, lval, mean, std, device, bs=16):
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for i in range(0, len(fval), bs):
        f = (fval[i:i + bs].to(device).float() - mean) / std
        logits = F.interpolate(head(f), size=lval.shape[-2:], mode="bilinear", align_corners=False)
        pred = logits.argmax(1)
        target = lval[i:i + bs].to(device).long()
        for p, t in zip(pred, target):
            cm += confusion_matrix(p, t)
    return compute_metrics(cm.cpu())


def sample_pixels(feats, labels, mean, std, device, per_image, seed):
    g = torch.Generator().manual_seed(seed)
    xs_all, ys_all = [], []
    for i in range(len(feats)):
        lab = labels[i].long()
        ys, xs = torch.nonzero((lab == ROCK) | (lab == BEDROCK), as_tuple=True)
        if len(ys) == 0:
            continue
        if len(ys) > per_image:
            sel = torch.randperm(len(ys), generator=g)[:per_image]
            ys, xs = ys[sel], xs[sel]
        h, w = lab.shape
        grid = torch.stack([(xs + 0.5) / w * 2 - 1, (ys + 0.5) / h * 2 - 1], -1).view(1, 1, -1, 2).float().to(device)
        f = (feats[i:i + 1].to(device).float() - mean) / std
        s = F.grid_sample(f, grid, mode="bilinear", align_corners=False).view(f.shape[1], -1).T
        xs_all.append(s.half().cpu())
        ys_all.append((lab[ys, xs] == ROCK).long())
    return torch.cat(xs_all), torch.cat(ys_all)


def roc_auc(scores, y):
    order = scores.argsort()
    ranks = torch.empty(len(order), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def binary_probe(ftr, ltr, fval, lval, mean, std, device, per_image, epochs, seed):
    xtr, ytr = sample_pixels(ftr, ltr, mean, std, device, per_image, seed)
    xva, yva = sample_pixels(fval, lval, mean, std, device, per_image, seed)
    print(f"    binary probe pixels: train {len(ytr)} (rock {int(ytr.sum())}), val {len(yva)} (rock {int(yva.sum())})")
    torch.manual_seed(seed)
    lin = nn.Linear(xtr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = ((ytr == 0).sum() / (ytr == 1).sum().clamp(min=1)).float().to(device)
    for _ in range(epochs):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), 4096):
            idx = perm[i:i + 4096]
            logit = lin(xtr[idx].to(device).float()).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, ytr[idx].float().to(device), pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        scores = torch.cat([lin(xva[i:i + 8192].to(device).float()).squeeze(1).cpu() for i in range(0, len(xva), 8192)])
    return roc_auc(scores, yva)

def run_encoder(name, loaders, args, device):
    print(f"\n=== {name} ===")
    t0 = time.time()
    extractor = Extractor(name, device, args)
    ftr, gtr, ltr = extract(extractor, loaders["train"], device)
    fval, gval, lval = extract(extractor, loaders["val"], device)
    del extractor
    print(f"  features: train {tuple(ftr.shape)}, val {tuple(fval.shape)}  ({time.time() - t0:.0f}s)")

    mean, std = channel_stats(ftr, device)
    head = fit_linear(ftr, gtr, mean, std, device, args.epochs, args.lr, args.batch, args.seed)
    metrics = eval_linear(head, fval, lval, mean, std, device)
    auc = binary_probe(ftr, ltr, fval, lval, mean, std, device, args.pixels_per_image, args.bin_epochs, args.seed)

    result = {
        "feature_grid": list(ftr.shape[-2:]),
        "channels": ftr.shape[1],
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "aAcc": metrics["aAcc"],
        "IoU": metrics["IoU"],
        "rock_iou": metrics["IoU"]["rock"],
        "bedrock_iou": metrics["IoU"]["bedrock"],
        "rock_to_bedrock_pct": metrics["rock"]["rock_to_bedrock"],
        "bedrock_to_rock_pct": metrics["rock"]["bedrock_to_rock"],
        "rock_vs_bedrock_auc": auc,
    }
    del ftr, fval, head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def print_table(results):
    print("\n" + "=" * 92)
    print(f"{'encoder':<18}{'grid':>8}{'dim':>6}{'mIoU':>8}{'rock IoU':>10}{'bedrock IoU':>13}"
          f"{'rock->bed %':>13}{'AUC r-vs-b':>12}")
    for name, r in results.items():
        print(f"{name:<18}{'x'.join(map(str, r['feature_grid'])):>8}{r['channels']:>6}{r['mIoU']:>8.2f}"
              f"{r['rock_iou']:>10.2f}{r['bedrock_iou']:>13.2f}{r['rock_to_bedrock_pct']:>13.2f}"
              f"{r['rock_vs_bedrock_auc']:>12.4f}")
    print("\nper-class IoU")
    print(f"{'class':<10}" + "".join(f"{n:>18}" for n in results))
    for c in CLASSES:
        print(f"{c:<10}" + "".join(f"{r['IoU'][c]:>18.2f}" for r in results.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=["dinov2_vits14", "dinov2_vitb14", "resnet50"])
    p.add_argument("--size", type=int, default=518, help="square input size; must be a multiple of 14")
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--bin-epochs", type=int, default=10)
    p.add_argument("--pixels-per-image", type=int, default=500)
    p.add_argument("--output-stride", type=int, default=16, choices=[8, 16], help="ResNet only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--root", default=DATA_ROOT)
    p.add_argument("--hub-dir", help="local clone of facebookresearch/dinov2 (skips the GitHub download)")
    p.add_argument("--out-dir", default=os.path.join("results", "probe"))
    args = p.parse_args()
    assert args.size % 14 == 0, "--size must be a multiple of 14"

    device = get_device()
    print(f"device {device}, input {args.size}x{args.size}, {args.n_train} train / {args.n_val} val images")
    loaders = {
        "train": subset_loader("train", args.n_train, args.size, args.seed, args.root, 16, args.workers),
        "val": subset_loader("val", args.n_val, args.size, args.seed, args.root, 16, args.workers),
    }

    results = {name: run_encoder(name, loaders, args, device) for name in args.encoders}
    print_table(results)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "probe_results.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nsaved {args.out_dir}/probe_results.json")