import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
from torch import nn

from data import IGNORE_INDEX, get_dataloaders
from evaluate import evaluate
from model import DeepLabV3Plus

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", default="runs/supervised")
parser.add_argument("--epochs", type=int, default=240)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--head-lr-mult", type=float, default=10.0)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--eval-every", type=int, default=1)
parser.add_argument("--log-every", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)

def main():
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.save_dir, "train.log")), logging.StreamHandler()])
    log = logging.getLogger()
    log.info(json.dumps(vars(args)))
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("mps" if torch.backends.mps.is_available() else device)

    train_loader, val_loader, _ = get_dataloaders(args.batch_size, num_workers=args.num_workers)
    model = DeepLabV3Plus().to(device)
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.SGD([{"params": model.backbone.parameters(), "lr": args.lr},
                                 {"params": head_params, "lr": args.lr * args.head_lr_mult}],
                                lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    start_epoch, best_miou = 0, 0.0
    latest_path = os.path.join(args.save_dir, "latest.pth")
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, best_miou = ckpt["epoch"] + 1, ckpt["best_miou"]
        log.info(f"resumed from {latest_path} at epoch {start_epoch}")

    iters_per_epoch = len(train_loader)
    total_iters = args.epochs * iters_per_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        start = time.time()
        loss_sum = torch.zeros((), device=device)
        for i, (x, y) in enumerate(train_loader):
            lr = args.lr * (1 - (epoch * iters_per_epoch + i) / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * args.head_lr_mult

            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()

            if (i + 1) % args.log_every == 0:
                sec_per_iter = (time.time() - start) / (i + 1)
                eta_h = sec_per_iter * (total_iters - epoch * iters_per_epoch - i - 1) / 3600
                log.info(f"epoch {epoch} iter {i + 1}/{iters_per_epoch} loss {loss_sum.item() / (i + 1):.4f} "
                         f"lr {lr:.6f} {sec_per_iter:.2f}s/it eta {eta_h:.1f}h")

        record = {"epoch": epoch, "train_loss": loss_sum.item() / iters_per_epoch, "train_time": time.time() - start}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            metrics = evaluate(model, val_loader, device)
            record.update(metrics)
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "mIoU": best_miou},
                           os.path.join(args.save_dir, "best.pth"))
            log.info(f"epoch {epoch} val mIoU {metrics['mIoU']:.2f} mAcc {metrics['mAcc']:.2f} "
                     f"rock IoU {metrics['IoU']['rock']:.2f} best mIoU {best_miou:.2f}")

        with open(os.path.join(args.save_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_miou": best_miou}, latest_path)

if __name__ == "__main__":
    main()
