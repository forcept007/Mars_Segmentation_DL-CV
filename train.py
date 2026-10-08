import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import DATA_ROOT, IGNORE_INDEX, S5MarsDataset, eval_transform, get_dataloaders
from evaluate import evaluate
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")

def merge(base, override):
    out = dict(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out

def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        cfg = merge(load_config(os.path.join(os.path.dirname(path), base)), cfg)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg

def poly_lr(base_lr, it, max_iters, power=0.9):
    return base_lr * (1 - it / max_iters) ** power

def build_loaders(cfg, pin_memory):
    root = cfg["root"] or DATA_ROOT
    if cfg["overfit"]:
        subset = S5MarsDataset("train", eval_transform(512), root, limit=cfg["overfit"])
        args = dict(num_workers=cfg["num_workers"], pin_memory=pin_memory, persistent_workers=cfg["num_workers"] > 0)
        return (DataLoader(subset, cfg["batch_size"], shuffle=True, drop_last=True, **args),
                DataLoader(subset, cfg["eval_batch_size"], **args))
    train_loader, val_loader, _ = get_dataloaders(cfg["batch_size"], cfg["eval_batch_size"], cfg["num_workers"],
                                                  root, tuple(cfg["scale_range"]), pin_memory)
    return train_loader, val_loader

def forever(loader):
    while True:
        yield from loader

def gpu_memory_gb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2 ** 30
    return 0.0

def main():
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    save_dir = cfg.get("save_dir") or os.path.join("runs", os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(save_dir, "train.log")), logging.StreamHandler()])
    log = logging.getLogger()
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log.info(json.dumps(cfg))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    use_amp = cfg["amp"] and device.type == "cuda"
    log.info(f"device {device}, amp {use_amp}")

    train_loader, val_loader = build_loaders(cfg["data"], pin_memory=device.type == "cuda")
    model = build_model(cfg["model"]).to(device)
    optim_cfg, sched = cfg["optim"], cfg["schedule"]
    optimizer = torch.optim.SGD(model.param_groups(optim_cfg["head_lr_mult"]), lr=optim_cfg["lr"],
                                momentum=optim_cfg["momentum"], weight_decay=optim_cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    writer = SummaryWriter(os.path.join(save_dir, "tb"))

    start_iter, best_miou, best_iter = 0, 0.0, 0
    last_path = os.path.join(save_dir, "last.pth")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        start_iter, best_miou, best_iter = ckpt["iter"], ckpt["best_miou"], ckpt["best_iter"]
        log.info(f"resumed from {last_path} at iter {start_iter}, "
                 f"lr {poly_lr(optim_cfg['lr'], start_iter, sched['max_iters'], optim_cfg['poly_power']):.6f}")

    max_iters = sched["max_iters"]
    stop_iter = sched["stop_iter"] or max_iters
    batches = forever(train_loader)
    model.train()
    loss_sum, window, window_start = torch.zeros((), device=device), 0, time.time()
    for it in range(start_iter, stop_iter):
        lr = poly_lr(optim_cfg["lr"], it, max_iters, optim_cfg["poly_power"])
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_mult"]

        x, y, _ = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.detach()
        window += 1
        done = it + 1

        if done % sched["log_interval"] == 0:
            avg_loss = loss_sum.item() / window
            sec_per_iter = (time.time() - window_start) / window
            log.info(f"iter {done}/{max_iters} loss {avg_loss:.4f} lr {lr:.6f} {sec_per_iter:.2f}s/it "
                     f"eta {sec_per_iter * (stop_iter - done) / 3600:.1f}h mem {gpu_memory_gb(device):.1f}GB")
            writer.add_scalar("train/loss", avg_loss, done)
            writer.add_scalar("train/lr", lr, done)
            writer.add_scalar("train/sec_per_iter", sec_per_iter, done)
            loss_sum, window, window_start = torch.zeros((), device=device), 0, time.time()

        if done % sched["eval_interval"] == 0 or done == stop_iter:
            metrics = evaluate(model, val_loader, device)
            if metrics["mIoU"] > best_miou:
                best_miou, best_iter = metrics["mIoU"], done
                torch.save({"model": model.state_dict(), "model_cfg": cfg["model"], "iter": done, "mIoU": best_miou},
                           os.path.join(save_dir, "best.pth"))
            for name in ("mIoU", "mAcc", "aAcc"):
                writer.add_scalar(f"val/{name}", metrics[name], done)
            writer.add_scalar("val/rock_IoU", metrics["IoU"]["rock"], done)
            writer.add_scalar("val/rock_to_bedrock", metrics["rock"]["rock_to_bedrock"], done)
            with open(os.path.join(save_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({"iter": done, **metrics}) + "\n")
            log.info(f"iter {done} val mIoU {metrics['mIoU']:.2f} mAcc {metrics['mAcc']:.2f} "
                     f"rock IoU {metrics['IoU']['rock']:.2f} best mIoU {best_miou:.2f} at iter {best_iter}")
            window_start = time.time()

        if done % sched["save_interval"] == 0 or done % sched["eval_interval"] == 0 or done == stop_iter:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "model_cfg": cfg["model"], "iter": done, "best_miou": best_miou, "best_iter": best_iter}, last_path)
            window_start = time.time()

    writer.close()
    log.info(f"finished at iter {stop_iter}; best val mIoU {best_miou:.2f} at iter {best_iter}")

if __name__ == "__main__":
    main()
