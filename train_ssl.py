import argparse
import json
import logging
import os
import random
import time
import copy

import numpy as np
import torch
import yaml
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import train

from data import DATA_ROOT, IGNORE_INDEX, S5MarsDataset, eval_transform, get_dataloaders
from evaluate import evaluate
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")

def ssl_loss(student_weak: torch.Tensor, student_strong: torch.Tensor, teacher_weak: torch.Tensor, label: torch.Tensor, cfg):
    # Converting float16 tensors to float32 for loss computation to avoid NaNs
    student_weak = student_weak.float()
    student_strong = student_strong.float()
    teacher_weak = teacher_weak.float()

    unlabeled = label == 255
    labeled = label != 255
    if labeled.any():
        L_sup = F.cross_entropy(student_weak, label, ignore_index=255)
    else:
        L_sup = student_weak.sum() * 0.0

    prob = teacher_weak.softmax(dim=1)
    conf, pseudo = prob.max(dim=1)

    hard_mask = unlabeled & (conf > cfg["t_hard"])
    soft_mask = unlabeled & (conf < cfg["t_soft"])
    num_unlabeled = unlabeled.sum().clamp(min=1)

    log_p = F.log_softmax(student_strong, dim=1)

    hard = F.nll_loss(log_p, pseudo, reduction='none')
    hard = (hard * hard_mask).sum() / num_unlabeled

    soft = -(prob * log_p).sum(dim=1)
    soft = (soft * soft_mask).sum() / num_unlabeled

    return L_sup + cfg['lambda_u'] * (hard + cfg['lambda_s'] * soft)

def ema_momentum(iteration: int, ema_max: float = 0.996):
    # iteration is how many optimizer steps training has taken so far. 
    # It's the same counter that train.py already uses for the learning rate schedule.

    # ema_max is a percentage cap of how much the teacher keeps of itself during training.
    # In S^5Mars, the teacher-student model uses a cap of 0.996
    return min(1 - 1 / (iteration + 1), ema_max)

def ema_update(teacher: torch.nn.Module, student: torch.nn.Module, m: float):
    with torch.no_grad():
        # Applying the update value returned from ema_momentum to the weights of teacher and student
        for teacher_w, student_w in zip(teacher.parameters(), student.parameters()):
            teacher_w.copy_(teacher_w * m + (1 - m) * student_w.detach())

        # Applying the update value returned from ema_momentum to the buffers of teacher and student
        for teacher_b, student_b in zip(teacher.buffers(), student.buffers()):
            teacher_b.copy_(teacher_b * m + (1 - m) * student_b.detach())

def main():
    # First ~30 lines are copied over from train.py to build the student model
    args = parser.parse_args()
    cfg = train.load_config(args.config, args.set)
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

    train_loader, val_loader = train.build_loaders(cfg["data"], pin_memory=device.type == "cuda")
    # Building the student and teacher. Teacher should start as a copy of the student and then is
    # updated with the exponentional moving average (EMA)
    student = build_model(cfg["model"]).to(device)
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()

    optim_cfg, sched = cfg["optim"], cfg["schedule"]
    optimizer = torch.optim.SGD(student.param_groups(optim_cfg["head_lr_mult"]), lr=optim_cfg["lr"],
                                momentum=optim_cfg["momentum"], weight_decay=optim_cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(os.path.join(save_dir, "tb"))

    # Copying the rest of the training loop from train.py, now with student and teacher models
    # infused in the training loop
    start_iter, best_t_miou, best_t_iter, best_s_miou, best_s_iter = 0, 0.0, 0, 0.0, 0
    last_path = os.path.join(save_dir, "last.pth")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        student.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        start_iter, best_t_miou, best_t_iter = ckpt["iter"], ckpt["best_t_miou"], ckpt["best_t_iter"]
        best_s_miou, best_s_iter = ckpt["best_s_miou"], ckpt["best_s_iter"]
        log.info(f"resumed from {last_path} at iter {start_iter}, "
                    f"lr {train.poly_lr(optim_cfg['lr'], start_iter, sched['max_iters'], optim_cfg['poly_power']):.6f}")

    max_iters = sched["max_iters"]
    stop_iter = sched["stop_iter"] or max_iters
    batches = train.forever(train_loader)
    student.train()
    loss_sum, window, window_start = torch.zeros((), device=device), 0, time.time()
    for it in range(start_iter, stop_iter):
        lr = train.poly_lr(optim_cfg["lr"], it, max_iters, optim_cfg["poly_power"])
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_mult"]

        x, y, _ = next(batches)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        x_strong = x
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = ssl_loss(student_weak=student(x), student_strong=student(x_strong), 
                            teacher_weak=teacher(x), label=y, cfg=cfg['ssl'])
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update the teacher
        m = ema_momentum(it, cfg['ssl']['ema_max'])
        ema_update(teacher, student, m)

        loss_sum += loss.detach()
        window += 1
        done = it + 1

        if done % sched["log_interval"] == 0:
            avg_loss = loss_sum.item() / window
            sec_per_iter = (time.time() - window_start) / window
            log.info(f"iter {done}/{max_iters} loss {avg_loss:.4f} lr {lr:.6f} {sec_per_iter:.2f}s/it "
                        f"eta {sec_per_iter * (stop_iter - done) / 3600:.1f}h mem {train.gpu_memory_gb(device):.1f}GB")
            writer.add_scalar("train/loss", avg_loss, done)
            writer.add_scalar("train/lr", lr, done)
            writer.add_scalar("train/sec_per_iter", sec_per_iter, done)
            loss_sum, window, window_start = torch.zeros((), device=device), 0, time.time()

        if done % sched["eval_interval"] == 0 or done == stop_iter:
            s_metrics = evaluate(student, val_loader, device)
            t_metrics = evaluate(teacher, val_loader, device)

            if s_metrics["mIoU"] > best_s_miou:
                best_s_miou, best_s_iter = s_metrics["mIoU"], done
                torch.save({"model": student.state_dict(), "model_cfg": cfg["model"], "iter": done, "mIoU": best_s_miou},
                            os.path.join(save_dir, "student_best.pth"))

            if t_metrics["mIoU"] > best_t_miou:
                best_t_miou, best_t_iter = t_metrics["mIoU"], done
                torch.save({"model": teacher.state_dict(), "model_cfg": cfg["model"], "iter": done, "mIoU": best_t_miou},
                            os.path.join(save_dir, "teacher_best.pth"))
            
            for name in ("mIoU", "mAcc", "aAcc"):
                writer.add_scalar(f"s_val/{name}", s_metrics[name], done)
                writer.add_scalar(f"t_val/{name}", t_metrics[name], done)
            writer.add_scalar("s_val/rock_IoU", s_metrics["IoU"]["rock"], done)
            writer.add_scalar("t_val/rock_IoU", t_metrics["IoU"]["rock"], done)
            writer.add_scalar("s_val/rock_to_bedrock", s_metrics["rock"]["rock_to_bedrock"], done)
            writer.add_scalar("t_val/rock_to_bedrock", t_metrics["rock"]["rock_to_bedrock"], done)

            with open(os.path.join(save_dir, "s_metrics.jsonl"), "a") as f:
                f.write(json.dumps({"iter": done, **s_metrics}) + "\n")
            log.info(f"student iter {done} val mIoU {s_metrics['mIoU']:.2f} mAcc {s_metrics['mAcc']:.2f} "
                        f"rock IoU {s_metrics['IoU']['rock']:.2f} best mIoU {best_s_miou:.2f} at iter {best_s_iter}")

            with open(os.path.join(save_dir, "t_metrics.jsonl"), "a") as f:
                f.write(json.dumps({"iter": done, **t_metrics}) + "\n")
            log.info(f"teacher iter {done} val mIoU {t_metrics['mIoU']:.2f} mAcc {t_metrics['mAcc']:.2f} "
                        f"rock IoU {t_metrics['IoU']['rock']:.2f} best mIoU {best_t_miou:.2f} at iter {best_t_iter}")

            window_start = time.time()

        if done % sched["save_interval"] == 0 or done % sched["eval_interval"] == 0 or done == stop_iter:
            torch.save({"student": student.state_dict(), "teacher": teacher.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "model_cfg": cfg["model"], "iter": done, "best_s_miou": best_s_miou, "best_s_iter": best_s_iter, 
                        "best_t_miou": best_t_miou, "best_t_iter": best_t_iter}, last_path)
            window_start = time.time()

    writer.close()
    log.info(f"Student finished at iter {stop_iter}; best val mIoU {best_s_miou:.2f} at iter {best_s_iter}")
    log.info(f"Teacher finished at iter {stop_iter}; best val mIoU {best_t_miou:.2f} at iter {best_t_iter}")

if __name__ == "__main__":
    main()
