import torch
import torch.nn.functional as F

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