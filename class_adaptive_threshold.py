import torch

class ClassAdaptiveThreshold:
    # Maintain class-wise hard thresholds using relative confidence EMAs.
    # Only pixels selected by valid_mask contribute to the statistics.
    # Warm-up and whether to activate these thresholds are handled by the trainer.

    def __init__(self, num_classes: int, momentum: float = 0.99, alpha: float = 1.0,
        min_threshold: float = 0.4, max_threshold: float = 0.7,
        initial_threshold: float = 0.5,device: str = "cpu",
        ):
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not min_threshold <= initial_threshold <= max_threshold:
            raise ValueError("initial_threshold must be within threshold bounds")

        self.num_classes = num_classes
        self.momentum = momentum
        self.alpha = alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.initial_threshold = initial_threshold
        self.thresholds = torch.full((num_classes,), initial_threshold, dtype=torch.float32, device=device)
        self.confidence_ema = torch.full_like(self.thresholds, initial_threshold)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.global_confidence_ema = torch.tensor(initial_threshold, dtype=torch.float32, device=device)
        self.global_initialized = False

    @torch.no_grad()
    def update(self, confidence: torch.Tensor, prediction: torch.Tensor, valid_mask: torch.Tensor):
        # Update EMAs using teacher predictions before any mixing/augmentation.
        # confidence, prediction, valid_mask: [B, H, W]
        confidence = confidence.detach().float()
        prediction = prediction.detach()
        valid_mask = valid_mask.bool()

        valid_conf = confidence[valid_mask]
        if valid_conf.numel() == 0:
            return

        batch_global_mean = valid_conf.mean()
        if not self.global_initialized:
            self.global_confidence_ema.copy_(batch_global_mean)
            self.global_initialized = True
        else:
            self.global_confidence_ema.lerp_(batch_global_mean, 1.0 - self.momentum)

        for c in range(self.num_classes):
            mask = valid_mask & (prediction == c)
            if not mask.any():
                continue
            class_mean = confidence[mask].mean()
            if not self.initialized[c]:
                self.confidence_ema[c] = class_mean
                self.initialized[c] = True
            else:
                self.confidence_ema[c] = self.momentum * self.confidence_ema[c] + (1.0 - self.momentum) * class_mean

        self.thresholds.fill_(self.initial_threshold)
        observed = self.initialized
        temp = self.initial_threshold + self.alpha * (self.confidence_ema[observed] - self.global_confidence_ema)
        self.thresholds[observed] = temp.clamp(min=self.min_threshold, max=self.max_threshold)

    def get_pixel_thresholds(self, prediction: torch.Tensor) -> torch.Tensor:
        return self.thresholds[prediction]

    def state_dict(self):
        return {
            "thresholds": self.thresholds.clone(),
            "confidence_ema": self.confidence_ema.clone(),
            "initialized": self.initialized.clone(),
            "global_confidence_ema": self.global_confidence_ema.clone(),
            "global_initialized": self.global_initialized,
        }

    def load_state_dict(self, state):
        self.thresholds.copy_(state["thresholds"])
        self.confidence_ema.copy_(state["confidence_ema"])
        self.initialized.copy_(state["initialized"])
        self.global_confidence_ema.copy_(state["global_confidence_ema"])
        self.global_initialized = bool(state["global_initialized"])