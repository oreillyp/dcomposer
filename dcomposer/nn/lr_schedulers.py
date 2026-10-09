import math

import torch


class LinearWarmupCosineDecayLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 0,
        decay_steps: int = 100_000,
        min_scale: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = max(int(decay_steps), 1)
        self.min_scale = float(min_scale)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _scale(self, step: int) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return float(step) / float(self.warmup_steps)

        decay_step = min(max(step - self.warmup_steps, 0), self.decay_steps)
        cosine = 0.5 * (
            1.0 + math.cos(math.pi * float(decay_step) / float(self.decay_steps))
        )
        return self.min_scale + (1.0 - self.min_scale) * cosine

    def get_lr(self):
        step = self.last_epoch
        scale = self._scale(step)
        return [base_lr * scale for base_lr in self.base_lrs]
