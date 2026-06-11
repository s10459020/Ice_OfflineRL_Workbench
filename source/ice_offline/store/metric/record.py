import csv

import torch

from ice_offline.config.paths import metric_path


class MetricRecorder:
    def __init__(self, task_id: str, initialized = False) -> None:
        self.path = metric_path(task_id)
        self.current: dict[str, float | None] = {}
        self.initialized = initialized
        self.last = {}

    def add(self, name: str, value: float | torch.Tensor | None) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.current[name] = value

    def add_grad_norm(self, name: str, loss: torch.Tensor, params) -> None:
        params = list(params)
        grads = torch.autograd.grad(
            loss, 
            params, 
            retain_graph=True,
            allow_unused=True,
        )

        value = torch.zeros((), device=loss.device)
        for grad in grads:
            if grad is not None:
                value = value + grad.detach().square().sum()
        value = value.sqrt()

        self.add(name, value)
        
    def flush(self, step) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.initialized else "w"
        
        with self.path.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not self.initialized:
                writer.writerow(["step", *self.current.keys()])
                self.initialized = True
            writer.writerow([step, *self.current.values()])

        self.last = self.current.copy()
        self.current.clear()
