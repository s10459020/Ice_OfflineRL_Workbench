import csv

import torch

from ice_offline.config.paths import metric_path


class MetricRecorder:
    def __init__(self, dataset_id: str, agent_id: str) -> None:
        self.path = metric_path(dataset_id, agent_id)
        self.current: dict[str, float] = {}
        self.history: list[dict[str, float]] = []
    
    def add(self, name: str, value: float | torch.Tensor) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.current[name] = value

    def add_grad_norm(self, name: str, loss, params: list) -> None:
        grads = torch.autograd.grad(
            loss, 
            params, 
            retain_graph=True,
        )

        value = sum(
            grad.detach().square().sum()
            for grad in grads
            if grad is not None
        ).sqrt()

        self.add(name, value)
        
    def flush(self) -> None:
        self.history.append(self.current.copy())
        self.current.clear()

    def save(self) -> None:
        names = list(self.history[0])

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", *names])

            for step, line in enumerate(self.history, start=1):
                writer.writerow([
                    step, 
                    *(line[name] for name in names),
                ])

