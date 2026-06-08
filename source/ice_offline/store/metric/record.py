import csv

import torch

from ice_offline.config.paths import metric_path


class MetricRecorder:
    def __init__(self, dataset_id: str, agent_id: str) -> None:
        self.path = metric_path(dataset_id, agent_id)
        self.current: dict[str, float | None] = {}
        self.history: list[dict[str, float | None]] = []
    
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
        
    def flush(self) -> None:
        self.history.append(self.current.copy())
        self.current.clear()

    def save(self) -> None:
        if not self.history:
            return

        names: list[str] = []
        for line in self.history:
            for name in line:
                if name not in names:
                    names.append(name)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", *names])

            for step, line in enumerate(self.history, start=1):
                writer.writerow([
                    step, 
                    *("" if line.get(name) is None else line.get(name, "") for name in names),
                ])

