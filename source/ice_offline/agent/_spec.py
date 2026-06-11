from pathlib import Path
from typing import Any

import torch

from ice_offline.dataset._types import Batch
from ice_offline.config.paths import model_path


MetricValues = dict[str, float | torch.Tensor | None]


class Agent:
    id: str
    device: str

    # ====================
    # Acting
    # ====================
    def act_best(self, observation: Any) -> Any:
        return self.act(observation)

    def set_seed(self, seed: int) -> None:
        pass

    # ====================
    # Training
    # ====================
    def update(self, batch: Batch) -> None:
        raise NotImplementedError

    def update_with_metrics(self, batch: Batch) -> MetricValues:
        self.update(batch)
        return {}

    def _grad_norm(self, loss: torch.Tensor, params) -> torch.Tensor:
        params = [p for p in params if p.requires_grad]
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
        return value.sqrt()

    # ====================
    # Persistence
    # ====================
    def save(self, task_id: str, step: int = 0) -> Path:
        path = model_path(task_id, step).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._save_dict(), path)
        return path

    def load(self, task_id: str, step: int = 0) -> None:
        path = model_path(task_id, step).with_suffix(".pt")
        state = torch.load(path, map_location=self.device)
        self._load_dict(state)

    def _save_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def _load_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError
