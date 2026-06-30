from pathlib import Path
from typing import Any

import numpy as np
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

    def eval(self, observations: Any, actions: Any, method: str) -> np.ndarray:
        observations_np = np.asarray(observations, dtype=np.float32)
        if observations_np.ndim == 0:
            return np.zeros(1, dtype=np.float32)
        if observations_np.ndim == 1:
            return np.zeros(1, dtype=np.float32)
        return np.zeros(observations_np.shape[0], dtype=np.float32)

    # ====================
    # Training
    # ====================
    def update(self, batch: Batch) -> MetricValues:
        raise NotImplementedError

    def metric_keys(self) -> list[str]:
        return []

    def _value(self, tensor: torch.Tensor) -> float:
        return float(tensor.item())

    def _grad_norm(self, loss: torch.Tensor, params) -> float:
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
        return self._value(value.sqrt())

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
