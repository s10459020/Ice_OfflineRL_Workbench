from pathlib import Path
from typing import Any

import torch

from ice_offline.dataset._types import Batch
from ice_offline.config.paths import model_path


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

    # ====================
    # Persistence
    # ====================
    def save(self, dataset_id: str, step: int = 0) -> Path:
        path = model_path(dataset_id, self.id, step).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._save_dict(), path)
        return path

    def load(self, dataset_id: str, step: int = 0) -> None:
        path = model_path(dataset_id, self.id, step).with_suffix(".pt")
        state = torch.load(path, map_location=self.device)
        self._load_dict(state)

    def _save_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def _load_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError


