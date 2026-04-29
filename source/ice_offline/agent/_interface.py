from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import torch

from ice_offline.paths import model_root


MODEL_ROOT = model_root()


def model_ref(model_id: str | Path, step: int) -> Path:
    return MODEL_ROOT / Path(model_id) / str(step)


class Agent(Protocol):
    agent_name: str

    def save(self, model_name: str | Path) -> Path: ...

    def load(self, model_name: str | Path) -> None: ...


class TorchAgent:
    device: str

    def _save(self) -> dict[str, Any]:
        raise NotImplementedError

    def _load(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    def save(self, model_name: str | Path) -> Path:
        path = Path(model_name).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._save(), path)
        return path

    def load(self, model_name: str | Path) -> None:
        path = Path(model_name).with_suffix(".pt")
        state = torch.load(path, map_location=self.device)
        self._load(state)
