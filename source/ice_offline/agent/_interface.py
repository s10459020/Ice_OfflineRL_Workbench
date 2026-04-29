from pathlib import Path
from typing import Any, Protocol

from ice_offline.paths import model_root


MODEL_ROOT = model_root()


def model_name(step: int, suffix: str) -> str:
    return f"model_{step}{suffix}"


def model_path(model_id: str | Path, step: int, suffix: str) -> Path:
    return MODEL_ROOT / Path(model_id) / model_name(step, suffix)


class Agent(Protocol):
    """Minimal interface required by the generic online trainer."""

    agent_name: str

    def act(self, observation: Any) -> int: ...

    def update(self, observation: Any, action: int, reward: float, next_observation: Any, done: bool,) -> None: ...

    def save(self, model_id: str | Path) -> Path: ...
