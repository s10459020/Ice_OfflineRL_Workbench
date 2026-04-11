from pathlib import Path
from typing import Any, Protocol


class Agent(Protocol):
    """Minimal interface required by the generic online trainer."""

    agent_name: str

    def act(self, observation: Any) -> int: ...

    def update(
        self,
        observation: Any,
        action: int,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None: ...

    def save(self, model_id: str | Path) -> Path: ...
