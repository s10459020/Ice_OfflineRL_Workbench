
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ice_offline.tools.paths import model_root


MODEL_ROOT = model_root()


def model_ref(model_id: str | Path, step: int) -> Path:
    return MODEL_ROOT / Path(model_id) / str(step)


class Agent(Protocol):
    agent_name: str

    def save(self, model_name: str | Path) -> Path: ...

    def load(self, model_name: str | Path) -> None: ...


@dataclass
class EnvSpec:
    observation_shape: tuple[int, ...] | None
    observation_cardinality: tuple[int, ...] | None
    action_shape: tuple[int, ...] | None
    action_cardinality: tuple[int, ...] | None


class TorchAgent:
    device: str

    def configure(
        self,
        env_spec: EnvSpec,
    ) -> None:
        return None

    def set_dim(self, obs_size: int, act_size: int) -> None:
        self.configure(
            EnvSpec(
                observation_shape=(obs_size,),
                observation_cardinality=None,
                action_shape=(1,),
                action_cardinality=(act_size,),
            )
        )

    def _save(self) -> dict[str, Any]:
        raise NotImplementedError

    def _load(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    def act_best(self, observation: Any) -> Any:
        return self.act(observation)

    def save(self, model_name: str | Path) -> Path:
        path = Path(model_name).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._save(), path)
        return path

    def load(self, model_name: str | Path) -> None:
        path = Path(model_name).with_suffix(".pt")
        state = torch.load(path, map_location=self.device)
        self._load(state)
