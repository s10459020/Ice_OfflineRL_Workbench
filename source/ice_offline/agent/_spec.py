
from typing import Any, Protocol
from pathlib import Path

import torch

from ice_offline.dataset._spec import TorchBuffer
from ice_offline.tools.paths import model_root


AgentBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def agent_batch(batch: TorchBuffer) -> AgentBatch:
    return (
        batch.obs_list,
        batch.act_list,
        batch.rew_list.view(-1, 1),
        batch.done_list.view(-1, 1),
        batch.next_obs_list,
    )


def model_ref(model_id: str, step: int) -> Path:
    return model_root() / model_id / str(step)


class Agent(Protocol):
    agent_name: str

    def act_best(self, observation: Any) -> Any: ...
    def update(self, batch: Any) -> None: ...
    def save(self, model_id: str, step: int = 0) -> Path: ...
    def load(self, model_name: str | Path) -> None: ...


class TorchAgent(Agent):
    device: str

    def _save_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def _load_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    def act_best(self, observation: Any) -> Any:
        return self.act(observation)

    def update(self, batch: Any) -> None:
        raise NotImplementedError

    def save(self, model_id: str, step: int = 0) -> Path:
        path = model_ref(model_id, step).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._save_dict(), path)
        return path

    def load(self, model_name: str | Path) -> None:
        path = Path(model_name).with_suffix(".pt")
        state = torch.load(path, map_location=self.device)
        self._load_dict(state)
