from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from ice_offline.agent._spec import Agent
from ice_offline.dataset._types import Batch


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(o))


class _Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.pi = _Pi(obs_size, act_size)


@dataclass
class BCDeterministicAgent(Agent):
    obs_size: int
    act_size: int
    learning_rate: float = 1e-3
    device: str = "cpu"

    # ====================
    # Init
    # ====================
    def __post_init__(self):
        self.actor = _Actor(self.obs_size, self.act_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Act
    # ====================
    def act(self, observation):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(o)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(o)
        return a.cpu().numpy()

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch):
        o, a, _, _, _ = batch
        self.update_actor(o, a)

    def update_actor(self, o: torch.Tensor, a: torch.Tensor) -> None:
        self.actor_optimizer.zero_grad()
        loss = self.loss_actor(o, a)
        loss.backward()
        self.actor_optimizer.step()

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])

    # ====================
    # Actor loss
    # ====================
    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # BC: minimize imitation error on dataset actions.
        a_pred = self.actor.pi(o)
        return F.mse_loss(a_pred, a)

