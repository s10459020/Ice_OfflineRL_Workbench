"""Behavior Cloning agent (stochastic)."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from ice_offline.agent._spec import Agent
from ice_offline.dataset._types import Batch

class _Pi(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        min_logstd: float = -4.0,
        max_logstd: float = 15.0,
    ):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd_head = torch.nn.Linear(256, act_size)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def dist(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(o)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(self.min_logstd, self.max_logstd)
        return torch.tanh(mean), logstd


class _Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.pi = _Pi(obs_size, act_size)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        mean, _ = self.pi.dist(o)
        return mean

    def sample(self, o: torch.Tensor) -> torch.Tensor:
        mean, logstd = self.pi.dist(o)
        return Normal(mean, logstd.exp()).rsample().clamp(-1.0, 1.0)


@dataclass
class BCStochasticAgent(Agent):
    agent_name: ClassVar[str] = "bc_stochastic"
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
    def act(self, observation, greedy: bool = True):
        o_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(o_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor(o) if greedy else self.actor.sample(o)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        o_np = np.asarray(observation_batch, dtype=np.float32)
        o = torch.as_tensor(o_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor(o) if greedy else self.actor.sample(o)
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
        # Stochastic BC: sample from the actor and minimize imitation error.
        a_pred = self.actor.sample(o)
        return F.mse_loss(a_pred, a)


