"""Behavior Cloning agent (stochastic)."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from ice_offline.agent._spec import TorchAgent
from ice_offline.dataset._spec import TorchBuffer

class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd_head = torch.nn.Linear(256, act_size)
        self.min_logstd = -4.0
        self.max_logstd = 15.0

    def dist(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(o)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(self.min_logstd, self.max_logstd)
        a_mean = torch.tanh(mean)
        return a_mean, logstd

    def mode(self, o: torch.Tensor) -> torch.Tensor:
        a_mean, _ = self.dist(o)
        return a_mean

    def sample(self, o: torch.Tensor) -> torch.Tensor:
        a_mean, logstd = self.dist(o)
        return Normal(a_mean, logstd.exp()).rsample().clamp(-1.0, 1.0)

    def forward(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_mean, logstd = self.dist(o)
        return a_mean, logstd

@dataclass
class BCAgentStochastic(TorchAgent):
    obs_size: int
    act_size: int
    learning_rate: float = 1e-3
    device: str = "cpu"

    def __post_init__(self):
        self.policy = _Pi(self.obs_size, self.act_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    def act(self, observation, greedy: bool = True):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.policy.mode(o) if greedy else self.policy.sample(o)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.policy.mode(o) if greedy else self.policy.sample(o)
        return a.cpu().numpy()

    def update(self, batch: TorchBuffer):
        o = batch.obs_list
        a = batch.act_list
        loss = self._loss(o, a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "pi": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.policy.load_state_dict(state["pi"])
        self.optimizer.load_state_dict(state["optimizer"])

    def _loss(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_pred = self.policy.sample(o)
        return F.mse_loss(a_pred, a)

    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._loss(o, a)
