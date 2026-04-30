"""Behavior Cloning continuous agent (deterministic)."""

import numpy as np
import torch
import torch.nn.functional as F
from ice_offline.agent._interface import TorchAgent


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

    def dist(self, o: torch.Tensor) -> torch.Tensor:
        return self.network(o)

    def mode(self, o: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dist(o))

    def sample(self, o: torch.Tensor) -> torch.Tensor:
        return self.mode(o)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.mode(o)


class BCAgentContinuousDeterministic(TorchAgent):
    # ====================
    # Init
    # ====================
    def __init__(self, obs_size: int, act_size: int):
        self.device = "cpu"
        self.learning_rate = 1e-3
        self.expl_noise_std = 0.1

        self.policy = _Pi(obs_size, act_size).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Public API
    # ====================
    def act(self, observation, greedy: bool = True):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.policy.mode(o)
            if not greedy:
                a = (a + torch.randn_like(a) * self.expl_noise_std).clamp(-1.0, 1.0)
        return a.cpu().numpy()[0]

    def update(self, batch):
        observation = batch["obs"]
        action = batch["act"]

        o = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        loss = self._loss(o, a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _save(self) -> dict[str, torch.Tensor]:
        return {
            "pi": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def _load(self, state: dict[str, torch.Tensor]) -> None:
        self.policy.load_state_dict(state["pi"])
        self.optimizer.load_state_dict(state["optimizer"])

    # ====================
    # bc mathmatics
    # ====================
    def _loss(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # loss_BC = E{log pi(a|s)}
        # if pi is gaussian, loss_BC => MSE(a, pi(s))
        a_pred = self.policy.mode(o)
        return F.mse_loss(a_pred, a)

    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._loss(o, a)
