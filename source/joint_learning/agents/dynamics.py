from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from joint_learning.lib.dataset import Batch


class DynamicsNetwork(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + act_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_size),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observations, actions], dim=-1))


class SCASDynamics:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        device: str,
        noise_scale: float = 0.003,
    ):
        self.device = torch.device(device)
        self.noise_scale = noise_scale
        self.model = DynamicsNetwork(obs_size, act_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def next_observation(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # s\hat' = M(s,a)
        return self.model(observations, actions)

    def noisy_observation(self, observations: torch.Tensor) -> torch.Tensor:
        # s\hat = s + \epsilon, \epsilon \sim N(0, \sigma^2 )
        return observations + torch.randn_like(observations) * self.noise_scale

    def update(self, batch: Batch) -> None:
        # Loss_Dynamics = E_D [\|M(s,a)-s'\|_2^2]
        observations, actions, _, next_observations, _ = batch
        loss = F.mse_loss(self.next_observation(observations, actions), next_observations)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def eval(self):
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        return self
