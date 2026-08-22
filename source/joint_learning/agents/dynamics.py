from pathlib import Path

import torch
from torch.nn import functional as F

from joint_learning.lib.dataset import Batch


class Dynamic(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        device: str,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=-1))

    def update(self, batch: Batch) -> None:
        # L_dyn = E_((s, a, s') \sim D) [\|M(s, a) - s'\|_2^2]
        observations, actions, _, next_observations, _ = batch
        loss = F.mse_loss(
            self(observations, actions),
            next_observations,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load(self, path: Path) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device))

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
