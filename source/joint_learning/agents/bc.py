from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from joint_learning.lib.dataset import Batch


class BCActor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observations))


class BCAgent:
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.actor = BCActor(obs_size, act_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

    # -------------------------------------------------------------------------
    # Public functions
    # -------------------------------------------------------------------------
    def act(self, observation) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observations = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(observations)
        return action.cpu().numpy()[0]

    def update(self, batch: Batch) -> None:
        # Actor
        loss_actor = self.loss_actor(batch)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state)

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_bc(self, batch: Batch) -> torch.Tensor:
        # Loss_BC = E_((s,a)\sim D) [\|\pi(s)-a\|_2^2]
        observations, actions, _, _, _ = batch
        predicted_actions = self.actor(observations)
        return F.mse_loss(predicted_actions, actions)

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = Loss_BC
        return self.loss_bc(batch)
