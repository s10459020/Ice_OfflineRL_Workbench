from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from joint_learning.datasets.lib import Batch


class Policy(torch.nn.Module):
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
        self.policy = Policy(obs_size, act_size).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters())

    def act(self, observation) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observations = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.policy(observations)
        return action.cpu().numpy()[0]

    def update(self, batch: Batch) -> None:
        loss_actor = self.loss_actor(batch)
        self.policy_optimizer.zero_grad()
        loss_actor.backward()
        self.policy_optimizer.step()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Behavior Cloning objective:
        #   L_pi = E_{(s,a)~D}[ || pi(s) - a ||_2^2 ]
        # The policy is trained to directly imitate actions in the offline dataset.
        observations, actions, _, _, _ = batch
        predicted_actions = self.policy(observations)
        return F.mse_loss(predicted_actions, actions)
