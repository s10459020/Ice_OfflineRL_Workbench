from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from ice_offline.joint_learning.dataset import Batch


class Policy(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int) -> None:
        super().__init__()
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean = torch.nn.Linear(256, act_size)
        self.logstd = torch.nn.Parameter(torch.zeros(1, act_size, dtype=torch.float32))
        self.min_logstd = -5.0
        self.max_logstd = 2.0

    def dist(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden(observations)
        mean = torch.tanh(self.mean(hidden))
        logstd = self.min_logstd + torch.sigmoid(self.logstd) * (self.max_logstd - self.min_logstd)
        return mean, logstd

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = self.dist(observations)
        return mean

    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, logstd = self.dist(observations)
        return Normal(mean, logstd.exp()).log_prob(actions).sum(dim=-1, keepdim=True)


class QNetwork(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observations, actions], dim=1))


class VNetwork(torch.nn.Module):
    def __init__(self, obs_size: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class IQLAgent:
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = 0.99
        self.target_update_rate = 0.005
        self.expectile = 0.7
        self.advantage_scale = 3.0
        self.cap_weight = 100.0

        self.policy = Policy(obs_size, act_size).to(device)
        self.q1 = QNetwork(obs_size, act_size).to(device)
        self.q2 = QNetwork(obs_size, act_size).to(device)
        self.target_q1 = QNetwork(obs_size, act_size).to(device)
        self.target_q2 = QNetwork(obs_size, act_size).to(device)
        self.value_function = VNetwork(obs_size).to(device)
        self.sync_target_hard()

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters())
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()))
        self.value_optimizer = torch.optim.Adam(self.value_function.parameters())

    def act(self, observation) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observations = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.policy(observations)
        return action.cpu().numpy()[0]

    def update(self, batch: Batch) -> None:
        loss_q = self.loss_q(batch)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        loss_v = self.loss_v(batch)
        self.value_optimizer.zero_grad()
        loss_v.backward()
        self.value_optimizer.step()

        loss_actor = self.loss_actor(batch)
        self.policy_optimizer.zero_grad()
        loss_actor.backward()
        self.policy_optimizer.step()
        self.sync_target_soft()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state)

    def sync_target_hard(self) -> None:
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    def sync_target_soft(self) -> None:
        with torch.no_grad():
            for source, target in zip(self.q1.parameters(), self.target_q1.parameters()):
                target.data.copy_(self.target_update_rate * source.data + (1.0 - self.target_update_rate) * target.data)
            for source, target in zip(self.q2.parameters(), self.target_q2.parameters()):
                target.data.copy_(self.target_update_rate * source.data + (1.0 - self.target_update_rate) * target.data)

    def target_q_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1 = self.target_q1(observations, actions)
        q2 = self.target_q2(observations, actions)
        return torch.minimum(q1, q2)

    def loss_q(self, batch: Batch) -> torch.Tensor:
        # IQL Q objective:
        #   y = r + gamma * (1 - d) * V(s')
        #   L_Q = E_D[ (Q_1(s,a) - y)^2 + (Q_2(s,a) - y)^2 ]
        # The value function supplies the bootstrapped target instead of a policy action.
        observations, actions, rewards, next_observations, dones = batch
        with torch.no_grad():
            target = rewards + self.discount_factor * self.value_function(next_observations) * (1.0 - dones)
        q1 = self.q1(observations, actions)
        q2 = self.q2(observations, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def loss_v(self, batch: Batch) -> torch.Tensor:
        # IQL expectile value objective:
        #   u = min_i Q_i_target(s,a) - V(s)
        #   L_V = E_D[ |tau - 1(u < 0)| * u^2 ]
        # Expectile regression fits V to high-value in-dataset actions without querying OOD actions.
        observations, actions, _, _, _ = batch
        with torch.no_grad():
            q = self.target_q_min(observations, actions)
        v = self.value_function(observations)
        diff = q - v
        weight = torch.abs(self.expectile - (diff < 0.0).float())
        return (weight * diff.pow(2)).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # IQL advantage-weighted behavior cloning:
        #   A(s,a) = min_i Q_i_target(s,a) - V(s)
        #   w = exp(beta * A(s,a))
        #   L_pi = E_D[ -clip(w, max=w_max) * log pi(a|s) ]
        # The actor imitates dataset actions more strongly when their estimated advantage is high.
        observations, actions, _, _, _ = batch
        with torch.no_grad():
            advantage = self.target_q_min(observations, actions) - self.value_function(observations)
            weight = (self.advantage_scale * advantage).exp().clamp(max=self.cap_weight)
        log_prob = self.policy.log_prob(observations, actions)
        return -(weight * log_prob).mean()
