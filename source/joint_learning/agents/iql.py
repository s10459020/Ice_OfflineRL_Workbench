from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from joint_learning.lib.dataset import Batch


class IQLActor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, min_logstd: float = -5.0, max_logstd: float = 2.0) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean = torch.nn.Linear(256, act_size)
        self.logstd = torch.nn.Parameter(torch.zeros(1, act_size, dtype=torch.float32))
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def dist(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.network(observations)
        mean = torch.tanh(self.mean(features))
        logstd = self.logstd.clamp(self.min_logstd, self.max_logstd)
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


class IQLCritic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, target_update_rate: float = 0.005) -> None:
        super().__init__()
        self.target_update_rate = target_update_rate
        self.q_networks = torch.nn.ModuleList([QNetwork(obs_size, act_size), QNetwork(obs_size, act_size)])
        self.t_networks = torch.nn.ModuleList([QNetwork(obs_size, act_size), QNetwork(obs_size, act_size)])
        self.sync_hard()
        for parameter in self.t_networks.parameters():
            parameter.requires_grad_(False)

    # ====================
    # Q functions
    # ====================
    def q_all(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(q_network(observations, actions) for q_network in self.q_networks)

    # ====================
    # Target functions
    # ====================
    def t_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            [q_network(observations, actions) for q_network in self.t_networks],
            dim=1,
        )
        return values.min(dim=1, keepdim=True).values

    # ====================
    # Parameter functions
    # ====================
    def online_parameters(self):
        return self.q_networks.parameters()

    def sync_hard(self) -> None:
        for source, target in zip(self.q_networks, self.t_networks):
            target.load_state_dict(source.state_dict())

    def sync_soft(self) -> None:
        with torch.no_grad():
            for source_network, target_network in zip(self.q_networks, self.t_networks):
                for source, target in zip(source_network.parameters(), target_network.parameters()):
                    target.data.copy_(
                        self.target_update_rate * source.data
                        + (1.0 - self.target_update_rate) * target.data
                    )


class IQLValue(torch.nn.Module):
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
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        discount_factor: float = 0.99,
        expectile: float = 0.7,
        advantage_scale: float = 3.0,
        cap_weight: float = 100.0,
        lr: float = 3e-4,
        device: str = "cuda"
    ) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = discount_factor
        self.expectile = expectile
        self.advantage_scale = advantage_scale
        self.cap_weight = cap_weight

        self.actor = IQLActor(obs_size, act_size).to(device)
        self.critic = IQLCritic(obs_size, act_size).to(device)
        self.value = IQLValue(obs_size).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.online_parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

    # ====================
    # Public functions
    # ====================
    def act(self, observation) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observations = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(observations)
        return action.cpu().numpy()[0]

    def update(self, batch: Batch) -> None:
        # Value
        loss_value = self.loss_value(batch)
        self.value_optimizer.zero_grad()
        loss_value.backward()
        self.value_optimizer.step()

        # Actor
        loss_actor = self.loss_actor(batch)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Critic
        loss_critic = self.loss_critic(batch)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Target
        self.critic.sync_soft()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state)

    # ====================
    # Loss functions
    # ====================
    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # y = r + \gamma(1 - d)V(s')
        # Loss_Critic = E_((s, a, r, s', d) \sim D) [\sum _(i = 1)^2 (Q_i (s, a) - y)^2]
        observations, actions, rewards, next_observations, dones = batch
        with torch.no_grad():
            target = rewards + self.discount_factor * self.value(next_observations) * (1.0 - dones)
        q1, q2 = self.critic.q_all(observations, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def loss_value(self, batch: Batch) -> torch.Tensor:
        # u(s, a) = Q_(min) ^target (s, a) - V(s)
        # Loss_Value = E_((s, a) \sim D) [|\tau - I(u < 0)|u^2]
        observations, actions, _, _, _ = batch
        with torch.no_grad():
            q = self.critic.t_min(observations, actions)
        v = self.value(observations)
        diff = q - v
        weight = torch.abs(self.expectile - (diff < 0.0).float())
        return (weight * diff.pow(2)).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # A(s, a) = Q_(min) ^target (s, a) - V(s)
        # w(s, a) = min(exp(\beta A(s, a)), w_(max))
        # Loss_Actor = -E_((s, a) \sim D) [w(s, a)log \pi(a | s)]
        observations, actions, _, _, _ = batch
        with torch.no_grad():
            advantage = self.critic.t_min(observations, actions) - self.value(observations)
            weight = (self.advantage_scale * advantage).exp().clamp(max=self.cap_weight)
        log_prob = self.actor.log_prob(observations, actions)
        return -(weight * log_prob).mean()
