from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from joint_learning.lib.dataset import Batch


class PolicyNetwork(torch.nn.Module):
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
        return self.network(observations)


class TD3Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, max_action: float = 1.0, tau: float = 0.005) -> None:
        super().__init__()
        self.act_size = act_size
        self.max_action = max_action
        self.tau = tau
        self.network = PolicyNetwork(obs_size, act_size)
        self.t_network = PolicyNetwork(obs_size, act_size)
        self.sync_hard()
        for parameter in self.t_network.parameters():
            parameter.requires_grad_(False)

    # ====================
    # Action functions
    # ====================
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.network(observations))

    def sample_uniform(self, observations: torch.Tensor, sample_count: int) -> torch.Tensor:
        batch_size = observations.shape[0]
        return torch.empty(
            (batch_size, sample_count, self.act_size),
            dtype=observations.dtype,
            device=observations.device,
        ).uniform_(-self.max_action, self.max_action)

    def sample_lhs(self, observations: torch.Tensor, sample_count: int) -> torch.Tensor:
        batch_size = observations.shape[0]
        strata = torch.arange(sample_count, device=observations.device, dtype=observations.dtype).view(1, sample_count, 1)
        strata = (strata + torch.rand(batch_size, sample_count, self.act_size, device=observations.device, dtype=observations.dtype)) / sample_count
        permutation = torch.argsort(torch.rand(batch_size, sample_count, self.act_size, device=observations.device), dim=1)
        strata = strata.expand(batch_size, -1, self.act_size).gather(1, permutation)
        return strata.mul(2.0 * self.max_action).sub(self.max_action)

    # ====================
    # Target functions
    # ====================
    def t(self, observations: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.t_network(observations))

    # ====================
    # Parameter functions
    # ====================
    def online_parameters(self):
        return self.network.parameters()

    def sync_hard(self) -> None:
        self.t_network.load_state_dict(self.network.state_dict())

    def sync_soft(self) -> None:
        with torch.no_grad():
            for source, target in zip(self.network.parameters(), self.t_network.parameters()):
                target.copy_(self.tau * source + (1.0 - self.tau) * target)


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


class TD3Critic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, tau: float = 0.005) -> None:
        super().__init__()
        self.tau = tau
        self.q_networks = torch.nn.ModuleList([QNetwork(obs_size, act_size), QNetwork(obs_size, act_size)])
        self.t_networks = torch.nn.ModuleList([QNetwork(obs_size, act_size), QNetwork(obs_size, act_size)])
        self.sync_hard()
        for parameter in self.t_networks.parameters():
            parameter.requires_grad_(False)

    # ====================
    # Q functions
    # ====================
    def q1(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q_networks[0](observations, actions)

    def q_all(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(q_network(observations, actions) for q_network in self.q_networks)

    def q_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(observations, actions), dim=1).min(dim=1, keepdim=True).values

    def q_mean(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(observations, actions), dim=1).mean(dim=1, keepdim=True)

    def q_all_n(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size, sample_count = observations.shape[0], actions.shape[1]
        flat_observations = observations.unsqueeze(1).expand(-1, sample_count, -1).reshape(-1, observations.shape[-1])
        flat_actions = actions.reshape(-1, actions.shape[-1])
        return tuple(q.reshape(batch_size, sample_count, 1) for q in self.q_all(flat_observations, flat_actions))

    # ====================
    # Target functions
    # ====================
    def t_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        values = torch.cat(tuple(network(observations, actions) for network in self.t_networks), dim=1)
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
                    target.copy_(self.tau * source + (1.0 - self.tau) * target)


class TD3Agent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        gamma: float = 0.99,
        sigma_td3: float = 0.2,
        policy_noise_clip: float = 0.5,
        policy_delay: int = 2,
        learning_rate: float = 3e-4,
        max_action: float = 1.0,
        device: str = "cuda",
    ) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.gamma = gamma
        self.policy_delay = policy_delay
        self.update_step = 0
        self.max_action = max_action
        self.sigma_td3 = sigma_td3 * max_action
        self.policy_noise_clip = policy_noise_clip

        self.actor = TD3Actor(obs_size, act_size, max_action).to(device)
        self.critic = TD3Critic(obs_size, act_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.online_parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.online_parameters(), lr=learning_rate)

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
        self.update_step += 1
        # Critic
        loss_critic = self.loss_critic(batch)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        if self.update_step % self.policy_delay == 0:
            # Actor
            loss_actor = self.loss_actor(batch)
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            # Target
            self.actor.sync_soft()
            self.critic.sync_soft()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.network.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.network.load_state_dict(state)
        self.actor.sync_hard()

    # ====================
    # Loss functions
    # ====================
    def target_td3(self, next_observations: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        # \epsilon = clip(N(0, \sigma_(TD3)^2 I), -c, c)
        # a' = clip(\pi^tar(s') + \epsilon, -a_(max), a_(max))
        # y = r + \gamma min_i Q_i^tar(s', a')
        with torch.no_grad():
            next_actions = self.actor.t(next_observations)
            noise = (torch.randn_like(next_actions) * self.sigma_td3).clamp(-self.policy_noise_clip, self.policy_noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            target_q = self.critic.t_min(next_observations, next_actions)
            return rewards + self.gamma * target_q * (1.0 - dones)

    def loss_td(self, batch: Batch) -> torch.Tensor:
        # Loss_TD = E_((s, a, r, s') \sim D) [\sum _(i = 1)^2 (Q_i (s, a) - y)^2]
        observations, actions, rewards, next_observations, dones = batch
        target = self.target_td3(next_observations, rewards, dones)
        q1, q2 = self.critic.q_all(observations, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_TD
        return self.loss_td(batch)

    def loss_opt(self, batch: Batch) -> torch.Tensor:
        # Loss_opt = -E_(s \sim D) [Q_1 (s, \pi(s))]
        observations, _, _, _, _ = batch
        actions = self.actor(observations)
        q = self.critic.q1(observations, actions)
        return -q.mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = Loss_opt
        return self.loss_opt(batch)
