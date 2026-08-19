import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from joint_learning.lib.dataset import Batch


class CQLActor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, min_log_std: float = -20.0, max_log_std: float = 2.0) -> None:
        super().__init__()
        self.act_size = act_size
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean = torch.nn.Linear(256, act_size)
        self.log_std = torch.nn.Linear(256, act_size)

    def dist(self, observations: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.network(observations)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.min_log_std, self.max_log_std)
        return Normal(mean, log_std.exp()), mean

    def log_prob_from_raw(self, dist: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        jacobian = 2.0 * (math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions))
        return (dist.log_prob(raw_actions) - jacobian).sum(dim=-1, keepdim=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        _, mean = self.dist(observations)
        return torch.tanh(mean)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # a = tanh(u), u \sim \pi_(raw) (.|s)
        dist, _ = self.dist(observations)
        raw_actions = dist.rsample()
        return torch.tanh(raw_actions), self.log_prob_from_raw(dist, raw_actions)

    def sample_n(self, observations: torch.Tensor, sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        # a\tilde_(curr,n) \sim \pi(. | s) or a\tilde_(next,n) \sim \pi(. | s')
        dist, _ = self.dist(observations)
        raw_actions = dist.rsample((sample_count,))
        actions = torch.tanh(raw_actions).transpose(0, 1)
        log_probs = self.log_prob_from_raw(dist, raw_actions).transpose(0, 1)
        return actions, log_probs

    def sample_uniform(self, observations: torch.Tensor, sample_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        # a\tilde_(rand,n) \sim U(A), log \mu_(rand)(a\tilde_(rand,n)) = A log 0.5
        batch_size = observations.shape[0]
        actions = torch.empty(
            batch_size,
            sample_count,
            self.act_size,
            device=observations.device,
            dtype=observations.dtype,
        ).uniform_(-1.0, 1.0)
        log_probs = torch.full(
            (batch_size, sample_count, 1),
            math.log(0.5 ** self.act_size),
            device=observations.device,
            dtype=observations.dtype,
        )
        return actions, log_probs


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


class CQLCritic(torch.nn.Module):
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

    def q_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(observations, actions), dim=1).min(dim=1, keepdim=True).values

    # ====================
    # Target functions
    # ====================
    def t_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            [q_network(observations, actions) for q_network in self.t_networks],
            dim=1,
        )
        return values.min(dim=1, keepdim=True).values

    def q_all_n(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size = observations.shape[0]
        sample_count = actions.shape[1]
        repeated_observations = observations.repeat_interleave(sample_count, dim=0)
        flat_actions = actions.reshape(-1, actions.shape[-1])
        return tuple(
            q.reshape(batch_size, sample_count, 1)
            for q in self.q_all(repeated_observations, flat_actions)
        )

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


class CQLAgent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        discount_factor: float = 0.99,
        conservative_sample_count: int = 10,
        initial_temperature: float = 1.0,
        learning_rate: float = 3e-4,
        device: str = "cuda",
    ) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = discount_factor
        self.target_entropy = -float(act_size)
        self.lambda_cql = 5.0
        self.conservative_sample_count = conservative_sample_count

        self.actor = CQLActor(obs_size, act_size).to(device)
        self.critic = CQLCritic(obs_size, act_size).to(device)

        self.log_temperature = torch.nn.Parameter(torch.full((1, 1), math.log(initial_temperature), dtype=torch.float32, device=device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.online_parameters(), lr=learning_rate)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=learning_rate)

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
        # Temperature
        loss_temperature = self.loss_temperature(batch)
        self.temperature_optimizer.zero_grad()
        loss_temperature.backward()
        self.temperature_optimizer.step()

        loss_conservative = self.loss_conservative(batch)

        # Critic
        loss_td = self.loss_td(batch)
        # Loss_Critic = Loss_TD + lambda_cql Loss_conservative
        loss = loss_td + self.lambda_cql * loss_conservative
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # Actor
        loss_actor = self.loss_actor(batch)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

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
    def temperature(self) -> torch.Tensor:
        # \alpha = exp(log \alpha)
        return self.log_temperature.exp()

    def loss_td(self, batch: Batch) -> torch.Tensor:
        # a' \sim \pi(. | s')
        # y = r + \gamma [Q_(min)' (s', a') - \alpha log \pi(a' | s')]
        # Loss_TD = E_((s, a, r, s') \sim D) [\sum _(i = 1)^2 (Q_i (s, a) - y)^2]
        observations, actions, rewards, next_observations, dones = batch
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_observations)
            target_q = self.critic.t_min(next_observations, next_actions)
            target = rewards + self.discount_factor * (
                target_q - self.temperature() * next_log_probs
            ) * (1.0 - dones)
        q1, q2 = self.critic.q_all(observations, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def loss_conservative(self, batch: Batch) -> torch.Tensor:
        # a\tilde_(rand,n) \sim U(A)
        # a\tilde_(curr,n) \sim \pi(. | s)
        # a\tilde_(next,n) \sim \pi(. | s')
        # z_(i,rand,n) = Q_i (s, a\tilde_(rand,n)) - log \mu_(rand) (a\tilde_(rand,n))
        # z_(i,curr,n) = Q_i (s, a\tilde_(curr,n)) - log \pi(a\tilde_(curr,n) | s)
        # z_(i,next,n) = Q_i (s, a\tilde_(next,n)) - log \pi(a\tilde_(next,n) | s')
        # Loss_conservative = \sum _(i = 1)^2 [E_(s \sim D) [log \sum_n exp({z_(i,rand,n), z_(i,curr,n), z_(i,next,n)})] - E_((s, a) \sim D) [Q_i (s, a)]]
        observations, actions, _, next_observations, _ = batch
        with torch.no_grad():
            policy_actions, policy_log_probs = self.actor.sample_n(observations, self.conservative_sample_count)
            next_policy_actions, next_policy_log_probs = self.actor.sample_n(next_observations, self.conservative_sample_count)
            random_actions, random_log_probs = self.actor.sample_uniform(observations, self.conservative_sample_count)
            log_probs = torch.cat([policy_log_probs, next_policy_log_probs, random_log_probs], dim=1)

        policy_q_values = self.critic.q_all_n(observations, policy_actions)
        next_policy_q_values = self.critic.q_all_n(observations, next_policy_actions)
        random_q_values = self.critic.q_all_n(observations, random_actions)
        data_q_values = self.critic.q_all(observations, actions)
        losses = []
        for policy_q, next_policy_q, random_q, data_q in zip(
            policy_q_values,
            next_policy_q_values,
            random_q_values,
            data_q_values,
        ):
            candidates = torch.cat([policy_q, next_policy_q, random_q], dim=1)
            logsumexp = torch.logsumexp(candidates - log_probs, dim=1).mean()
            losses.append(logsumexp - data_q.mean())
        return sum(losses)

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = E_(s \sim D, a \sim \pi) [\alpha log \pi(a | s) - Q_(min) (s, a)]
        observations, _, _, _, _ = batch
        actions, log_probs = self.actor.sample(observations)
        q = self.critic.q_min(observations, actions)
        return (self.temperature().detach() * log_probs - q).mean()

    def loss_temperature(self, batch: Batch) -> torch.Tensor:
        # Loss_Temperature = -E_(s \sim D, a \sim \pi) [log \alpha(log \pi(a | s) + H_t)]
        observations, _, _, _, _ = batch
        with torch.no_grad():
            _, log_probs = self.actor.sample(observations)
        return -(self.log_temperature * (log_probs + self.target_entropy).detach()).mean()
