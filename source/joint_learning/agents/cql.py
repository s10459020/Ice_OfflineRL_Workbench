import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from joint_learning.lib.dataset import Batch


class CQLActor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int) -> None:
        super().__init__()
        self.act_size = act_size
        self.n_samples = 10
        self.min_logstd = -20.0
        self.max_logstd = 2.0
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean = torch.nn.Linear(256, act_size)
        self.logstd = torch.nn.Linear(256, act_size)

    def dist(self, observations: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        hidden = self.hidden(observations)
        mean = self.mean(hidden)
        logstd = self.logstd(hidden).clamp(self.min_logstd, self.max_logstd)
        return Normal(mean, logstd.exp()), mean

    def log_prob_from_raw(self, dist: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        jacobian = 2.0 * (math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions))
        return (dist.log_prob(raw_actions) - jacobian).sum(dim=-1, keepdim=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        _, mean = self.dist(observations)
        return torch.tanh(mean)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # a=tanh(u), u \sim \pi_(raw) (.|s)
        dist, _ = self.dist(observations)
        raw_actions = dist.rsample()
        return torch.tanh(raw_actions), self.log_prob_from_raw(dist, raw_actions)

    def sample_n(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # a\tilde_n \sim \pi(.|s)
        dist, _ = self.dist(observations)
        raw_actions = dist.rsample((self.n_samples,))
        actions = torch.tanh(raw_actions).transpose(0, 1)
        log_probs = self.log_prob_from_raw(dist, raw_actions).transpose(0, 1)
        return actions, log_probs

    def sample_random_n(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # a\tilde_n \sim U(-1, 1), log \mu(a\tilde_n|s) = A log 0.5
        batch_size = observations.shape[0]
        actions = torch.empty(
            batch_size,
            self.n_samples,
            self.act_size,
            device=observations.device,
            dtype=observations.dtype,
        ).uniform_(-1.0, 1.0)
        log_probs = torch.full(
            (batch_size, self.n_samples, 1),
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
    def __init__(self, obs_size: int, act_size: int, target_update_rate: float) -> None:
        super().__init__()
        self.target_update_rate = target_update_rate
        self.q_networks = torch.nn.ModuleList([QNetwork(obs_size, act_size), QNetwork(obs_size, act_size)])
        self.target_q_networks = torch.nn.ModuleList([QNetwork(obs_size, act_size), QNetwork(obs_size, act_size)])
        self.sync_hard()

    def q_all(self, observations: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(q_network(observations, actions) for q_network in self.q_networks)

    def q_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(observations, actions), dim=1).min(dim=1, keepdim=True).values

    def target_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            [q_network(observations, actions) for q_network in self.target_q_networks],
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

    def online_parameters(self):
        return self.q_networks.parameters()

    def sync_hard(self) -> None:
        for source, target in zip(self.q_networks, self.target_q_networks):
            target.load_state_dict(source.state_dict())

    def sync_soft(self) -> None:
        with torch.no_grad():
            for source_network, target_network in zip(self.q_networks, self.target_q_networks):
                for source, target in zip(source_network.parameters(), target_network.parameters()):
                    target.data.copy_(
                        self.target_update_rate * source.data
                        + (1.0 - self.target_update_rate) * target.data
                    )


class CQLAgent:
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = 0.99
        self.target_update_rate = 0.005
        self.target_entropy = -float(act_size)
        self.threshold = 2.0

        self.actor = CQLActor(obs_size, act_size).to(device)
        self.critic = CQLCritic(obs_size, act_size, self.target_update_rate).to(device)

        self.log_alpha = torch.nn.Parameter(torch.full((1, 1), math.log(1.0), dtype=torch.float32, device=device))
        self.log_cql_lambda = torch.nn.Parameter(torch.full((1, 1), math.log(10.0), dtype=torch.float32, device=device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.online_parameters())
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha])
        self.cql_lambda_optimizer = torch.optim.Adam([self.log_cql_lambda])

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
        loss_td = self.loss_td(batch)
        loss_conservative = self.loss_conservative(batch)

        # Loss_weight = -\lambda'_(CQL) (Loss_conservative-\tau_(CQL))
        loss_weight = -(self.cql_lambda() * (loss_conservative.detach() - self.threshold)).mean()
        self.cql_lambda_optimizer.zero_grad()
        loss_weight.backward()
        self.cql_lambda_optimizer.step()

        loss = loss_td + self.cql_lambda().detach() * loss_conservative
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        loss_actor = self.loss_actor(batch)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        loss_temperature = self.loss_temperature(batch)
        self.alpha_optimizer.zero_grad()
        loss_temperature.backward()
        self.alpha_optimizer.step()
        self.critic.sync_soft()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state)

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def alpha(self) -> torch.Tensor:
        # \alpha = exp(log \alpha)
        return self.log_alpha.exp()

    def cql_lambda(self) -> torch.Tensor:
        # \lambda'_(CQL) = clip(exp(log \lambda'_(CQL)), 0, 10^6 )
        return self.log_cql_lambda.exp().clamp(0.0, 1e6)

    def loss_td(self, batch: Batch) -> torch.Tensor:
        # a' \sim \pi(.|s')
        # y=r+\gamma(1-d)[Q_(min) ' (s',a')-\alpha log \pi(a'|s')]
        # Loss_TD=E_D [\sum _(i \in {1,2}) (Q_i (s,a)-y)^2]
        observations, actions, rewards, next_observations, dones = batch
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_observations)
            target_q = self.critic.target_min(next_observations, next_actions)
            target = rewards + self.discount_factor * (target_q - self.alpha() * next_log_probs) * (1.0 - dones)
        q1, q2 = self.critic.q_all(observations, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def loss_conservative(self, batch: Batch) -> torch.Tensor:
        # Loss_conservative=\sum _(i=1)^2 [E_(s~D) [log \sum_n exp(Q_i (s,a\tilde_n)-log \mu(a\tilde_n|s))]-E_((s,a)~D) [Q_i (s,a)]]
        observations, actions, _, next_observations, _ = batch
        with torch.no_grad():
            policy_actions, policy_log_probs = self.actor.sample_n(observations)
            next_policy_actions, next_policy_log_probs = self.actor.sample_n(next_observations)
            random_actions, random_log_probs = self.actor.sample_random_n(observations)
            log_probs = torch.cat([policy_log_probs, next_policy_log_probs, random_log_probs], dim=1)

        q1_policy, q2_policy = self.critic.q_all_n(observations, policy_actions)
        q1_next_policy, q2_next_policy = self.critic.q_all_n(observations, next_policy_actions)
        q1_random, q2_random = self.critic.q_all_n(observations, random_actions)

        q1_candidates = torch.cat([q1_policy, q1_next_policy, q1_random], dim=1)
        q2_candidates = torch.cat([q2_policy, q2_next_policy, q2_random], dim=1)
        q1_logsumexp = torch.logsumexp(q1_candidates - log_probs, dim=1).mean()
        q2_logsumexp = torch.logsumexp(q2_candidates - log_probs, dim=1).mean()
        q1_data, q2_data = self.critic.q_all(observations, actions)
        q1_data = q1_data.mean()
        q2_data = q2_data.mean()
        return (q1_logsumexp - q1_data) + (q2_logsumexp - q2_data)

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor=E_(s\sim D,a\sim\pi) [\alpha log\pi(a|s)-Q_(min) (s,a)]
        observations, _, _, _, _ = batch
        actions, log_probs = self.actor.sample(observations)
        q = self.critic.q_min(observations, actions)
        return (self.alpha().detach() * log_probs - q).mean()

    def loss_temperature(self, batch: Batch) -> torch.Tensor:
        # Loss_Temperature=E_(s\sim D,a\sim\pi) [-\alpha(log\pi(a|s)+H_t)]
        observations, _, _, _, _ = batch
        with torch.no_grad():
            _, log_probs = self.actor.sample(observations)
        return -(self.alpha() * (log_probs + self.target_entropy)).mean()
