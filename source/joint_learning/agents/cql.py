import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from joint_learning.datasets.lib import Batch


class GaussianPolicy(torch.nn.Module):
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
        dist, _ = self.dist(observations)
        raw_actions = dist.rsample()
        return torch.tanh(raw_actions), self.log_prob_from_raw(dist, raw_actions)

    def sample_n(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self.dist(observations)
        raw_actions = dist.rsample((self.n_samples,))
        actions = torch.tanh(raw_actions).transpose(0, 1)
        log_probs = self.log_prob_from_raw(dist, raw_actions).transpose(0, 1)
        return actions, log_probs

    def sample_random_n(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class CQLAgent:
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = 0.99
        self.target_update_rate = 0.005
        self.target_entropy = -float(act_size)
        self.threshold = 2.0

        self.policy = GaussianPolicy(obs_size, act_size).to(device)
        self.q1 = QNetwork(obs_size, act_size).to(device)
        self.q2 = QNetwork(obs_size, act_size).to(device)
        self.target_q1 = QNetwork(obs_size, act_size).to(device)
        self.target_q2 = QNetwork(obs_size, act_size).to(device)
        self.sync_target_hard()

        self.log_alpha = torch.nn.Parameter(torch.full((1, 1), math.log(1.0), dtype=torch.float32, device=device))
        self.log_cql_weight = torch.nn.Parameter(torch.full((1, 1), math.log(10.0), dtype=torch.float32, device=device))
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters())
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha])
        self.cql_weight_optimizer = torch.optim.Adam([self.log_cql_weight])

    def act(self, observation) -> np.ndarray:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        observations = torch.as_tensor(observation_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.policy(observations)
        return action.cpu().numpy()[0]

    def update(self, batch: Batch) -> None:
        loss_td = self.loss_td(batch)
        loss_cql = self.loss_cql(batch)

        # CQL Lagrange multiplier objective:
        #   L_alpha' = -alpha' * (L_CQL - threshold)
        # The multiplier grows when the conservative penalty is above the target threshold.
        loss_cql_weight = -(self.cql_weight() * (loss_cql.detach() - self.threshold)).mean()
        self.cql_weight_optimizer.zero_grad()
        loss_cql_weight.backward()
        self.cql_weight_optimizer.step()

        loss = loss_td + self.cql_weight().detach() * loss_cql
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        loss_actor = self.loss_actor(batch)
        self.policy_optimizer.zero_grad()
        loss_actor.backward()
        self.policy_optimizer.step()

        loss_alpha = self.loss_alpha(batch)
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()
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

    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def cql_weight(self) -> torch.Tensor:
        return self.log_cql_weight.exp().clamp(0.0, 1e6)

    def target_q_min(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q1 = self.target_q1(observations, actions)
        q2 = self.target_q2(observations, actions)
        return torch.minimum(q1, q2)

    def loss_td(self, batch: Batch) -> torch.Tensor:
        # SAC Bellman objective used inside CQL:
        #   a' ~ pi(.|s')
        #   y = r + gamma * (1 - d) * [min_i Q_i_target(s',a') - alpha log pi(a'|s')]
        #   L_TD = E_D[ sum_i (Q_i(s,a) - y)^2 ]
        observations, actions, rewards, next_observations, dones = batch
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_observations)
            target_q = self.target_q_min(next_observations, next_actions)
            target = rewards + self.discount_factor * (target_q - self.alpha() * next_log_probs) * (1.0 - dones)
        q1 = self.q1(observations, actions)
        q2 = self.q2(observations, actions)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def evaluate_q_n(self, observations: torch.Tensor, sampled_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = observations.shape[0]
        sample_count = sampled_actions.shape[1]
        repeated_observations = observations.repeat_interleave(sample_count, dim=0)
        flat_actions = sampled_actions.reshape(-1, sampled_actions.shape[-1])
        q1 = self.q1(repeated_observations, flat_actions).view(batch_size, sample_count, 1)
        q2 = self.q2(repeated_observations, flat_actions).view(batch_size, sample_count, 1)
        return q1, q2

    def loss_cql(self, batch: Batch) -> torch.Tensor:
        # Conservative Q-Learning penalty:
        #   L_CQL = E_s[ logsumexp_a(Q(s,a) - log mu(a|s)) ] - E_{(s,a)~D}[ Q(s,a) ]
        # Candidate actions include current-policy, next-state-policy, and uniform random actions.
        # This lowers Q on broad out-of-dataset actions while preserving dataset action values.
        observations, actions, _, next_observations, _ = batch
        with torch.no_grad():
            policy_actions, policy_log_probs = self.policy.sample_n(observations)
            next_policy_actions, next_policy_log_probs = self.policy.sample_n(next_observations)
            random_actions, random_log_probs = self.policy.sample_random_n(observations)
            log_probs = torch.cat([policy_log_probs, next_policy_log_probs, random_log_probs], dim=1)

        q1_policy, q2_policy = self.evaluate_q_n(observations, policy_actions)
        q1_next_policy, q2_next_policy = self.evaluate_q_n(observations, next_policy_actions)
        q1_random, q2_random = self.evaluate_q_n(observations, random_actions)

        q1_candidates = torch.cat([q1_policy, q1_next_policy, q1_random], dim=1)
        q2_candidates = torch.cat([q2_policy, q2_next_policy, q2_random], dim=1)
        q1_logsumexp = torch.logsumexp(q1_candidates - log_probs, dim=1).mean()
        q2_logsumexp = torch.logsumexp(q2_candidates - log_probs, dim=1).mean()
        q1_data = self.q1(observations, actions).mean()
        q2_data = self.q2(observations, actions).mean()
        return (q1_logsumexp - q1_data) + (q2_logsumexp - q2_data)

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # SAC actor objective:
        #   L_pi = E_{s~D,a~pi}[ alpha * log pi(a|s) - min_i Q_i(s,a) ]
        # The policy maximizes Q while maintaining entropy through alpha.
        observations, _, _, _, _ = batch
        actions, log_probs = self.policy.sample(observations)
        q = torch.minimum(self.q1(observations, actions), self.q2(observations, actions))
        return (self.alpha().detach() * log_probs - q).mean()

    def loss_alpha(self, batch: Batch) -> torch.Tensor:
        # SAC temperature objective:
        #   L_alpha = E_{s~D,a~pi}[ -alpha * (log pi(a|s) + H_target) ]
        # This tunes entropy so stochastic exploration pressure matches the target entropy.
        observations, _, _, _, _ = batch
        with torch.no_grad():
            _, log_probs = self.policy.sample(observations)
        return -(self.alpha() * (log_probs + self.target_entropy)).mean()
