from dataclasses import dataclass

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from ice_offline.agent._spec import Agent
from ice_offline.dataset._types import Batch


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, min_logstd: float = -20.0, max_logstd: float = 2.0):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd_head = torch.nn.Linear(256, act_size)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def forward(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(o)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(self.min_logstd, self.max_logstd)
        return mean, logstd


class SACActor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, pi_cls: type[torch.nn.Module] = _Pi, n_samples: int = 10):
        super().__init__()
        self.pi = pi_cls(obs_size, act_size)
        self.n_samples = n_samples

    def _dist(self, o: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean, logstd = self.pi(o)
        return Normal(mean, logstd.exp()), mean

    def _log_prob(self, dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        jacobian = 2.0 * (math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
        return (dist.log_prob(raw_action) - jacobian).sum(dim=-1, keepdim=True)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        _, mean = self._dist(o)
        return torch.tanh(mean)

    def sample(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self._dist(o)
        raw_action = dist.rsample()
        return torch.tanh(raw_action), self._log_prob(dist, raw_action)

    def sample_n(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist, _ = self._dist(o)
        raw_action = dist.rsample((self.n_samples,))
        action = torch.tanh(raw_action).transpose(0, 1)
        log_prob = self._log_prob(dist, raw_action).transpose(0, 1)
        # action: (B, N, A), log_prob: (B, N, 1)
        return action, log_prob


class _Q(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([o, a], dim=1))


class SACCritic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, q_count: int = 2, q_cls: type[torch.nn.Module] = _Q, tau: float = 0.005):
        super().__init__()
        self.tau = tau
        self.q_networks = torch.nn.ModuleList(
            [q_cls(obs_size, act_size) for _ in range(q_count)]
        )
        self.tq_networks = torch.nn.ModuleList(
            [q_cls(obs_size, act_size) for _ in range(q_count)]
        )
        self.sync_target_hard()

    def q_all(self, o: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(q(o, a) for q in self.q_networks)

    def tq_all(self, o: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(tq(o, a) for tq in self.tq_networks)

    def q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(o, a), dim=1).min(dim=1, keepdim=True).values

    def tq_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.tq_all(o, a), dim=1).min(dim=1, keepdim=True).values

    # ====================
    # SAC target sync
    # ====================
    def sync_target_hard(self) -> None:
        for q, tq in zip(self.q_networks, self.tq_networks):
            tq.load_state_dict(q.state_dict())

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for q, tq in zip(self.q_networks, self.tq_networks):
                for p, tp in zip(q.parameters(), tq.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


class _SACTemperature(torch.nn.Module):
    def __init__(self, 
        act_size: int, 
        learning_rate: float = 3e-4, 
        initial_temperature: float = 1.0, 
        target_entropy: float | None = -3
    ):
        super().__init__()
        self.log_alpha = torch.nn.Parameter(
            torch.full((1, 1), math.log(initial_temperature), dtype=torch.float32)
        )
        self.target_entropy = target_entropy if target_entropy is not None else -float(act_size)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    def forward(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        # loss = E{s~D,a~pi}[ -temp * (log pi(a|s) + target_entropy) ]
        # d3rl SAC source uses target_entropy = -action_size.
        return -(self() * (log_prob + self.target_entropy).detach()).mean()


@dataclass
class SACAgent(Agent):
    obs_size: int
    act_size: int
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temp_learning_rate: float = 3e-4
    gamma: float = 0.99
    initial_temperature: float = 1.0
    target_entropy: float | None = None
    device: str = "cuda"

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        self.actor = SACActor(self.obs_size, self.act_size).to(self.device)
        self.critic = SACCritic(self.obs_size, self.act_size).to(self.device)
        self.temp = _SACTemperature(
            self.act_size,
            learning_rate=self.temp_learning_rate,
            initial_temperature=self.initial_temperature,
            target_entropy=self.target_entropy,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

    # ====================
    # Act
    # ====================
    def act(self, observation, greedy: bool = True):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor(o) if greedy else self.actor.sample(o)[0]
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor(o) if greedy else self.actor.sample(o)[0]
        return a.cpu().numpy()

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch):
        self.update_critic(batch)
        self.update_actor(batch)
        self.update_temperature(batch)
        self.critic.update_target_soft()

    def update_critic(self, batch: Batch) -> None:
        self.critic_optimizer.zero_grad()
        loss_critic = self.loss_critic(batch)
        loss_critic.backward()
        self.critic_optimizer.step()

    def update_actor(self, batch: Batch) -> None:
        self.actor_optimizer.zero_grad()
        loss_actor = self.loss_actor(batch)
        loss_actor.backward()
        self.actor_optimizer.step()
    
    def update_temperature(self, batch: Batch) -> None:
        o, _, _, _, _ = batch
        _, log_prob = self.actor.sample(o)
        self.temp.optimizer.zero_grad()
        loss_temperature = self.temp.loss(log_prob)
        loss_temperature.backward()
        self.temp.optimizer.step()

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "temperature": self.temp.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temp.optimizer.state_dict(),
        }

    def _load_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.temp.load_state_dict(state["temperature"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.temp.optimizer.load_state_dict(state["temperature_optimizer"])

    # ====================
    # Critic loss
    # ====================
    def target_sac(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # y = r + gamma * (min Q(s',a') - temp * log pi(a'|s'))
        with torch.no_grad():
            an, log_prob = self.actor.sample(on)
            tq = self.critic.tq_min(on, an)
            return r + self.gamma * (tq - self.temp() * log_prob) * (1.0 - d)

    def loss_td(self, batch: Batch) -> torch.Tensor:
        # loss = sum_i E{s,a,r,s'~D}[ MSE(Qi(s,a) - y) ]
        o, a, r, on, d = batch
        target = self.target_sac(on, r, d)
        return sum(F.mse_loss(q, target) for q in self.critic.q_all(o, a))

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        return self.loss_td(batch)

    # ====================
    # Actor loss
    # ====================
    def loss_sac(self, batch: Batch) -> torch.Tensor:
        # loss = E{s~D,a~pi}[ temp * log pi(a|s) - min Q(s,a) ]
        o, _, _, _, _ = batch
        a, log_prob = self.actor.sample(o)
        q = self.critic.q_min(o, a)
        return (self.temp() * log_prob - q).mean()
    
    def loss_actor(self, batch: Batch) -> torch.Tensor:
        return self.loss_sac(batch)
