"""Implicit Q-Learning agent (minimal fixed structure)."""

import numpy as np
import torch
from torch.distributions import Normal

from ice_offline.agent._spec import Agent
from ice_offline.agent._spec import MetricValues
from ice_offline.dataset._types import Batch


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd = torch.nn.Parameter(torch.zeros(1, act_size, dtype=torch.float32))
        self.min_logstd = config.get("min_logstd", -5.0)
        self.max_logstd = config.get("max_logstd", 2.0)

    def dist(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(o)
        mean = self.mean_head(h)
        squashed_mean = torch.tanh(mean)
        base = self.max_logstd - self.min_logstd
        # use_std_parameter
        logstd = self.min_logstd + torch.sigmoid(self.logstd) * base
        return squashed_mean, logstd

class _Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.pi = _Pi(obs_size, act_size, config)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        mean, _ = self.pi.dist(o)
        return mean

    def sample(self, o: torch.Tensor) -> torch.Tensor:
        mean, logstd = self.pi.dist(o)
        return Normal(mean, logstd.exp()).rsample().clamp(-1.0, 1.0)

    def log_prob(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mean, logstd = self.pi.dist(o)
        return Normal(mean, logstd.exp()).log_prob(a).sum(dim=-1, keepdims=True)

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

class _V(torch.nn.Module):
    def __init__(self, obs_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.expectile = config.get("expectile", 0.7)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.network(o)

class _Critic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}):
        super().__init__()
        self.target_update_rate = config.get("target_update_rate", 0.005)
        self.q1 = _Q(obs_size, act_size)
        self.q2 = _Q(obs_size, act_size)
        self.targ_q1 = _Q(obs_size, act_size)
        self.targ_q2 = _Q(obs_size, act_size)
        self.v = _V(obs_size, config)
        self.sync_target_hard()

    # ====================
    # IQL target sync
    # ====================
    def sync_target_hard(self) -> None:
        self.targ_q1.load_state_dict(self.q1.state_dict())
        self.targ_q2.load_state_dict(self.q2.state_dict())

    def q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        return torch.cat([q1, q2], dim=1).min(dim=1, keepdim=True).values

    def target_q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.targ_q1(o, a)
        q2 = self.targ_q2(o, a)
        return torch.cat([q1, q2], dim=1).min(dim=1, keepdim=True).values

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.targ_q1.parameters()):
                p_targ.data.copy_(
                    self.target_update_rate * p.data
                    + (1.0 - self.target_update_rate) * p_targ.data
                )
            for p, p_targ in zip(self.q2.parameters(), self.targ_q2.parameters()):
                p_targ.data.copy_(
                    self.target_update_rate * p.data
                    + (1.0 - self.target_update_rate) * p_targ.data
                )

    # ====================
    # Params
    # ====================
    def param_q(self):
        return list(self.q1.parameters()) + list(self.q2.parameters())

    def param_v(self):
        return self.v.parameters()

    def param_critic(self):
        return self.param_q() + list(self.param_v())

class IQLAgent(Agent):
    def __init__(self, obs_size: int, act_size: int, config: dict[str, object] = {}, device: str = "cuda") -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.device = device
        self.discount_factor = config.get("discount_factor", 0.99)
        self.advantage_scale = config.get("advantage_scale", 3.0)
        self.cap_weight = config.get("cap_weight", 100.0)
        self.actor = _Actor(self.obs_size, self.act_size, config).to(self.device)
        self.critic = _Critic(self.obs_size, self.act_size, config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.param_critic())

    # ====================
    # Act
    # ====================
    def act(self, observation, greedy: bool = True):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(o) if greedy else self.actor.sample(o)
        return action.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(o) if greedy else self.actor.sample(o)
        return action.cpu().numpy()

    def eval(self, observations, actions, method: str) -> np.ndarray:
        o = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(actions, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if method == "Pi":
                values = self.actor.log_prob(o, a).squeeze(-1)
            elif method == "V":
                values = self.critic.v(o).squeeze(-1)
            elif method == "Q":
                values = self.critic.q_min(o, a).squeeze(-1)
            else:
                return super().eval(observations, actions, method)
        return values.cpu().numpy().astype(np.float32)

    # ====================
    # Update
    # ====================
    def metric_keys(self) -> list[str]:
        return [
            "loss_q",
            "grad_q",
            "loss_v",
            "grad_v",
            "loss_critic",
            "grad_critic",
            "loss_actor",
            "grad_actor",
            "target_q",
        ]

    def update(self, batch: Batch) -> MetricValues:
        metrics = self.update_critic(batch)
        metrics |= self.update_actor(batch)
        self.critic.update_target_soft()
        return metrics

    def update_critic(self, batch: Batch) -> MetricValues:
        loss_critic, metrics = self.loss_critic(batch)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        return metrics

    def update_actor(self, batch: Batch) -> MetricValues:
        loss_actor, metrics = self.loss_actor(batch)
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        return metrics

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    # ====================
    # critic
    # ====================
    def target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return r + self.discount_factor * self.critic.v(on) * (1.0 - d)

    def loss_q(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, a, r, on, d = batch
        target = self.target(on, r, d)
        q1 = self.critic.q1(o, a)
        q2 = self.critic.q2(o, a)
        # loss_q = E_batch{(target - Q)^2}
        loss = (q1 - target).pow(2).mean() + (q2 - target).pow(2).mean()
        return loss, {
            "loss_q": self._value(loss.detach()),
            "grad_q": self._grad_norm(loss, self.critic.param_q()),
            "target_q": self._value(self.critic.target_q_min(o, a).mean().detach()),
        }

    def loss_v(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, a, _, _, _ = batch
        # L2_tau = |tau - 1(u<0)| * u^2
        # loss_v = E_batch{ L2_tau(Q-V) }
        q_t = self.critic.target_q_min(o, a)
        v_t = self.critic.v(o)
        diff = q_t.detach() - v_t
        weight = (self.critic.v.expectile - (diff < 0.0).float()).abs().detach()
        loss = (weight * diff.pow(2)).mean()
        return loss, {
            "loss_v": self._value(loss.detach()),
            "grad_v": self._grad_norm(loss, self.critic.param_v()),
        }

    def loss_critic(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_q, metrics_q = self.loss_q(batch)
        loss_v, metrics_v = self.loss_v(batch)
        loss = loss_q + loss_v
        return loss, metrics_q | metrics_v | {
            "loss_critic": self._value(loss.detach()),
            "grad_critic": self._grad_norm(loss, self.critic.param_critic()),
        }

    # ====================
    # actor
    # ====================
    def weight(self, batch: Batch) -> torch.Tensor:
        o, a, _, _, _ = batch
        # weight = -exp( scale * (Q - V))
        with torch.no_grad():
            q_t = self.critic.target_q_min(o, a)
            v_t = self.critic.v(o)
            adv = q_t - v_t
            return -(self.advantage_scale * adv).exp().clamp(max=self.cap_weight)

    def loss_actor(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        o, a, _, _, _ = batch
        # loss_pi = E_batch{weight * log_pi}
        weight = self.weight(batch)
        log_pi = self.actor.log_prob(o, a)
        loss = (weight * log_pi).mean()
        return loss, {
            "loss_actor": self._value(loss.detach()),
            "grad_actor": self._grad_norm(loss, self.actor.parameters()),
        }
