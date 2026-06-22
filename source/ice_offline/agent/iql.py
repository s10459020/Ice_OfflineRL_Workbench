"""Implicit Q-Learning agent (minimal fixed structure)."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Normal
from ice_offline.agent._spec import Agent
from ice_offline.dataset._types import Batch

class _Pi(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        min_logstd: float = -5.0,
        max_logstd: float = 2.0,
    ):
        super().__init__()
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd = torch.nn.Parameter(torch.zeros(1, act_size, dtype=torch.float32))
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd

    def dist(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(o)
        mean = self.mean_head(h)
        squashed_mean = torch.tanh(mean)
        base = self.max_logstd - self.min_logstd
        # use_std_parameter
        logstd = self.min_logstd + torch.sigmoid(self.logstd) * base
        return squashed_mean, logstd

class _Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.pi = _Pi(obs_size, act_size)

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
    def __init__(self, obs_size: int, tau: float):
        super().__init__()
        self.tau = tau
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
    def __init__(self, obs_size: int, act_size: int, q_tau: float = 0.005, v_tau: float = 0.7):
        super().__init__()
        self.q_tau = q_tau
        self.q1 = _Q(obs_size, act_size)
        self.q2 = _Q(obs_size, act_size)
        self.targ_q1 = _Q(obs_size, act_size)
        self.targ_q2 = _Q(obs_size, act_size)
        self.v = _V(obs_size, tau=v_tau)
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
                p_targ.data.copy_(self.q_tau * p.data + (1.0 - self.q_tau) * p_targ.data)
            for p, p_targ in zip(self.q2.parameters(), self.targ_q2.parameters()):
                p_targ.data.copy_(self.q_tau * p.data + (1.0 - self.q_tau) * p_targ.data)

@dataclass
class IQLAgent(Agent):
    obs_size: int
    act_size: int
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    advantage_scale: float = 3.0
    cap_weight: float = 100.0
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.actor = _Actor(self.obs_size, self.act_size).to(self.device)
        self.critic = _Critic(self.obs_size, self.act_size).to(self.device)
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.critic_optim = torch.optim.Adam(
              list(self.critic.q1.parameters())
            + list(self.critic.q2.parameters())
            + list(self.critic.v.parameters()),
            lr=self.critic_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

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
    def update(self, batch: Batch):
        self.update_critic(batch)
        self.update_actor(batch)
        self.critic.update_target_soft()

    def update_critic(self, batch: Batch) -> None:
        critic_loss = self.loss_critic(batch)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

    def update_actor(self, batch: Batch) -> None:
        actor_loss = self.loss_actor(batch)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optim.load_state_dict(state["actor_optimizer"])
        self.critic_optim.load_state_dict(state["critic_optimizer"])

    # ====================
    # critic
    # ====================
    def target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return r + self.gamma * self.critic.v(on) * (1.0 - d)

    def loss_q(self, batch: Batch) -> torch.Tensor:
        o, a, r, on, d = batch
        target = self.target(on, r, d)
        q1 = self.critic.q1(o, a)
        q2 = self.critic.q2(o, a)
        # loss_q = E_batch{(target - Q)^2}
        return (q1 - target).pow(2).mean() + (q2 - target).pow(2).mean()

    def loss_v(self, batch: Batch) -> torch.Tensor:
        o, a, _, _, _ = batch
        # L2_tau = |tau - 1(u<0)| * u^2
        # loss_v = E_batch{ L2_tau(Q-V) }
        q_t = self.critic.target_q_min(o, a)
        v_t = self.critic.v(o)
        diff = q_t.detach() - v_t
        weight = (self.critic.v.tau - (diff < 0.0).float()).abs().detach()
        return (weight * diff.pow(2)).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        return self.loss_q(batch) + self.loss_v(batch)

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

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        o, a, _, _, _ = batch
        # loss_pi = E_batch{weight * log_pi}
        weight = self.weight(batch)
        log_pi = self.actor.log_prob(o, a)
        return (weight * log_pi).mean()

