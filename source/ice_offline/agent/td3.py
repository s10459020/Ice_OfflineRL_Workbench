from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ice_offline.agent._spec import Agent
from ice_offline.dataset._types import Batch


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, max_action: float = 1.0):
        super().__init__()
        self.max_action = max_action
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.network(o))


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


class TD3Actor(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        tau: float = 0.005,
        noise_scale: float = 0.2,
        noise_clip: float = 0.5,
        max_action: float = 1.0,
        pi_cls: type[torch.nn.Module] = _Pi,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.act_size = act_size
        self.tau = tau
        self.noise_scale = noise_scale * max_action
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.pi = pi_cls(obs_size, act_size, max_action)
        self.tpi = pi_cls(obs_size, act_size, max_action)
        self.sync_target_hard()

    def noise_action(self, a: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(a) * self.noise_scale
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        return (a + noise).clamp(-self.max_action, self.max_action)

    # ====================
    # TD3 target sync
    # ====================
    def sync_target_hard(self) -> None:
        self.tpi.load_state_dict(self.pi.state_dict())

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.pi.parameters(), self.tpi.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


class TD3Critic(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        q_count: int = 2,
        q_cls: type[torch.nn.Module] = _Q,
        tau: float = 0.005,
    ):
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

    def q_mean(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.q_all(o, a), dim=1).mean(dim=1, keepdim=True)

    def tq_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.tq_all(o, a), dim=1).min(dim=1, keepdim=True).values

    def tq_mean(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.tq_all(o, a), dim=1).mean(dim=1, keepdim=True)

    # ====================
    # TD3 target sync
    # ====================
    def sync_target_hard(self) -> None:
        for q, tq in zip(self.q_networks, self.tq_networks):
            tq.load_state_dict(q.state_dict())

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for q, tq in zip(self.q_networks, self.tq_networks):
                for p, tp in zip(q.parameters(), tq.parameters()):
                    tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


@dataclass
class TD3Agent(Agent):
    obs_size: int
    act_size: int
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    update_actor_interval: int = 2
    q_count: int = 2
    update_step: int = 0
    device: str = "cuda"

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        self.actor = TD3Actor(self.obs_size, self.act_size).to(self.device)
        self.critic = TD3Critic(self.obs_size, self.act_size, q_count=self.q_count).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(),
            lr=self.actor_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.q_networks.parameters(),
            lr=self.critic_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Act
    # ====================
    def act(self, observation):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(o)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(o)
        return a.cpu().numpy()

    def eval(self, observations, actions, method: str) -> np.ndarray:
        o = torch.as_tensor(np.asarray(observations, dtype=np.float32), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.asarray(actions, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if method == "Pi":
                mode = self.actor.pi(o)
                values = -((mode - a) ** 2).sum(dim=-1)
            elif method == "Q":
                values = self.critic.q_min(o, a).squeeze(-1)
            else:
                return super().eval(observations, actions, method)
        return values.cpu().numpy().astype(np.float32)

    # ====================
    # Update
    # ====================
    def update(self, batch: Batch):
        self.update_step += 1
        self.update_critic(batch)
        if self.update_step % self.update_actor_interval == 0:
            self.update_actor(batch)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

    def update_critic(self, batch: Batch) -> None:
        self.critic_optimizer.zero_grad()
        critic_loss = self.loss_critic(batch)
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, batch: Batch) -> None:
        self.actor_optimizer.zero_grad()
        actor_loss = self.loss_actor(batch)
        actor_loss.backward()
        self.actor_optimizer.step()

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def _load_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_step = int(state["update_step"])

    # ====================
    # Critic loss
    # ====================
    def target_td3(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # y = r + gamma * Q(s',pi(s'))
        with torch.no_grad():
            an = self.actor.tpi(on)
            an = self.actor.noise_action(an)
            tq = self.critic.tq_min(on, an)
            return r + self.gamma * tq * (1.0 - d)

    def loss_td(self, batch: Batch) -> torch.Tensor:
        # loss = E{s,a,r,s'~D}[ MSE(Q(s,a) - y) ]
        o, a, r, on, d = batch
        target = self.target_td3(on, r, d)
        return sum(F.mse_loss(q, target) for q in self.critic.q_all(o, a))

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        return self.loss_td(batch)

    # ====================
    # Actor loss
    # ====================
    def loss_td3(self, batch: Batch) -> torch.Tensor:
        # loss = E{s~D}[ -Q(s,pi(s)) ]
        # with normalization trick
        o, _, _, _, _ = batch
        a = self.actor.pi(o)
        q = self.critic.q_min(o, a)
        alpha = 1.0 / q.abs().mean().detach() # normalize
        return -alpha * q.mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        return self.loss_td3(batch)
