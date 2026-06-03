from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ice_offline.agent._spec import TorchAgent
from ice_offline.dataset._spec import TorchBuffer


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
        x = torch.cat([o, a], dim=1)
        return self.network(x)


class _TD3_Actor(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        tau: float = 0.005,
        noise_scale: float = 0.2,
        noise_clip: float = 0.5,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.tau = tau
        self.noise_scale = noise_scale * max_action
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.pi = _Pi(obs_size, act_size, max_action)
        self.tpi = _Pi(obs_size, act_size, max_action)
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


class _TD3_Critic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, tau: float = 0.005):
        super().__init__()
        self.tau = tau

        # TD3 twin Q networks
        self.q1 = _Q(obs_size, act_size)
        self.q2 = _Q(obs_size, act_size)

        # TD3 target Q networks
        self.tq1 = _Q(obs_size, act_size)
        self.tq2 = _Q(obs_size, act_size)
        self.sync_target_hard()

    def q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        return torch.cat([q1, q2], dim=1).min(dim=1, keepdim=True).values

    def tq_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        tq1 = self.tq1(o, a)
        tq2 = self.tq2(o, a)
        return torch.cat([tq1, tq2], dim=1).min(dim=1, keepdim=True).values

    # ====================
    # TD3 target sync
    # ====================
    def sync_target_hard(self) -> None:
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.tq1.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for p, tp in zip(self.q2.parameters(), self.tq2.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


@dataclass
class TD3BCAgent(TorchAgent):
    obs_size: int
    act_size: int
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    alpha: float = 2.5
    update_actor_interval: int = 2
    update_step: int = 0
    device: str = "cpu"

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        self.actor = _TD3_Actor(
            obs_size=self.obs_size,
            act_size=self.act_size,
        ).to(self.device)
        self.critic = _TD3_Critic(
            obs_size=self.obs_size,
            act_size=self.act_size,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(),
            lr=self.actor_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
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

    # ====================
    # Update
    # ====================
    def update(self, batch: TorchBuffer):
        o = batch.obs_list
        a = batch.act_list
        r = batch.rew_list.view(-1, 1)
        on = batch.next_obs_list
        d = batch.done_list.view(-1, 1)

        self.update_critic(o, a, r, on, d)

        self.update_step += 1
        if self.update_step % self.update_actor_interval == 0:
            self.update_actor(o, a)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

    def update_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> None:
        self.critic_optimizer.zero_grad()
        critic_loss = self.loss_critic(o, a, r, on, d)
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, o: torch.Tensor, a: torch.Tensor) -> None:
        self.actor_optimizer.zero_grad()
        actor_loss = self.loss_actor(o, a)
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
    # Critic mathmatics
    # ====================
    def td_target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            an = self.actor.noise_action(self.actor.tpi(on))
            tq = self.critic.tq_min(on, an)
            return r + self.gamma * tq * (1.0 - d)
        
    def loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        y = self.td_target(on, r, d)
        q1 = self.critic.q1(o, a)
        q2 = self.critic.q2(o, a)
        return F.mse_loss(q1, y) + F.mse_loss(q2, y)

    # ====================
    # Actor mathmatics
    # ====================
    def loss_bc(self, a: torch.Tensor, a_pred: torch.Tensor) -> torch.Tensor:
        return ((a - a_pred) ** 2).mean()

    def loss_td3(self, o: torch.Tensor, a_pred: torch.Tensor) -> torch.Tensor:
        q = self.critic.q1(o, a_pred)
        lam = self.alpha / q.abs().mean().detach()
        return lam * -q.mean()

    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_pred = self.actor.pi(o)
        return self.loss_td3(o, a_pred) + self.loss_bc(a, a_pred)


