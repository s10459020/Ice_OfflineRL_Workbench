from dataclasses import dataclass

import torch
import numpy as np
import torch.nn.functional as F

from ice_offline.agent._spec import TorchAgent
from ice_offline.dataset._spec import TorchBuffer


class _M(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, obs_size),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a], -1)
        return self.network(x)


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
        x = torch.cat([o, a], -1)
        return self.network(x)


class _TD3_Actor(torch.nn.Module):
    # DDPG: target policy
    # TD3: policy smoothing
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
        self.noise_scale = noise_scale
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
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


class _TD3_Critic(torch.nn.Module):
    # DDPG: target critic
    # TD3: multiple critics
    # TD3: clipped critic
    def __init__(self, obs_size: int, act_size: int, tau: float = 0.005):
        super().__init__()
        self.tau = tau

        # four Q networks
        self.q1 = _Q(obs_size, act_size)
        self.q2 = _Q(obs_size, act_size)
        self.q3 = _Q(obs_size, act_size)
        self.q4 = _Q(obs_size, act_size)

        # target Q networks
        self.tq1 = _Q(obs_size, act_size)
        self.tq2 = _Q(obs_size, act_size)
        self.tq3 = _Q(obs_size, act_size)
        self.tq4 = _Q(obs_size, act_size)

        self.sync_target_hard()

    def q_all(self, o: torch.Tensor, a: torch.Tensor) -> list[_Q]:
        return [self.q1(o, a), self.q2(o, a), self.q3(o, a), self.q4(o, a)]

    def q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q_cat = torch.cat(self.q_all(o, a), dim=1) # (B, Q)     
        q_min, _ = torch.min(q_cat, dim=1) # (B,)       
        return q_min
    
    def q_mean(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q_list = self.q_all(o, a)
        return sum(q_list) / len(q_list)

    def tq_all(self, o: torch.Tensor, a: torch.Tensor) -> list[_Q]:
        return [self.tq1(o, a), self.tq2(o, a), self.tq3(o, a), self.tq4(o, a)]

    def tq_min(self, on: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        tq_cat = torch.cat(self.tq_all(on, a), dim=1) # (B, 4)     
        tq_min, _ = torch.min(tq_cat, dim=1, keepdim=True) # (B, 1)       
        return tq_min
    
    def tq_mean(self, on: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        tq_list = self.tq_all(on, a)
        return sum(tq_list) / len(tq_list)

    # ====================
    # TD3 target sync
    # ====================
    def sync_target_hard(self) -> None:
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())
        self.tq3.load_state_dict(self.q3.state_dict())
        self.tq4.load_state_dict(self.q4.state_dict())

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.tq1.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.q2.parameters(), self.tq2.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.q3.parameters(), self.tq3.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.q4.parameters(), self.tq4.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

@dataclass
class ScasDynamicAgent(TorchAgent):
    obs_dim: int
    act_dim: int
    learning_rate: float = 1e-3
    device: str = "cpu"

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        self.model = _M(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    # ====================
    # Prepare
    # ====================
    def prepare(self) -> torch.nn.Module:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        return self.model

    # ====================
    # Update
    # ====================
    def update(self, batch: TorchBuffer):
        o = batch.obs_list
        a = batch.act_list
        on = batch.next_obs_list

        self.update_dynamic(o, a, on)

    def update_dynamic(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss = self.loss_dynamic(o, a, on)
        loss.backward() 
        self.optimizer.step()

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {"model": self.model.state_dict()}

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state["model"])
  
    # ====================
    # Dynamic loss
    # ====================
    def loss_dynamic(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        # loss: E_{s,a,s'~D}[ ||M(s,a) - s'||^2 ]
        pred = self.model(o, a)
        return F.mse_loss(pred, on)
        

@dataclass
class ScasMeanAgent(TorchAgent):
    obs_dim: int
    act_dim: int
    dynamics: ScasDynamicAgent
    actor_learning_rate: float = 2e-4
    critic_learning_rate: float = 3e-4
    beta: float = 3e-3
    alpha: float = 5.0
    lmbda: float = 0.25
    gamma: float = 0.99
    update_step: int = 0
    policy_freq: int = 2
    max_weight: float = 50.0
    device: str = "cpu"

    # ====================
    # Init
    # ====================
    def __post_init__(self) -> None:
        self.actor = _TD3_Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = _TD3_Critic(self.obs_dim, self.act_dim).to(self.device)
        self.dynamics = self.dynamics.prepare().to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(), 
            lr=self.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.q1.parameters())
            + list(self.critic.q2.parameters())
            + list(self.critic.q3.parameters())
            + list(self.critic.q4.parameters()),
            lr=self.critic_learning_rate,
        )

    # ====================
    # Act
    # ====================
    def act(self, observation):
        o_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(o_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(o)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch):
        o_np = np.asarray(observation_batch, dtype=np.float32)
        o = torch.as_tensor(o_np, dtype=torch.float32, device=self.device)
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
        d = batch.done_list.view(-1, 1)
        on = batch.next_obs_list

        self.update_step += 1
        self.update_critic(o, a, r, on, d)
        if self.update_step % self.policy_freq == 0:
            self.update_actor(o, on)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

    def update_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> None:
        self.critic_optimizer.zero_grad()
        critic_loss = self.loss_critic(o, a, r, on, d)
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, o: torch.Tensor, on: torch.Tensor) -> None:
        self.actor_optimizer.zero_grad()
        actor_loss = self.loss_actor(o, on)
        actor_loss.backward()
        self.actor_optimizer.step()

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_step = int(state["update_step"])
  
    # ====================
    # Critic loss
    # ====================
    def td_target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            an = self.actor.noise_action(self.actor.tpi(on))
            tq = self.critic.tq_mean(on, an)
            return r + self.gamma * tq * (1 - d)
    
    def loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        y = self.td_target(on, r, d)
        q1 = self.critic.q1(o, a)
        q2 = self.critic.q2(o, a)
        q3 = self.critic.q3(o, a)
        q4 = self.critic.q4(o, a)
        loss_q1 = F.mse_loss(q1, y)
        loss_q2 = F.mse_loss(q2, y)
        loss_q3 = F.mse_loss(q3, y)
        loss_q4 = F.mse_loss(q4, y)
        return loss_q1 + loss_q2 + loss_q3 + loss_q4

    # ====================
    # Actor loss
    # ====================    
    def _s_perturbed(self, s: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(s.shape, device=s.device) * self.beta
        return s + noise
    
    def loss_td3(self, s:torch.Tensor) -> torch.Tensor:
        a = self.actor.pi(s)
        q = self.critic.q_min(s, a)
        alpha = 1.0 / q.abs().mean().detach() # TD3BC (1-alpha)
        return -alpha * q.mean() # mean over batch
    
    def loss_correction(self, s: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        # R2 = E_{s,s'~D}, {ps~perturbed(s)} [exp( alpha* ( V' - V ) ) * ||M(s,a) - s'||^2]
        a = self.actor.pi(s)
        v = self.critic.q_mean(s, a) # scas V(s) = Q(s, pi(s))
        an = self.actor.pi(on)
        vn = self.critic.q_mean(on, an) # scas V(s') = Q(s', pi(s'))
        ps = self._s_perturbed(s)

        weight = (
            self.alpha * (vn.detach() - v.detach())
        ).exp().clamp(max = self.max_weight)

        grad = (self.dynamics(ps, a) - on) ** 2
        return (weight * grad).mean() # mean over batch

    def loss_actor(self, s: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.lmbda) * self.loss_td3(s) + self.lmbda * self.loss_correction(s, on)


