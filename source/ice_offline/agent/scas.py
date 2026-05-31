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
    def __init__(self, obs_size: int, act_size: int, tau: float = 0.005, max_action: float = 1.0):
        super().__init__()
        self.tau = tau
        self.pi = _Pi(obs_size, act_size, max_action)
        self.tpi = _Pi(obs_size, act_size, max_action)
        self.sync_target_hard()

    # ====================
    # callable
    # ====================
    def pi_act(self, o: torch.Tensor) -> torch.Tensor:
        return self.pi(o)
    
    def tpi_act(self, o: torch.Tensor) -> torch.Tensor:
        return self.tpi(o)

    # ====================
    # target sync
    # ====================
    def get_parameters(self):
        return self.pi.parameters()
    
    def sync_target_hard(self) -> None:
        self.tpi.load_state_dict(self.pi.state_dict())
        
    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.pi.parameters(), self.tpi.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

class _TD3_Critic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, tau: float):
        super().__init__()
        self.tau = tau

        # "double Q" for 4q
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
 

    # ====================
    # callable
    # ====================
    def q_values(self, o: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.q1(o, a), self.q2(o, a), self.q3(o, a), self.q4(o, a))

    def q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2, q3, q4 = self.q_values(o, a) # (B, 1) *4   
        q_cat = torch.cat([q1, q2, q3, q4], dim=1) # (B, 4)     
        q_min, _ = torch.min(q_cat, dim=1) # (B,)       
        return q_min
    
    def q_mean(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2, q3, q4 = self.q_values(o, a)                       
        return (q1 + q2 + q3 + q4) / 4
    
    def tq_values(self, o: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.tq1(o, a), self.tq2(o, a), self.tq3(o, a), self.tq4(o, a))

    def tq_min(self, sn: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        tq1, tq2, tq3, tq4 = self.tq_values(sn, a) # (B, 1) *4   
        tq_cat = torch.cat([tq1, tq2, tq3, tq4], dim=1) # (B, 4)     
        tq_min, _ = torch.min(tq_cat, dim=1, keepdim=True) # (B, 1)       
        return tq_min
    
    def tq_mean(self, sn: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        tq1, tq2, tq3, tq4 = self.tq_values(sn, a)                       
        return (tq1 + tq2 + tq3 + tq4) / 4.0

    # ====================
    # sync
    # ====================
    def get_parameters(self):
        return (
            list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.q3.parameters())
            + list(self.q4.parameters())
        )
    
    def sync_target_hard(self) -> None:
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())
        self.tq3.load_state_dict(self.q3.state_dict())
        self.tq4.load_state_dict(self.q4.state_dict())

    def update_target_soft(self) -> None:
        # DDPG soft target: theta_target <= (1 - tau)theta_target + (tau)theta
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
class ScasDynamic(TorchAgent):
    obs_dim: int
    act_dim: int
    learning_rate: float = 1e-3
    device: torch.device = "cpu"

    def __post_init__(self) -> None:
        self.model = _M(self.obs_dim, self.act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    # ====================
    # extend 
    # ====================
    def prepare(self) -> torch.nn.Module:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        return self.model
    
    def update(self, batch: TorchBuffer):
        s = batch.obs_list
        a = batch.act_list
        sn = batch.next_obs_list

        self.optimizer.zero_grad()
        loss = self.loss_dynamic(s, a, sn)
        loss.backward() 
        self.optimizer.step()

    # ====================
    # extend 
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {"model": self.model.state_dict()}

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state["model"])
  

    # ====================
    # mathmatics
    # ====================
    def loss_dynamic(self, s: torch.Tensor, a: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        # loss: E_{s,a,s'~D} [||M(s,a) - s'||^2]
        pred = self.model(s, a)
        return F.mse_loss(pred, sn)
        

@dataclass
class ScasAgent(TorchAgent):
    obs_dim: int
    act_dim: int
    dynamics: ScasDynamic
    max_action: float = 1.0
    tau: float = 0.005
    beta: float = 3e-3
    alpha: float = 5.0
    lmbda: float = 0.25
    gamma: float = 0.99
    update_step: int = 0
    policy_freq: int = 2
    max_weight: float = 50.0
    device: torch.device = "cpu"

    def __post_init__(self) -> None:
        self.actor = _TD3_Actor(self.obs_dim, self.act_dim, tau=self.tau, max_action=self.max_action).to(self.device)
        self.critic = _TD3_Critic(self.obs_dim, self.act_dim, tau=self.tau).to(self.device)
        self.dynamics = self.dynamics.prepare().to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.get_parameters(), lr=2e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.get_parameters(), lr=3e-4)

    # ====================
    # public API
    # ====================
    def act(self, observation, greedy: bool = True):
        s_np = np.asarray(observation, dtype=np.float32)[None, :]
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi_act(s)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        s_np = np.asarray(observation_batch, dtype=np.float32)
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi_act(s)
        return a.cpu().numpy()

    def update(self, batch: TorchBuffer):
        s = batch.obs_list
        a = batch.act_list
        r = batch.rew_list.view(-1, 1)
        d = batch.done_list.view(-1, 1)
        sn = batch.next_obs_list
      
        # update every cycle
        self.critic_optimizer.zero_grad()
        critic_loss = self.loss_critic(s, a, r, sn, d)
        critic_loss.backward()
        self.critic_optimizer.step()

        # lazy update
        self.update_step += 1
        if self.update_step % self.policy_freq == 0:
            self.actor_optimizer.zero_grad()
            actor_loss = self.loss_actor(s, sn)
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic.update_target_soft()
            self.actor.update_target_soft()

    # ====================
    # extend 
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
    # critic mathmatics
    # ====================
    def td_target(self, sn: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            an = self.actor.tpi_act(sn)
            noise = (torch.randn_like(an) * 0.2).clamp(-0.5, 0.5)
            an = (an + noise).clamp(-self.max_action, self.max_action)
            tq = self.critic.tq_min(sn,an)
            return r + self.gamma * tq * (1 - d)
    
    def loss_critic(
        self,
        o: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        sn: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        y = self.td_target(sn, r, d)
        q1, q2, q3, q4 = self.critic.q_values(o, a)
        loss_q1 = F.mse_loss(q1, y)
        loss_q2 = F.mse_loss(q2, y)
        loss_q3 = F.mse_loss(q3, y)
        loss_q4 = F.mse_loss(q4, y)
        return loss_q1 + loss_q2 + loss_q3 + loss_q4

    # ====================
    # actor mathmatics
    # ====================    
    def _s_perturbed(self, s: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(s.shape, device=s.device) * self.beta
        return s + noise
    
    def loss_td3(self, s:torch.Tensor) -> torch.Tensor:
        a = self.actor.pi_act(s)
        q = self.critic.q_min(s, a)
        alpha = 1.0 / q.abs().mean().detach() # TD3BC (1-alpha)
        return -alpha * q.mean() # mean over batch
    
    def loss_correction(self, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        # R2 = E_{s,s'~D}, {ps~perturbed(s)} [exp( alpha* ( V' - V ) ) * ||M(s,a) - s'||^2]
        a = self.actor.pi_act(s)
        v = self.critic.q_mean(s, a) # scas V(s) = Q(s, pi(s))
        an = self.actor.pi_act(sn)
        vn = self.critic.q_mean(sn, an) # scas V(s') = Q(s', pi(s'))
        ps = self._s_perturbed(s)

        weight = (
            self.alpha * (vn.detach() - v.detach())
        ).exp().clamp(max = self.max_weight)

        grad = (self.dynamics(ps, a) - sn) ** 2
        return (weight * grad).mean() # mean over batch

    def loss_actor(self, s: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.lmbda) * self.loss_td3(s) + self.lmbda * self.loss_correction(s, sn)
