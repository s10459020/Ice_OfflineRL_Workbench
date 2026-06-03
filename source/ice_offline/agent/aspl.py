from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import qmc

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
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

        # hidden linear init
        torch.nn.init.constant_(self.network[0].bias, 0.1)
        torch.nn.init.constant_(self.network[2].bias, 0.1)
        torch.nn.init.constant_(self.network[4].bias, 0.1)
        torch.nn.init.kaiming_uniform_(self.network[0].weight, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.network[2].weight, a=5**0.5)
        torch.nn.init.kaiming_uniform_(self.network[4].weight, a=5**0.5)

        # output linear init
        torch.nn.init.uniform_(self.network[6].weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.network[6].bias, -1e-3, 1e-3)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.network(o))


class _Q(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),   # 0
            torch.nn.ReLU(),                   # 1
            torch.nn.LayerNorm(256),           # 2
            torch.nn.Linear(256, 256),         # 3
            torch.nn.ReLU(),                   # 4
            torch.nn.LayerNorm(256),           # 5
            torch.nn.Linear(256, 256),         # 6
            torch.nn.ReLU(),                   # 7
            torch.nn.LayerNorm(256),           # 8
            torch.nn.Linear(256, 256),         # 9
            torch.nn.ReLU(),                   # 10
            torch.nn.LayerNorm(256),           # 11
            torch.nn.Linear(256, 1),           # 12
        )

        # hidden linear init
        torch.nn.init.kaiming_uniform_(self.network[0].weight, a=5**0.5)
        torch.nn.init.constant_(self.network[0].bias, 0.1)
        torch.nn.init.kaiming_uniform_(self.network[3].weight, a=5**0.5)
        torch.nn.init.constant_(self.network[3].bias, 0.1)
        torch.nn.init.kaiming_uniform_(self.network[6].weight, a=5**0.5)
        torch.nn.init.constant_(self.network[6].bias, 0.1)
        torch.nn.init.kaiming_uniform_(self.network[9].weight, a=5**0.5)
        torch.nn.init.constant_(self.network[9].bias, 0.1)

        # output init
        torch.nn.init.uniform_(self.network[12].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.network[12].bias, -3e-3, 3e-3)


    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([o, a], -1)
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

    def sync_target_hard(self) -> None:
        self.tpi.load_state_dict(self.pi.state_dict())
        
    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p, tp in zip(self.pi.parameters(), self.tpi.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)


class _TD3_Critic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, tau: float):
        super().__init__()
        self.tau = tau

        # "double Q" for 4q
        self.q1 = _Q(obs_size, act_size)
        self.q2 = _Q(obs_size, act_size)

        # target Q networks
        self.tq1 = _Q(obs_size, act_size)
        self.tq2 = _Q(obs_size, act_size)
        self.sync_target_hard()
 
    def q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        q_cat = torch.cat([q1, q2], dim=1) # (B, 2)     
        q_min, _ = torch.min(q_cat, dim=1) # (B,)       
        return q_min

    def tq_min(self, sn: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        tq1 = self.tq1(sn, a)
        tq2 = self.tq2(sn, a)
        tq_cat = torch.cat([tq1, tq2], dim=1) # (B, 2)     
        tq_min, _ = torch.min(tq_cat, dim=1, keepdim=True) # (B, 1)       
        return tq_min

    def sync_target_hard(self) -> None:
        self.tq1.load_state_dict(self.q1.state_dict())
        self.tq2.load_state_dict(self.q2.state_dict())

    def update_target_soft(self) -> None:
        # DDPG soft target: theta_target <= (1 - tau)theta_target + (tau)theta
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.tq1.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for p, tp in zip(self.q2.parameters(), self.tq2.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data) 



@dataclass
class AsplAgent(TorchAgent):
    obs_dim: int
    act_dim: int
    max_action: float
    tau: float = 0.005
    alpha: float = 2.5
    gamma: float = 0.99
    q_mean: float = 0
    noise_clip: float = 0.5
    update_step: int = 0
    policy_freq: int = 2
    noise_scale: float = 0.2
    learning_rate: float = 3e-4
    num_sample: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.actor = _TD3_Actor(
            self.obs_dim,
            self.act_dim,
            tau=self.tau,
            noise_scale=self.noise_scale,
            noise_clip=self.noise_clip,
            max_action=self.max_action,
        ).to(self.device)
        self.critic = _TD3_Critic(self.obs_dim, self.act_dim, tau=self.tau).to(self.device)
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_dim)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(), 
            lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
            lr=self.learning_rate,
        )

        # ignore use_lr_scheduler


    # ====================
    # public API
    # ====================
    def act(self, observation):
        s_np = np.asarray(observation, dtype=np.float32)[None, :]
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(s)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch):
        s_np = np.asarray(observation_batch, dtype=np.float32)
        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor.pi(s)
        return a.cpu().numpy()

    def set_seed(self, seed: int) -> None:
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_dim, seed=seed)

    def update(self, batch: TorchBuffer):
        s = batch.obs_list
        a = batch.act_list
        r = batch.rew_list.view(-1, 1)
        d = batch.done_list.view(-1, 1)
        sn = batch.next_obs_list

        self.update_step += 1

        # update every cycle
        self.critic_optimizer.zero_grad()
        critic_loss = self.loss_critic(s, a, r, sn, d)
        critic_loss.backward()
        self.critic_optimizer.step()

        # lazy update
        if self.update_step % self.policy_freq == 0:
            self.actor_optimizer.zero_grad()
            actor_loss = self.loss_td3_variant(s)
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
    # common
    # ====================
    def _sample_lhs_actions(self, batch_size: int, num_samples: int) -> torch.Tensor:
        samples = self._lhs_sampler.random(n=num_samples) # (N, A)[0 ~ 1]

        samples = qmc.scale(
            samples, 
            [-self.max_action] * self.act_dim, 
            [self.max_action] * self.act_dim
        ) # [-a ~ a]

        samples = torch.as_tensor(
            samples,
            dtype=torch.float32,
            device=self.device
        )

        samples = samples.unsqueeze(1) # (N, 1, A)
        samples = samples.repeat(1, batch_size, 1) # (N, B, A)
        return samples


    # ====================
    # critic mathmatics
    # ====================
    def td_target(self, sn: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            an = self.actor.noise_action(self.actor.tpi(sn))
            tq = self.critic.tq_min(sn,an)
            return r + self.gamma * tq * (1.0 - d)
    
    def _c(self, target: torch.Tensor):
        current = torch.abs(target).mean().item() # mean(B, 1) > (1)
        self.q_mean = ((self.update_step - 1) * self.q_mean + current) / self.update_step
        return self.q_mean

    def _d(self, a:torch.Tensor, a_samples: torch.Tensor):
        # d(a,a~) = [ (a - a~) / (a_max - a_min) ]^2
        # source use constant (a_max-a_min) ~= 2*a_max
        # d range: [-0.5, 0.5]
        # a shape: (B, A)
        # a_samples shape: (N, B, A)
        diff = (a - a_samples)**2 # (N, B, A) 
        normalize = (2 * self.max_action)**2
        result = diff / normalize 
        return result.mean(dim=2, keepdim=True) # (N, B, 1)
        
    def _q_pseudo(self, a: torch.Tensor, a_sample: torch.Tensor, q_target: torch.Tensor):
        # Q~(s,a~) = Q^(s,a) - c * d(a,a~)
        # c is scaling coefficent
        # q use q_target in source
        # q_target range: [-Q, Q], shape: (B, 1)
        with torch.no_grad():
            d = self._d(a, a_sample) # (N, B, 1)
            c = self._c(q_target) # (1)
            return q_target - c * d  # (N, B, 1)

    def loss_td_with_target(self, o: torch.Tensor, a: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        # double Q TD
        q1 = self.critic.q1(o, a)
        q2 = self.critic.q2(o, a)
        loss_q1 = F.mse_loss(q1, q_target)
        loss_q2 = F.mse_loss(q2, q_target)
        return loss_q1 + loss_q2
    
    def loss_punish_with_target(self, s: torch.Tensor, a: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        # E_{s~D}{(a~)~U}[ Q(s,a~) - Q~(s,a~) ]^2
        a_samples = self._sample_lhs_actions(s.shape[0], self.num_sample)                      # (N,B,A)
        q_pseudo = self._q_pseudo(a, a_samples, q_target)                                # (N,B,1)

        # reshape
        s_reshape = s.unsqueeze(0).expand(self.num_sample, -1, -1).reshape(-1, s.shape[1])  # (B,S) > (1,B,S) > (N,B,S) > (N*B,S)
        a_samples_reshape = a_samples.view(-1, a.shape[1])                                  # (N,B,A) > (N*B,A)
        q_pseudo_reshape = q_pseudo.view(-1, 1)                                             # (N*B,1)  
        
        q_values = (self.critic.q1(s_reshape, a_samples_reshape), self.critic.q2(s_reshape, a_samples_reshape))
        losses = [F.mse_loss(q_value, q_pseudo_reshape) for q_value in q_values]
        return sum(losses)

    def loss_critic(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        sn: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        # loss = TD + alpha * Punish
        # source use same noise target
        q_target = self.td_target(sn, r, d)
        loss_td = self.loss_td_with_target(s, a, q_target)
        loss_aspl = self.loss_punish_with_target(s, a, q_target)
        return loss_td + self.alpha * loss_aspl


    # ====================
    # actor mathmatics
    # ====================  
    def loss_td3_variant(self, s:torch.Tensor) -> torch.Tensor:
        # TD3 + action noise
        a = self.actor.noise_action(self.actor.pi(s))
        q = self.critic.q1(s, a)
        return -q.mean() # mean over batch



