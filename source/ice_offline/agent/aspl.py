from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.stats import qmc

from ice_offline.agent.td3 import TD3Actor
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3 import TD3Critic
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


@dataclass
class AsplAgent(TD3Agent):
    max_action: float = 1.0
    tau: float = 0.005
    alpha: float = 2.5
    q_mean: float = 0
    noise_clip: float = 0.5
    noise_scale: float = 0.2
    learning_rate: float = 3e-4
    num_sample: int = 1

    def __post_init__(self) -> None:
        self.actor = TD3Actor(
            self.obs_size,
            self.act_size,
            tau=self.tau,
            noise_scale=self.noise_scale,
            noise_clip=self.noise_clip,
            max_action=self.max_action,
            pi_cls=_Pi,
        ).to(self.device)
        self.critic = TD3Critic(
            self.obs_size,
            self.act_size,
            q_count=self.q_count,
            tau=self.tau,
            q_cls=_Q,
        ).to(self.device)
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_size)
        self.actor_learning_rate = self.learning_rate
        self.critic_learning_rate = self.learning_rate

        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(), 
            lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.q_networks.parameters(),
            lr=self.learning_rate,
        )

        # ignore use_lr_scheduler


    # ====================
    # Update
    # ====================
    def set_seed(self, seed: int) -> None:
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_size, seed=seed)

    def update(self, batch: TorchBuffer):
        s = batch.obs_list
        a = batch.act_list
        r = batch.rew_list.view(-1, 1)
        d = batch.done_list.view(-1, 1)
        sn = batch.next_obs_list

        self.update_step += 1
        self.update_critic(s, a, r, sn, d)

        if self.update_step % self.update_actor_interval == 0:
            self.update_actor(s)
            self.critic.update_target_soft()
            self.actor.update_target_soft()

    def update_actor(self, s: torch.Tensor) -> None:
        self.actor_optimizer.zero_grad()
        actor_loss = self.loss_td3_variant(s)
        actor_loss.backward()
        self.actor_optimizer.step()
  
    # ====================
    # common
    # ====================
    def _sample_lhs_actions(self, batch_size: int, num_samples: int) -> torch.Tensor:
        samples = self._lhs_sampler.random(n=num_samples) # (N, A)[0 ~ 1]

        samples = qmc.scale(
            samples, 
            [-self.max_action] * self.act_size, 
            [self.max_action] * self.act_size
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
    def target_td3(self, sn: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
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
        q1 = self.critic.q_networks[0](o, a)
        q2 = self.critic.q_networks[1](o, a)
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
        
        q_values = (
            self.critic.q_networks[0](s_reshape, a_samples_reshape),
            self.critic.q_networks[1](s_reshape, a_samples_reshape),
        )
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
        q_target = self.target_td3(sn, r, d)
        loss_td = self.loss_td_with_target(s, a, q_target)
        loss_aspl = self.loss_punish_with_target(s, a, q_target)
        return loss_td + self.alpha * loss_aspl


    # ====================
    # actor mathmatics
    # ====================  
    def loss_td3_variant(self, s:torch.Tensor) -> torch.Tensor:
        # TD3 + action noise
        a = self.actor.noise_action(self.actor.pi(s))
        q = self.critic.q_networks[0](s, a)
        return -q.mean() # mean over batch



