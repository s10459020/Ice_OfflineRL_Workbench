from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.stats import qmc

from ice_offline.agent.td3 import TD3Actor
from ice_offline.agent.td3 import TD3Agent
from ice_offline.agent.td3 import TD3Critic
from ice_offline.dataset._types import Batch


class _Pi(torch.nn.Module):
    # source design
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
    # source design
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


class AsplActor(TD3Actor):
    def __init__(self, seed: int = 42, num_sample: int = 1, *args, **kwargs):
        super().__init__(*args, pi_cls=_Pi, **kwargs)
        self.num_sample = num_sample
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_size, seed=seed)

    def set_seed(self, seed: int) -> None:
        self._lhs_sampler = qmc.LatinHypercube(d=self.act_size, seed=seed)

    def sample_actions_lhs(self, batch_size: int) -> torch.Tensor:
        samples = self._lhs_sampler.random(n=self.num_sample) # (N, A)[0 ~ 1]

        samples = qmc.scale(
            samples, 
            [-self.max_action] * self.act_size, 
            [self.max_action] * self.act_size
        ) # [-a ~ a]

        samples = torch.as_tensor(
            samples,
            dtype=torch.float32,
            device=next(self.pi.parameters()).device
        )

        samples = samples.unsqueeze(1) # (N, 1, A)
        samples = samples.repeat(1, batch_size, 1) # (N, B, A)
        return samples

    def action_distance(self, a: torch.Tensor, a_samples: torch.Tensor) -> torch.Tensor:
        # d(a,a~) = [ (a - a~) / (a_max - a_min) ]^2
        # source use constant (a_max-a_min) ~= 2*a_max
        # d range: [-0.5, 0.5]
        # a shape: (B, A)
        # a_samples shape: (N, B, A)
        diff = (a - a_samples)**2 # (N, B, A) 
        normalize = (2 * self.max_action)**2
        result = diff / normalize 
        return result.mean(dim=2, keepdim=True) # (N, B, 1)


class AsplCritic(TD3Critic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, q_cls=_Q, **kwargs)
        self.moving_avg = 0.0

    def update_moving_avg(self, update_step: int, target: torch.Tensor) -> float:
        current = torch.abs(target).mean().item() # mean(B, 1) > (1)
        self.moving_avg = ((update_step - 1) * self.moving_avg + current) / update_step
        return self.moving_avg

    def q_pseudo(self, update_step: int, q_target: torch.Tensor, action_distance: torch.Tensor) -> torch.Tensor:
        # Q~(s,a~) = Q^(s,a) - c * d(a,a~)
        # c is scaling coefficent
        # q use q_target in source
        # q_target range: [-Q, Q], shape: (B, 1)
        with torch.no_grad():
            c = self.update_moving_avg(update_step, q_target) # (1)
            return q_target - c * action_distance  # (N, B, 1)


@dataclass
class AsplAgent(TD3Agent):
    alpha: float = 2.5
    learning_rate: float = 3e-4

    def __post_init__(self) -> None:
        self.actor = AsplActor(
            obs_size=self.obs_size,
            act_size=self.act_size,
        ).to(self.device)
        self.critic = AsplCritic(
            self.obs_size,
            self.act_size,
            q_count=self.q_count,
        ).to(self.device)
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

    def set_seed(self, seed: int) -> None:
        self.actor.set_seed(seed)

    def loss_td_with_target(self, batch: Batch, q_target: torch.Tensor) -> torch.Tensor:
        o, a, _, _, _ = batch
        # double Q TD
        q1 = self.critic.q_networks[0](o, a)
        q2 = self.critic.q_networks[1](o, a)
        loss_q1 = F.mse_loss(q1, q_target)
        loss_q2 = F.mse_loss(q2, q_target)
        return loss_q1 + loss_q2
    
    def loss_punish_with_target(self, batch: Batch, q_target: torch.Tensor) -> torch.Tensor:
        s, a, _, _, _ = batch
        # E_{s~D}{(a~)~U}[ Q(s,a~) - Q~(s,a~) ]^2
        a_samples = self.actor.sample_actions_lhs(s.shape[0])                                # (N,B,A)
        action_distance = self.actor.action_distance(a, a_samples)                            # (N,B,1)
        q_pseudo = self.critic.q_pseudo(self.update_step, q_target, action_distance)           # (N,B,1)

        # reshape
        s_reshape = s.unsqueeze(0).expand(self.actor.num_sample, -1, -1).reshape(-1, s.shape[1])  # (B,S) > (1,B,S) > (N,B,S) > (N*B,S)
        a_samples_reshape = a_samples.view(-1, a.shape[1])                                  # (N,B,A) > (N*B,A)
        q_pseudo_reshape = q_pseudo.view(-1, 1)                                             # (N*B,1)  
        
        q_values = (
            self.critic.q_networks[0](s_reshape, a_samples_reshape),
            self.critic.q_networks[1](s_reshape, a_samples_reshape),
        )
        losses = [F.mse_loss(q_value, q_pseudo_reshape) for q_value in q_values]
        return sum(losses)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # loss = TD + alpha * Punish
        # source use same noise target
        s, a, r, sn, d = batch
        q_target = self.target_td3(sn, r, d)
        loss_td = self.loss_td_with_target(batch, q_target)
        loss_aspl = self.loss_punish_with_target(batch, q_target)
        return loss_td + self.alpha * loss_aspl


    # ====================
    # actor mathmatics
    # ====================  
    def loss_td3(self, batch: Batch) -> torch.Tensor:
        # use q1 for actor update
        s, _, _, _, _ = batch
        a = self.actor.noise_action(self.actor.pi(s))
        q = self.critic.q_networks[0](s, a) 
        return -q.mean() # mean over batch




