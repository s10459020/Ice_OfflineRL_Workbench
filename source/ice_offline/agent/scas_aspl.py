from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ice_offline.agent.aspl import AsplActor
from ice_offline.agent.aspl import AsplCritic
from ice_offline.agent.scas_min import ScasMinAgent
from ice_offline.dataset._types import Batch


@dataclass
class ScasAsplAgent(ScasMinAgent):
    aspl_alpha: float = 2.5

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
        self.dynamics = self.dynamics.prepare().to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.pi.parameters(),
            lr=self.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.q_networks.parameters(),
            lr=self.critic_learning_rate,
        )

    # ====================
    # Critic loss
    # ====================
    def loss_td_with_target(self, s: torch.Tensor, a: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        # loss = E{s,a,r,s'~D}[ MSE(Q(s,a) - y) ]
        return sum(F.mse_loss(q, q_target) for q in self.critic.q_all(s, a))

    def loss_punish_with_target(self, s: torch.Tensor, a: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        # E_{s~D,a~U}[ Q(s,a~) - Q~(s,a~) ]^2
        a_samples = self.actor.sample_actions_lhs(s.shape[0]) # (N, B, A)
        action_distance = self.actor.action_distance(a, a_samples) # (N, B, 1)
        q_pseudo = self.critic.q_pseudo(self.update_step, q_target, action_distance) # (N, B, 1)

        s_reshape = s.unsqueeze(0).expand(self.actor.num_sample, -1, -1).reshape(-1, s.shape[1])
        a_samples_reshape = a_samples.view(-1, a.shape[1])
        q_pseudo_reshape = q_pseudo.view(-1, 1)

        q_values = self.critic.q_all(s_reshape, a_samples_reshape)
        return sum(F.mse_loss(q, q_pseudo_reshape) for q in q_values)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # loss = L_TD3 + alpha_ASPL * L_ASPL
        s, a, r, sn, d = batch
        q_target = self.target_td3(sn, r, d)
        loss_td = self.loss_td_with_target(s, a, q_target)
        loss_punish = self.loss_punish_with_target(s, a, q_target)
        return loss_td + self.aspl_alpha * loss_punish
