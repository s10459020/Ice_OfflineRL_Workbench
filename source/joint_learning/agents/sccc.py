import math

import torch
from torch.nn import functional as F

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.scas import SCASAgent
from joint_learning.lib.dataset import Batch


class SCCCAgent(SCASAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamic: Dynamic,
        lambda_q: float = 10.0,
        conservative_count: int = 16,
        threshold: float = 2.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)
        self.lambda_q = lambda_q
        self.conservative_count = conservative_count
        self.threshold = threshold

    # -------------------------------------------------------------------------
    # Help functions
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_conservative(self, batch: Batch) -> torch.Tensor:
        # Loss_conservative=\sum _(i=1)^2 E_D [ReLU(log \sum_n exp(Q_i (s,a\tilde_n))-log N-Q_i (s,a)+\tau)]
        observations, actions, _, _, _ = batch
        batch_size = observations.shape[0]
        sampled_actions = self.actor.sample_uniform(observations, self.conservative_count)
        flat_actions = sampled_actions.reshape(batch_size * self.conservative_count, self.act_size)
        flat_observations = observations.unsqueeze(1).expand(-1, self.conservative_count, -1)
        flat_observations = flat_observations.reshape(batch_size * self.conservative_count, self.obs_size)

        losses = []
        q_data_all = self.critic.q_all(observations, actions)
        q_sample_all = self.critic.q_all(flat_observations, flat_actions)
        for q_data, q_sample in zip(q_data_all, q_sample_all):
            q_sample = q_sample.view(batch_size, self.conservative_count, 1)
            penalty_value = torch.logsumexp(q_sample, dim=1) - math.log(self.conservative_count)
            losses.append(F.relu(penalty_value - q_data + self.threshold).mean())
        return sum(losses)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_TD + \lambda_Q Loss_conservative
        return self.loss_td(batch) + self.lambda_q * self.loss_conservative(batch)
