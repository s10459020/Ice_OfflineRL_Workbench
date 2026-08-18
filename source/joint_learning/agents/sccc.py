import math

import torch
from torch.nn import functional as F

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.scas import SCASAgent
from joint_learning.agents.variant import GPAgent, NAgent
from joint_learning.lib.dataset import Batch


class SCCCAgent(SCASAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamic: Dynamic,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)
        self.lambda_q = 10.0
        self.conservative_sample_count = 16
        self.conservative_threshold = 2.0

    # ====================
    # Loss functions
    # ====================
    def loss_conservative(self, batch: Batch) -> torch.Tensor:
        # Loss_conservative = \sum _(i = 1)^2 E_((s, a) \sim D) [ReLU(log \sum_n exp(Q_i (s, a\tilde_n)) - log N - Q_i (s, a) + \tau)]
        observations, actions, _, _, _ = batch
        batch_size = observations.shape[0]
        sampled_actions = self.actor.sample_uniform(observations, self.conservative_sample_count)
        losses = []
        q_data_all = self.critic.q_all(observations, actions)
        q_sample_all = self.critic.q_all_n(observations, sampled_actions)
        for q_data, q_sample in zip(q_data_all, q_sample_all):
            penalty_value = torch.logsumexp(q_sample, dim=1) - math.log(self.conservative_sample_count)
            losses.append(F.relu(penalty_value - q_data + self.conservative_threshold).mean())
        return sum(losses)

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_TD + \lambda_Q Loss_conservative
        return self.loss_td(batch) + self.lambda_q * self.loss_conservative(batch)


class SCCCNAgent(NAgent, SCCCAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)
        self.lambda_a = 0.005


class SCCCGPAgent(GPAgent, SCCCAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)


class SCCCGPNAgent(GPAgent, SCCCNAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)
        self.lambda_a = 0.01
