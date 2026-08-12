import math

import torch
from torch.nn import functional as F

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scas import SCASAgent
from joint_learning.lib.dataset import Batch


class SCCAgent(SCASAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamics: SCASDynamics,
        weight_conservative: float = 10.0,
        conservative_count: int = 16,
        threshold: float = 2.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, dynamics=dynamics, device=device)
        self.weight_conservative = weight_conservative
        self.conservative_count = conservative_count
        self.threshold = threshold

    # -------------------------------------------------------------------------
    # Help functions
    # -------------------------------------------------------------------------
    def sample_conservative_actions(self, batch_size: int) -> torch.Tensor:
        # Conservative candidate actions:
        # a\tilde_n \sim U(-a_(max), a_(max))
        return torch.empty(
            (batch_size, self.conservative_count, self.act_size),
            dtype=torch.float32,
            device=self.device,
        ).uniform_(-self.max_action, self.max_action)

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_conservative_constraint(self, batch: Batch) -> torch.Tensor:
        # SCC conservative critic loss:
        # Loss_Conservative_Constraint = \sum_(i \in {1,2}) E_D [ReLU(log \sum_n exp(Q_i (s,a\tilde_n)) - log N - Q_i (s,a) + \tau)]
        # This raises a margin between sampled OOD actions and dataset actions.
        observations, actions, _, _, _ = batch
        batch_size = observations.shape[0]
        sampled_actions = self.sample_conservative_actions(batch_size)
        flat_actions = sampled_actions.reshape(batch_size * self.conservative_count, self.act_size)
        flat_observations = observations.unsqueeze(1).expand(-1, self.conservative_count, -1)
        flat_observations = flat_observations.reshape(batch_size * self.conservative_count, self.obs_size)

        losses = []
        for q_network in [self.q1, self.q2]:
            q_data = q_network(observations, actions)
            q_sample = q_network(flat_observations, flat_actions).view(batch_size, self.conservative_count, 1)
            penalty_value = torch.logsumexp(q_sample, dim=1) - math.log(self.conservative_count)
            losses.append(F.relu(penalty_value - q_data + self.threshold).mean())
        return sum(losses)

    def loss_q(self, batch: Batch) -> torch.Tensor:
        # SCC critic loss:
        # Loss_Q = Loss_TD + \alpha Loss_Conservative_Constraint
        return self.loss_td(batch) + self.weight_conservative * self.loss_conservative_constraint(batch)
