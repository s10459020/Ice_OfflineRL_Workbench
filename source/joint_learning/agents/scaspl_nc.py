import torch

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scaspl_n import SCASPLNAgent
from joint_learning.datasets.lib import Batch


class SCASPLNCAgent(SCASPLNAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamics: SCASDynamics,
        weight_compensate: float = 10.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, dynamics=dynamics, device=device)
        self.weight_compensate = weight_compensate

    def loss_compensate(self, batch: Batch) -> torch.Tensor:
        # Compensation term:
        #   L_C = -E[mean_i Q_i(s, a_data)].
        # SCASPL-NC keeps SCASPL-N actor normalization and adds critic compensation.
        observations, actions, _, _, _ = batch
        q = torch.stack([self.q1(observations, actions), self.q2(observations, actions)], dim=0)
        return -q.mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # SCASPL-NC critic loss:
        #   L_Q = L_TD + lambda_p * L_PL + lambda_c * L_C.
        return super().loss_critic(batch) + self.weight_compensate * self.loss_compensate(batch)
