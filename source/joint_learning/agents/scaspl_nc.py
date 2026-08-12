import torch

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scaspl_n import SCASPLNAgent
from joint_learning.lib.dataset import Batch


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

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_compensation(self, batch: Batch) -> torch.Tensor:
        # Compensation term:
        # Loss_Compensation = -E_D [(1/2)\sum_(i=1)^2 Q_i (s,a)]
        # SCASPL-NC keeps SCASPL-N actor normalization and adds critic compensation.
        observations, actions, _, _, _ = batch
        q = torch.stack([self.q1(observations, actions), self.q2(observations, actions)], dim=0)
        return -q.mean()

    def loss_q(self, batch: Batch) -> torch.Tensor:
        # SCASPL-NC critic loss:
        # Loss_Q = Loss_TD + \lambda_p * Loss_Pseudo_Label_Constraint + \lambda_c * Loss_Compensation
        return super().loss_q(batch) + self.weight_compensate * self.loss_compensation(batch)
