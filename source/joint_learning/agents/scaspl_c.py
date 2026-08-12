import torch

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scaspl import SCASPLAgent
from joint_learning.lib.dataset import Batch


class SCASPLCAgent(SCASPLAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamics: SCASDynamics,
        weight_compensate: float = 1.0,
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
        # It offsets pseudo-label punishment by lifting dataset-action values.
        observations, actions, _, _, _ = batch
        q = torch.stack([self.q1(observations, actions), self.q2(observations, actions)], dim=0)
        return -q.mean()

    def loss_q(self, batch: Batch) -> torch.Tensor:
        # SCASPL-C critic loss:
        # Loss_Q = Loss_TD + \lambda_p * Loss_Pseudo_Label_Constraint + \lambda_c * Loss_Compensation
        return super().loss_q(batch) + self.weight_compensate * self.loss_compensation(batch)
