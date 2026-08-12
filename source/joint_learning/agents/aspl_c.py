import torch

from joint_learning.agents.aspl import ASPLAgent
from joint_learning.lib.dataset import Batch


class ASPLCAgent(ASPLAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        weight_compensate: float = 1.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, device=device)
        self.weight_compensate = weight_compensate

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_compensation(self, batch: Batch) -> torch.Tensor:
        # Compensation term:
        # Loss_Compensation = -E_D [(1/2)\sum_(i=1)^2 Q_i (s,a)]
        # Maximizing dataset-action Q prevents ASPL punishment from only pushing sampled actions down.
        observations, actions, _, _, _ = batch
        q = torch.stack([self.q1(observations, actions), self.q2(observations, actions)], dim=0)
        return -q.mean()

    def loss_q(self, batch: Batch) -> torch.Tensor:
        # ASPL-C critic loss:
        # Loss_Q = Loss_TD + \lambda_p * Loss_Pseudo_Label_Constraint + \lambda_c * Loss_Compensation
        return (
            self.loss_td(batch)
            + self.weight_punish * self.loss_pseudo_label_constraint(batch)
            + self.weight_compensate * self.loss_compensation(batch)
        )
