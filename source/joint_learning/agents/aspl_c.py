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

    def loss_compensate(self, batch: Batch) -> torch.Tensor:
        # Compensation term:
        #   L_C = -E[mean_i Q_i(s, a_data)].
        # Maximizing dataset-action Q prevents ASPL punishment from only pushing sampled actions down.
        observations, actions, _, _, _ = batch
        q = torch.stack([self.q1(observations, actions), self.q2(observations, actions)], dim=0)
        return -q.mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # ASPL-C critic loss:
        #   L_Q = L_TD + lambda_p * L_ASPL + lambda_c * L_C.
        return (
            self.loss_td(batch)
            + self.weight_punish * self.loss_punish(batch)
            + self.weight_compensate * self.loss_compensate(batch)
        )
