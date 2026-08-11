import torch

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scaspl import SCASPLAgent
from joint_learning.lib.dataset import Batch


class SCASPLGPAgent(SCASPLAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamics: SCASDynamics,
        weight_gp: float = 1.0,
        gp_count: int = 16,
        gp_threshold: float = 1.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, dynamics=dynamics, device=device)
        self.weight_gp = weight_gp
        self.gp_count = gp_count
        self.gp_threshold = gp_threshold

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # SCASPL-GP critic loss:
        #   L_Q = L_TD + lambda_p * L_PL + lambda_gp * L_GP.
        return super().loss_critic(batch) + self.weight_gp * self.loss_gradient_penalty(
            batch,
            self.gp_count,
            self.gp_threshold,
        )
