import torch

from joint_learning.agents.td3bc_xn import TD3BCXNAgent
from joint_learning.datasets.lib import Batch


class TD3BCXNGPAgent(TD3BCXNAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        weight_gp: float = 1.0,
        gp_count: int = 16,
        gp_threshold: float = 1.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, device=device)
        self.weight_gp = weight_gp
        self.gp_count = gp_count
        self.gp_threshold = gp_threshold

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # TD3BC-XN-GP critic loss:
        #   L_Q = L_TD + lambda_gp * L_GP.
        # The actor uses TD3BC-XN: no normalization, original TD3BC actor weight.
        return self.loss_td(batch) + self.weight_gp * self.loss_gradient_penalty(
            batch,
            self.gp_count,
            self.gp_threshold,
        )
