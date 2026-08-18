import torch

from joint_learning.agents.td3bc import TD3BCAgent
from joint_learning.lib.dataset import Batch


class TD3BCXNAgent(TD3BCAgent):
    def __init__(self, obs_size: int, act_size: int, lambda_td3: float = 2.5, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, lambda_td3=lambda_td3, device=device)

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = \lambda_TD3 Loss_TD3 + Loss_BC
        return self.lambda_td3 * self.loss_td3(batch) + self.loss_bc(batch)
