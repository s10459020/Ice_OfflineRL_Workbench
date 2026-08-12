import torch

from joint_learning.agents.td3bc import TD3BCAgent
from joint_learning.lib.dataset import Batch


class TD3BCXNAgent(TD3BCAgent):
    def __init__(self, obs_size: int, act_size: int, weight_td3: float = 2.5, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, weight_td3=weight_td3, device=device)

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_policy(self, batch: Batch) -> torch.Tensor:
        # TD3BC-XN actor loss:
        # Loss_Policy = \alpha * Loss_TD3 + Loss_Behavior_Cloning
        # Loss_TD3 = -E[ Q_(min)(s,\pi(s))]
        # XN removes the original TD3BC normalization while keeping the original actor weight.
        return self.weight_td3 * self.loss_td3(batch) + self.loss_behavior_cloning(batch)
