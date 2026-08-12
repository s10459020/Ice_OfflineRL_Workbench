import torch
from torch.nn import functional as F

from joint_learning.agents.td3 import TD3Agent
from joint_learning.lib.dataset import Batch


class TD3BCAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, weight_td3: float = 2.5, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, device=device)
        self.weight_td3 = weight_td3

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_behavior_cloning(self, batch: Batch) -> torch.Tensor:
        # Behavior cloning regularizer:
        # Loss_Behavior_Cloning = E_D [\|\pi(s)-a\|_2^2 ]
        # This keeps the learned policy close to dataset actions in offline RL.
        observations, actions, _, _, _ = batch
        predicted_actions = self.policy(observations)
        return F.mse_loss(predicted_actions, actions)

    def loss_policy(self, batch: Batch) -> torch.Tensor:
        # Original TD3BC actor loss:
        # Loss_Policy = \alpha * Loss_Normalized_Policy + Loss_Behavior_Cloning
        # Loss_Normalized_Policy = -E[Q_(min)(s,\pi(s))]/E[|Q_(min)(s,\pi(s))|]
        # In the paper, this normalized objective is the official td3bc model.
        return self.weight_td3 * self.loss_normalized_policy(batch) + self.loss_behavior_cloning(batch)
