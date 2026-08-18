import torch
from torch.nn import functional as F

from joint_learning.agents.td3 import TD3Agent
from joint_learning.lib.dataset import Batch


class TD3BCAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, lambda_td3: float = 2.5, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, device=device)
        self.lambda_td3 = lambda_td3

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_bc(self, batch: Batch) -> torch.Tensor:
        # Loss_BC = E_D [\|\pi(s)-a\|_2^2]
        observations, actions, _, _, _ = batch
        predicted_actions = self.actor(observations)
        return F.mse_loss(predicted_actions, actions)

    def loss_normalized(self, batch: Batch) -> torch.Tensor:
        # Loss_normalized = -E[Q_(min) (s,\pi(s))]/E[|Q_(min) (s,\pi(s))|]
        observations, _, _, _, _ = batch
        policy_actions = self.actor(observations)
        q = self.critic.q_min(observations, policy_actions)
        return -q.mean() / q.abs().mean().detach()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = \lambda_TD3 Loss_normalized + Loss_BC
        return self.lambda_td3 * self.loss_normalized(batch) + self.loss_bc(batch)
