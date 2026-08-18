import torch

from joint_learning.agents.aspl import ASPLAgent
from joint_learning.lib.dataset import Batch


class ASPLCAgent(ASPLAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        lambda_c: float = 1.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, device=device)
        self.lambda_c = lambda_c

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_compensation(self, batch: Batch) -> torch.Tensor:
        # Loss_compensation = -E_((s, a) \sim D) [(1/2)\sum _(i = 1)^2 Q_i (s, a)]
        observations, actions, _, _, _ = batch
        return -self.critic.q_mean(observations, actions).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_TD + \lambda_p Loss_pseudo + \lambda_c Loss_compensation
        return (
            self.loss_td(batch)
            + self.lambda_p * self.loss_pseudo(batch)
            + self.lambda_c * self.loss_compensation(batch)
        )
