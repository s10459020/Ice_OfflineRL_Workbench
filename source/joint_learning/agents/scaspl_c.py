import torch

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.scaspl import SCASPLAgent
from joint_learning.lib.dataset import Batch


class SCASPLCAgent(SCASPLAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamic: Dynamic,
        lambda_c: float = 1.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)
        self.lambda_c = lambda_c

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_compensation(self, batch: Batch) -> torch.Tensor:
        # Loss_compensation = -E_D [(1/2)\sum _(i=1)^2 Q_i (s,a)]
        observations, actions, _, _, _ = batch
        return -self.critic.q_mean(observations, actions).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_TD + \lambda_p Loss_pseudo + \lambda_c Loss_compensation
        return super().loss_critic(batch) + self.lambda_c * self.loss_compensation(batch)
