import torch

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.scas import SCASAgent
from joint_learning.datasets.lib import Batch


class SCASGPAgent(SCASAgent):
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

    def update(self, batch: Batch) -> None:
        self.update_step += 1
        loss_critic = self.loss_critic(batch)
        self.q_optimizer.zero_grad()
        loss_critic.backward()
        self.q_optimizer.step()

        if self.update_step % self.update_actor_interval == 0:
            loss_actor = self.loss_actor(batch)
            self.policy_optimizer.zero_grad()
            loss_actor.backward()
            self.policy_optimizer.step()
            self.sync_target_soft()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # SCAS-GP critic loss:
        #   L_Q = L_TD + lambda_gp * L_GP.
        return self.loss_td(batch) + self.weight_gp * self.loss_gradient_penalty(
            batch,
            self.gp_count,
            self.gp_threshold,
        )
