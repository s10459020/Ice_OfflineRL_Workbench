import torch

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.sccc import SCCCAgent
from joint_learning.lib.dataset import Batch


class SCCCNAgent(SCCCAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, lambda_q=10.0, device=device)
        self.lambda_s = 0.005

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_normalized(self, batch: Batch):
        # Loss_normalized = -E[Q_(min) (s,\pi(s))]/E[|Q_(min) (s,\pi(s))|]
        observations, _, _, _, _ = batch
        policy_actions = self.actor(observations)
        q = self.critic.q_min(observations, policy_actions)
        return -q.mean() / q.abs().mean().detach()

    def loss_actor(self, batch: Batch):
        # Loss_Actor = (1-\lambda_s) Loss_normalized + \lambda_s Loss_correction
        return (1.0 - self.lambda_s) * self.loss_normalized(batch) + self.lambda_s * self.loss_correction(batch)
