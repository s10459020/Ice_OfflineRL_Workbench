import torch

from joint_learning.lib.dataset import Batch


class NAgent:
    # ====================
    # Loss functions
    # ====================
    def loss_td3(self, batch: Batch) -> torch.Tensor:
        # Loss_TD3-N = -E_(s \sim D) [Q(s, \pi(s))]/E_(s \sim D) [|Q(s, \pi(s))|]
        observations, _, _, _, _ = batch
        q = self.actor_q(observations)
        return -q.mean() / q.abs().mean().detach()
