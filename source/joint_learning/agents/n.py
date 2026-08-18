import torch

from joint_learning.lib.dataset import Batch


class NAgent:
    def loss_td3(self, batch: Batch) -> torch.Tensor:
        observations, _, _, _, _ = batch
        q = self.actor_q(observations)
        return -q.mean() / q.abs().mean().detach()
