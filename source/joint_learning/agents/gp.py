import torch
from torch.nn import functional as F

from joint_learning.lib.dataset import Batch


class GPAgent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        lambda_gp: float = 1.0,
        gp_count: int = 16,
        gp_threshold: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, **kwargs)
        self.lambda_gp = lambda_gp
        self.gp_count = gp_count
        self.gp_threshold = gp_threshold

    # ====================
    # Loss functions
    # ====================
    def loss_gradient(self, batch: Batch) -> torch.Tensor:
        # Loss_GP = E_(s \sim D) [(1/K)\sum_k \sum _(i = 1)^2 ReLU(\|\nabla_(a\tilde_k) Q_i (s, a\tilde_k)\|_2 - \tau_(GP))^2]
        observations, _, _, _, _ = batch
        sampled_actions = self.actor.sample_uniform(observations, self.gp_count).requires_grad_(True)
        q_values = self.critic.q_all_n(observations, sampled_actions)
        penalties = []
        for q in q_values:
            gradient = torch.autograd.grad(
                q.sum(),
                sampled_actions,
                create_graph=True,
                retain_graph=True,
            )[0]
            penalties.append(F.relu(gradient.norm(p=2, dim=-1) - self.gp_threshold).square())
        return torch.stack(penalties, dim=0).sum(dim=0).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_(base) + \lambda_(GP) Loss_GP
        return super().loss_critic(batch) + self.lambda_gp * self.loss_gradient(batch)
