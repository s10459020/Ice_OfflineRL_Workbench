import torch
from torch.nn import functional as F

from joint_learning.lib.dataset import Batch


class NAgent:
    lambda_N = 1.0

    # ====================
    # Loss functions
    # ====================
    def loss_td3(self, batch: Batch) -> torch.Tensor:
        # alpha = lambda_N / E_(s \sim D) [|Q_(min) (s, \pi(s))|]
        # Loss_TD3-N = -alpha E_(s \sim D) [Q_(min) (s, \pi(s))]
        observations, _, _, _, _ = batch
        actions = self.actor(observations)
        q = self.critic.q_min(observations, actions)
        alpha = self.lambda_N / q.abs().mean().detach()
        return -alpha * q.mean()


class GPAgent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        lambda_gp: float = 1.0,
        gp_sample_count: int = 16,
        gradient_threshold: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, **kwargs)
        self.lambda_gp = lambda_gp
        self.gp_sample_count = gp_sample_count
        self.gradient_threshold = gradient_threshold

    # ====================
    # Loss functions
    # ====================
    def loss_gradient(self, batch: Batch) -> torch.Tensor:
        # Loss_GP = E_(s \sim D) [(1/K)\sum_k \sum _(i = 1)^2 ReLU(\|\nabla_(a\tilde_k) Q_i (s, a\tilde_k)\|_2 - \delta_(GP))^2]
        observations, _, _, _, _ = batch
        sampled_actions = self.actor.sample_uniform(observations, self.gp_sample_count).requires_grad_(True)
        q_values = self.critic.q_all_n(observations, sampled_actions)
        penalties = []
        for q in q_values:
            gradient = torch.autograd.grad(
                q.sum(),
                sampled_actions,
                create_graph=True,
                retain_graph=True,
            )[0]
            penalties.append(F.relu(gradient.norm(p=2, dim=-1) - self.gradient_threshold).square())
        return torch.stack(penalties, dim=0).sum(dim=0).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_(base) + \lambda_(gp) Loss_GP
        return super().loss_critic(batch) + self.lambda_gp * self.loss_gradient(batch)


class CAgent:
    lambda_c = 1.0

    def loss_compensation(self, batch: Batch) -> torch.Tensor:
        # Loss_compensation = -E_((s, a) \sim D) [(1/2)\sum _(i = 1)^2 Q_i (s, a)]
        observations, actions, _, _, _ = batch
        return -self.critic.q_mean(observations, actions).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_(base) + \lambda_c Loss_compensation
        return super().loss_critic(batch) + self.lambda_c * self.loss_compensation(batch)
