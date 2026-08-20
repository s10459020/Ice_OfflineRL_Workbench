import torch
from torch.nn import functional as F

from joint_learning.lib.dataset import Batch


class NAgent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        lambda_n: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, **kwargs)
        self.lambda_n = lambda_n

    # ====================
    # Loss functions
    # ====================
    def alpha_norm(self, batch: Batch) -> torch.Tensor:
        # alpha_norm = \lambda_N / E_(s \sim D) [|Q_(opt) (s, \pi(s))|]
        observations, _, _, _, _ = batch
        actions = self.actor(observations)
        q = self.critic.q_min(observations, actions)
        return self.lambda_n / q.abs().mean().detach()

    def loss_opt(self, batch: Batch) -> torch.Tensor:
        # Loss_opt = -alpha_norm E_(s \sim D) [Q_(opt) (s, \pi(s))]
        return self.alpha_norm(batch) * super().loss_opt(batch)


class GPAgent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        lambda_gp: float = 1.0,
        k_gp: int = 16,
        delta_gp: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, **kwargs)
        self.lambda_gp = lambda_gp
        self.k_gp = k_gp
        self.delta_gp = delta_gp

    # ====================
    # Loss functions
    # ====================
    def loss_gradient(self, batch: Batch) -> torch.Tensor:
        # Loss_GP = E_(s \sim D) [(1/K_(GP))\sum_k \sum _(i = 1)^2 ReLU(\|\nabla_(a\tilde_k) Q_i (s, a\tilde_k)\|_2 - \delta_(GP))^2]
        observations, _, _, _, _ = batch
        sampled_actions = self.actor.sample_uniform(observations, self.k_gp).requires_grad_(True)
        q_values = self.critic.q_all_n(observations, sampled_actions)
        penalties = []
        for q in q_values:
            gradient = torch.autograd.grad(
                q.sum(),
                sampled_actions,
                create_graph=True,
                retain_graph=True,
            )[0]
            penalties.append(F.relu(gradient.norm(p=2, dim=-1) - self.delta_gp).square())
        return torch.stack(penalties, dim=0).sum(dim=0).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_(base) + \lambda_(GP) Loss_GP
        return super().loss_critic(batch) + self.lambda_gp * self.loss_gradient(batch)


class CAgent:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        lambda_comp: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, **kwargs)
        self.lambda_comp = lambda_comp

    def loss_compensation(self, batch: Batch) -> torch.Tensor:
        # Loss_compensation = -E_((s, a) \sim D) [(1/2)\sum _(i = 1)^2 Q_i (s, a)]
        observations, actions, _, _, _ = batch
        return -self.critic.q_mean(observations, actions).mean()

    def loss_critic(self, batch: Batch) -> torch.Tensor:
        # Loss_Critic = Loss_(base) + \lambda_(comp) Loss_compensation
        return super().loss_critic(batch) + self.lambda_comp * self.loss_compensation(batch)
