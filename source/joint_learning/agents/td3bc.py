import torch
from torch.nn import functional as F

from joint_learning.agents.td3 import TD3Agent
from joint_learning.agents.variant import GPAgent
from joint_learning.lib.dataset import Batch


class TD3BCAgent(TD3Agent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        lambda_n: float = 2.5,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, **kwargs)
        self.lambda_n = lambda_n

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_bc(self, batch: Batch) -> torch.Tensor:
        # L_BC = E_((s, a) \sim D) [(\pi(s) - a)^2]
        observations, actions, _, _, _ = batch
        predicted_actions = self.actor(observations)
        return F.mse_loss(predicted_actions, actions)

    def alpha_norm(self, batch: Batch) -> torch.Tensor:
        # alpha_norm = \lambda_N / E_(s \sim D) [|Q_1 (s, \pi(s))|]
        observations, _, _, _, _ = batch
        actions = self.actor(observations)
        q = self.critic.q1(observations, actions)
        return self.lambda_n / q.abs().mean().detach()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # L_actor = alpha_norm L_opt + L_BC
        return self.alpha_norm(batch) * self.loss_opt(batch) + self.loss_bc(batch)


class TD3BCXNAgent(TD3BCAgent):
    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # L_actor = \lambda_N L_opt + L_BC
        return self.lambda_n * self.loss_opt(batch) + self.loss_bc(batch)


class TD3BCPAgent(TD3BCXNAgent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        lambda_n: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, lambda_n=lambda_n, **kwargs)


class TD3BCGPAgent(GPAgent, TD3BCAgent):
    pass


class TD3BCXNGPAgent(GPAgent, TD3BCXNAgent):
    pass


class TD3BCPGPAgent(GPAgent, TD3BCPAgent):
    pass
