import torch
from torch.nn import functional as F

from joint_learning.agents.td3 import TD3Agent
from joint_learning.agents.variant import GPAgent
from joint_learning.lib.dataset import Batch


class TD3BCAgent(TD3Agent):
    def __init__(self, obs_size: int, act_size: int, lambda_n: float = 2.5, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, device=device)
        self.lambda_n = lambda_n

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_bc(self, batch: Batch) -> torch.Tensor:
        # Loss_BC = E_((s, a) \sim D) [\|\pi(s) - a\|_2^2]
        observations, actions, _, _, _ = batch
        predicted_actions = self.actor(observations)
        return F.mse_loss(predicted_actions, actions)

    def loss_normalized(self, batch: Batch) -> torch.Tensor:
        # Loss_normalized = -E_(s \sim D) [Q_1 (s, \pi(s))]/E_(s \sim D) [|Q_1 (s, \pi(s))|]
        observations, _, _, _, _ = batch
        q = self.actor_q(observations)
        return -q.mean() / q.abs().mean().detach()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = \lambda_n Loss_normalized + Loss_BC
        return self.lambda_n * self.loss_normalized(batch) + self.loss_bc(batch)


class TD3BCXNAgent(TD3BCAgent):
    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = \lambda_n Loss_TD3 + Loss_BC
        return self.lambda_n * self.loss_td3(batch) + self.loss_bc(batch)


class TD3BCPAgent(TD3BCXNAgent):
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, lambda_n=0.01, device=device)


class TD3BCGPAgent(GPAgent, TD3BCAgent):
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, device=device)


class TD3BCXNGPAgent(GPAgent, TD3BCXNAgent):
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, device=device)


class TD3BCPGPAgent(GPAgent, TD3BCPAgent):
    def __init__(self, obs_size: int, act_size: int, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, device=device)
