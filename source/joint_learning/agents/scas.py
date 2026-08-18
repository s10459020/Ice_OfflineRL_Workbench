import torch

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.td3 import TD3Agent
from joint_learning.agents.variant import GPAgent, NAgent
from joint_learning.lib.dataset import Batch


class SCASAgent(TD3Agent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamic: Dynamic,
        lambda_a: float = 0.25,
        value_gap_scale: float = 5.0,
        max_weight: float = 50.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, device=device)
        self.dynamic = dynamic
        self.lambda_a = lambda_a
        self.value_gap_scale = value_gap_scale
        self.max_weight = max_weight

    # -------------------------------------------------------------------------
    # Help functions
    # -------------------------------------------------------------------------
    def actor_q(self, observations: torch.Tensor) -> torch.Tensor:
        actions = self.actor(observations)
        return self.critic.q_min(observations, actions)

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_correction(self, batch: Batch) -> torch.Tensor:
        # V(s) = Q_(min) (s, \pi(s)), \Delta V = V(s') - V(s)
        # w(s, s') = min(exp(\beta \Delta V), w_(max))
        # s\hat = s + \epsilon, \epsilon \sim N(0, \sigma^2 I)
        # Loss_correction = E_((s, a, r, s', d) \sim D) [w(s, s')\|M(s\hat, \pi(s\hat)) - s'\|_2^2]
        observations, _, _, next_observations, _ = batch
        value = self.actor_q(observations)
        next_value = self.actor_q(next_observations)
        weight = (self.value_gap_scale * (next_value.detach() - value.detach())).exp().clamp(max=self.max_weight)

        noisy_observations = self.dynamic.noisy_observation(observations)
        noisy_actions = self.actor(noisy_observations)
        predicted_next_observations = self.dynamic(noisy_observations, noisy_actions)
        return (weight * ((predicted_next_observations - next_observations) ** 2)).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = (1 - \lambda_A) Loss_TD3 + \lambda_A Loss_correction
        return (1.0 - self.lambda_a) * self.loss_td3(batch) + self.lambda_a * self.loss_correction(batch)


class SCASNAgent(NAgent, SCASAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, lambda_a=0.005, device=device)


class SCASGPAgent(GPAgent, SCASAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)


class SCASGPNAgent(GPAgent, SCASNAgent):
    def __init__(self, obs_size: int, act_size: int, dynamic: Dynamic, device: str = "cuda") -> None:
        super().__init__(obs_size, act_size, dynamic=dynamic, device=device)
        self.lambda_a = 0.002
