import torch

from joint_learning.agents.dynamics import Dynamic
from joint_learning.agents.td3 import TD3Agent
from joint_learning.lib.dataset import Batch


class SCASAgent(TD3Agent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamic: Dynamic,
        lambda_s: float = 0.25,
        scale_gap: float = 5.0,
        max_gap: float = 50.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, device=device)
        self.dynamic = dynamic
        self.lambda_s = lambda_s
        self.scale_gap = scale_gap
        self.max_gap = max_gap

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_td3(self, batch: Batch) -> torch.Tensor:
        observations, _, _, _, _ = batch
        actions = self.actor(observations)
        q = self.critic.q_min(observations, actions)
        return -q.mean()

    def loss_correction(self, batch: Batch) -> torch.Tensor:
        # V(s) = Q_(min) (s,\pi(s)), \Delta V = V(s') - V(s)
        # w(s,s') = min(exp(\beta \Delta V), w_(max))
        # s\hat = s + \epsilon, \epsilon \sim N(0, \sigma^2 I)
        # Loss_correction = E_D [w(s,s')\|M(s\hat,\pi(s\hat))-s'\|_2^2]
        observations, _, _, next_observations, _ = batch
        actions = self.actor(observations)
        next_actions = self.actor(next_observations)
        value = self.critic.q_min(observations, actions)
        next_value = self.critic.q_min(next_observations, next_actions)
        weight = (self.scale_gap * (next_value.detach() - value.detach())).exp().clamp(max=self.max_gap)

        noisy_observations = self.dynamic.noisy_observation(observations)
        noisy_actions = self.actor(noisy_observations)
        predicted_next_observations = self.dynamic(noisy_observations, noisy_actions)
        return (weight * ((predicted_next_observations - next_observations) ** 2)).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = (1-\lambda_s) Loss_TD3 + \lambda_s Loss_correction
        return (1.0 - self.lambda_s) * self.loss_td3(batch) + self.lambda_s * self.loss_correction(batch)
