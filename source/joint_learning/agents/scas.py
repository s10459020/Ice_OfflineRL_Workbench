import torch

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.td3 import TD3Agent
from joint_learning.datasets.lib import Batch


class SCASAgent(TD3Agent):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        dynamics: SCASDynamics,
        weight_correction: float = 0.25,
        scale_gap: float = 5.0,
        max_gap: float = 50.0,
        device: str = "cuda",
    ) -> None:
        super().__init__(obs_size, act_size, device=device)
        self.dynamics = dynamics
        self.weight_correction = weight_correction
        self.scale_gap = scale_gap
        self.max_gap = max_gap

    def loss_correction(self, batch: Batch) -> torch.Tensor:
        # SCAS correction:
        #   w = clip(exp(beta * (V(s') - V(s))), max=w_max)
        #   L_SCAS = E[w * ||M(s_noise, pi(s_noise)) - s'||^2].
        # V(s) is approximated by min_i Q_i(s, pi(s)).
        observations, _, _, next_observations, _ = batch
        actions = self.policy(observations)
        next_actions = self.policy(next_observations)
        value = torch.minimum(self.q1(observations, actions), self.q2(observations, actions))
        next_value = torch.minimum(self.q1(next_observations, next_actions), self.q2(next_observations, next_actions))
        weight = (self.scale_gap * (next_value.detach() - value.detach())).exp().clamp(max=self.max_gap)

        noisy_observations = self.dynamics.noisy_observation(observations)
        noisy_actions = self.policy(noisy_observations)
        predicted_next_observations = self.dynamics.next_observation(noisy_observations, noisy_actions)
        return (weight * ((predicted_next_observations - next_observations) ** 2)).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # SCAS actor loss:
        #   L_pi = (1 - lambda_s) * L_TD3 + lambda_s * L_SCAS.
        return (1.0 - self.weight_correction) * self.loss_td3(batch) + self.weight_correction * self.loss_correction(batch)
