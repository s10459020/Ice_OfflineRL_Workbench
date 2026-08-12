import torch

from joint_learning.agents.dynamics import SCASDynamics
from joint_learning.agents.td3 import TD3Agent
from joint_learning.lib.dataset import Batch


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

    # -------------------------------------------------------------------------
    # Loss functions
    # -------------------------------------------------------------------------
    def loss_state_correction(self, batch: Batch) -> torch.Tensor:
        # Policy-value estimate and transition-improvement weight:
        # V(s) = Q_(min)(s,\pi(s)), \Delta V = V(s') - V(s)
        # w(s,s') = min(exp(\beta \Delta V), w_(max))
        # Noisy-state action prediction and SCAS correction:
        # s\hat = s + \epsilon, \epsilon \sim N(0, \sigma^2 I)
        # Loss_State_Correction = E_D [w(s,s')\|M(s\hat,\pi(s\hat))-s'\|_2^2 ]
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

    def loss_policy(self, batch: Batch) -> torch.Tensor:
        # SCAS actor loss:
        # Loss_Policy = (1 - \lambda_s) * Loss_TD3 + \lambda_s * Loss_State_Correction
        return (1.0 - self.weight_correction) * self.loss_td3(batch) + self.weight_correction * self.loss_state_correction(batch)
