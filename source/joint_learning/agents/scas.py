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
        beta_scas: float = 5.0,
        sigma_s: float = 0.003,
        w_max: float = 50.0,
        **kwargs,
    ) -> None:
        super().__init__(obs_size, act_size, **kwargs)
        self.dynamic = dynamic
        self.lambda_a = lambda_a
        self.beta_scas = beta_scas
        self.sigma_s = sigma_s
        self.w_max = w_max

    # ====================
    # Help functions
    # ====================
    def perturb_observations(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        # s\hat = s + \epsilon, \epsilon \sim N(0, \sigma_s^2 I)
        return observations + torch.randn_like(observations) * self.sigma_s

    # ====================
    # Loss functions
    # ====================
    def loss_opt(self, batch: Batch) -> torch.Tensor:
        # Loss_opt = -E_(s \sim D) [Q_(min) (s, \pi(s))]
        observations, _, _, _, _ = batch
        actions = self.actor(observations)
        q = self.critic.q_min(observations, actions)
        return -q.mean()

    def loss_correction(self, batch: Batch) -> torch.Tensor:
        # V(s) = Q_(mean) (s, \pi(s)) = (Q_1 (s, \pi(s)) + Q_2 (s, \pi(s)))/2
        # \Delta V = V(s') - V(s)
        # w(s, s') = min(exp(\beta_(SCAS) \Delta V), w_(max))
        # s\hat = s + \epsilon, \epsilon \sim N(0, \sigma_s^2 I)
        # Loss_correction = E_((s, a, r, s') \sim D) [w(s, s')\|M(s\hat, \pi(s\hat)) - s'\|_2^2]
        observations, _, _, next_observations, _ = batch
        actions = self.actor(observations)
        next_actions = self.actor(next_observations)
        value = self.critic.q_mean(observations, actions)
        next_value = self.critic.q_mean(next_observations, next_actions)
        weight = (
            self.beta_scas
            * (next_value.detach() - value.detach())
        ).exp().clamp(max=self.w_max)

        noisy_observations = self.perturb_observations(observations)
        noisy_actions = self.actor(noisy_observations)
        predicted_next_observations = self.dynamic(noisy_observations, noisy_actions)
        return (weight * ((predicted_next_observations - next_observations) ** 2)).mean()

    def loss_actor(self, batch: Batch) -> torch.Tensor:
        # Loss_Actor = (1 - \lambda_A) Loss_opt + \lambda_A Loss_correction
        return (
            (1.0 - self.lambda_a) * self.loss_opt(batch)
            + self.lambda_a * self.loss_correction(batch)
        )


class SCASNAgent(NAgent, SCASAgent):
    pass


class SCASGPAgent(GPAgent, SCASAgent):
    pass


class SCASGPNAgent(GPAgent, SCASNAgent):
    pass
