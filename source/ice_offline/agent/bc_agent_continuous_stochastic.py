"""Behavior Cloning continuous agent (stochastic)."""

import torch
import torch.nn.functional as F
from torch.distributions import Normal


class BCPolicyContinuousStochastic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd_head = torch.nn.Linear(256, act_size)
        self.min_logstd = -4.0
        self.max_logstd = 15.0

    def forward(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(obs_t)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(self.min_logstd, self.max_logstd)
        squashed_mean = torch.tanh(mean)
        return squashed_mean, logstd


class BCAgentContinuousStochastic:
    # ====================
    # Init
    # ====================
    def __init__(self, obs_size: int, act_size: int):
        self.device = "cpu"
        self.learning_rate = 1e-3

        self.policy = BCPolicyContinuousStochastic(obs_size, act_size).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Public API
    # ====================
    def action_best_batch(self, obs_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            squashed_mean, _ = self.policy(obs_t)
        return squashed_mean.cpu().numpy()

    def action_best(self, obs):
        return self.action_best_batch([obs])[0]

    def action_sample_batch(self, obs_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            squashed_mean, logstd = self.policy(obs_t)
            sampled = (
                Normal(squashed_mean, logstd.exp()).rsample().clamp(-1.0, 1.0)
            )
        return sampled.cpu().numpy()

    def action_sample(self, obs):
        return self.action_sample_batch([obs])[0]

    def update(self, batch):
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)

        squashed_mean, logstd = self.policy(obs_t)
        loss = self._loss(squashed_mean, logstd, act_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ====================
    # bc mathmatics
    # ====================
    def _loss(
        self, squashed_mean: torch.Tensor, logstd: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        pred_action_t = (
            Normal(squashed_mean, logstd.exp()).rsample().clamp(-1.0, 1.0)
        )
        return F.mse_loss(pred_action_t, act_t)
