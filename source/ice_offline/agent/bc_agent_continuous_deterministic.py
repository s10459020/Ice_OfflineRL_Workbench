"""Behavior Cloning continuous agent (deterministic)."""

import torch
import torch.nn.functional as F


class BCPolicyContinuousDeterministic(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, obs_t: torch.Tensor) -> torch.Tensor:
        mean = self.network(obs_t)
        return torch.tanh(mean)


class BCAgentContinuousDeterministic:
    # ====================
    # Init
    # ====================
    def __init__(self, obs_size: int, act_size: int):
        self.device = "cpu"
        self.learning_rate = 1e-3

        self.policy = BCPolicyContinuousDeterministic(obs_size, act_size).to(
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
            action = self.policy(obs_t)
        return action.cpu().numpy()

    def action_best(self, obs):
        return self.action_best_batch([obs])[0]

    def update(self, batch):
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)

        pred_act_t = self.policy(obs_t)
        loss = self._loss(pred_act_t, act_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ====================
    # bc mathmatics
    # ====================
    def _loss(self, pred_action_t: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred_action_t, act_t)
