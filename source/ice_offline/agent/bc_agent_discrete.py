"""Behavior Cloning discrete agent (minimal fixed structure)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class BCPolicy(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, obs_t: torch.Tensor) -> Categorical:
        logits = self.network(obs_t)
        return Categorical(logits=logits)


class BCAgentDiscrete:
    # ====================
    # Init
    # ====================
    def __init__(self, obs_size: int, act_size: int):
        self.device = "cpu"
        self.beta = 0.5
        self.learning_rate = 1e-3

        self.policy = BCPolicy(obs_size, act_size).to(self.device)

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
            dist = self.policy(obs_t)
            action = torch.argmax(dist.logits, dim=1).long()
        return action.cpu().numpy()

    def action_best(self, obs):
        return int(self.action_best_batch([obs])[0])
    

    def update(self, batch):
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(batch["act"], dtype=torch.long, device=self.device).view(-1)

        dist = self.policy(obs_t)
        loss = self._loss(dist.logits, act_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ====================
    # bc mathmatics
    # ====================
    def _loss_bc(self, logits: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        # theta* = max E[log pi]
        #        = min E[-log pi]
        # loss   = -log pi
        #        = -log_softmax(logits)
        log_softmax = F.log_softmax(logits, dim=1)
        return F.nll_loss(log_softmax, act_t)

    def _loss_regular(self, logits: torch.Tensor) -> torch.Tensor:
        # loss = beta * ||logits||^2
        return self.beta * (logits**2).mean()

    def _loss(self, logits: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        return self._loss_bc(logits, act_t) + self._loss_regular(logits)
