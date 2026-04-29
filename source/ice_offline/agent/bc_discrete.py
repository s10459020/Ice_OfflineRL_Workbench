"""Behavior Cloning discrete agent (minimal fixed structure)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def dist(self, o: torch.Tensor) -> Categorical:
        return Categorical(logits=self.network(o))

    def mode(self, o: torch.Tensor) -> torch.Tensor:
        return self.dist(o).logits.argmax(dim=1).long()

    def sample(self, o: torch.Tensor) -> torch.Tensor:
        return self.dist(o).sample()

    def forward(self, o: torch.Tensor) -> Categorical:
        return self.dist(o)


class BCAgentDiscrete:
    # ====================
    # Init
    # ====================
    def __init__(self, obs_size: int, act_size: int):
        self.device = "cpu"
        self.beta = 0.5
        self.learning_rate = 1e-3

        self.policy = _Pi(obs_size, act_size).to(self.device)

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
    def act(self, observation, epsilon: float = 0.0):
        o = torch.as_tensor(np.asarray(observation, dtype=np.float32)[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.critic.q(o)
            a = q.argmax(dim=1).long()
            if epsilon > 0.0:
                rand_a = torch.randint(0, self.critic._q.action_size, (1,), device=self.device)
                if torch.rand((1,), device=self.device).item() < epsilon:
                    a = rand_a
        return int(a.item())

    def update(self, batch):
        observation = batch["obs"]
        action = batch["act"]

        o = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.long, device=self.device).view(-1)

        loss = self._loss(o, a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ====================
    # bc mathmatics
    # ====================
    def _loss_bc(self, logits: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # theta* = max E[log pi]
        #        = min E[-log pi]
        # loss   = -log pi
        #        = -log_softmax(logits)
        log_softmax = F.log_softmax(logits, dim=1)
        return F.nll_loss(log_softmax, a)

    def _loss_regular(self, logits: torch.Tensor) -> torch.Tensor:
        # loss = beta * ||logits||^2
        return self.beta * (logits**2).mean()

    def _loss(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        logits = self.policy.dist(o).logits
        return self._loss_bc(logits, a) + self._loss_regular(logits)

    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._loss(o, a)
