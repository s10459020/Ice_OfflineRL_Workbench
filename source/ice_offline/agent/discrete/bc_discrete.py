"""Behavior Cloning discrete agent (minimal fixed structure)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ice_offline.agent._spec import TorchAgent
from ice_offline.dataset._spec import TorchBuffer


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.action_size = act_size
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_size),
        )

    def forward(self, o: torch.Tensor) -> Categorical:
        logits = self.network(o)
        return Categorical(logits=logits)


class BCDiscreteAgent(TorchAgent):
    def __init__(self, obs_size: int, act_size: int, beta: float = 0.5, device: str = "cuda"):
        self.device = device
        self.learning_rate = 1e-3
        self.beta = beta
        self.policy = _Pi(obs_size, act_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    def act(self, observation, epsilon: float = 0.0):
        o = torch.as_tensor(np.asarray(observation, dtype=np.float32)[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.policy(o).logits
            a = torch.argmax(logits, dim=1).long()
            if epsilon > 0.0 and torch.rand((1,), device=self.device).item() < epsilon:
                a = torch.randint(0, self.policy.action_size, (1,), device=self.device)
        return int(a.item())

    def act_batch(self, observation_batch, epsilon: float = 0.0):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.policy(o).logits
            a = torch.argmax(logits, dim=1).long()
            if epsilon > 0.0:
                batch_size = int(a.shape[0])
                rand_a = torch.randint(0, self.policy.action_size, (batch_size,), device=self.device)
                replace_mask = torch.rand((batch_size,), device=self.device) < epsilon
                a = torch.where(replace_mask, rand_a, a)
        return a.cpu().numpy()

    def update(self, batch: TorchBuffer):
        o = batch.obs_list
        a = batch.act_list.long().view(-1)
        loss = self.loss_actor(o, a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "pi": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.policy.load_state_dict(state["pi"])
        self.optimizer.load_state_dict(state["optimizer"])

    def loss_bc(self, logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, action)

    def loss_regular(self, logits: torch.Tensor) -> torch.Tensor:
        penalty = (logits ** 2).mean()
        return self.beta * penalty

    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        logits = self.policy(o).logits
        loss_bc = self.loss_bc(logits, a)
        loss_regular = self.loss_regular(logits)
        return loss_bc + loss_regular

