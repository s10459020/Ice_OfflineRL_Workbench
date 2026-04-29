"""Implicit Q-Learning discrete agent (Q-learning symmetric version)."""

import numpy as np
import torch
import torch.nn.functional as F


class _Adam:
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, params):
        return torch.optim.Adam(
            params,
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )


class _Q(torch.nn.Module):
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

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.network(o)


class _V(torch.nn.Module):
    def __init__(self, obs_size: int, tau: float):
        super().__init__()
        self.tau = tau
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.network(o)


class IQLAgentDiscrete:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        learning_rate: float = 6.25e-5,
        gamma: float = 0.99,
        v_tau: float = 0.7,
    ):
        self.device = "cpu"
        self.gamma = gamma

        self.q = _Q(obs_size, act_size).to(self.device)
        self.v = _V(obs_size, tau=v_tau).to(self.device)
        self.optim = _Adam(learning_rate)(
            list(self.q.parameters()) + list(self.v.parameters())
        )

    def act(self, observation, epsilon: float = 0.0):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.q(o)
            if np.random.random() < epsilon:
                a = torch.randint(0, self.q.action_size, (1,), device=self.device)
            else:
                a = q.argmax(dim=1).long()
        return int(a.cpu().numpy()[0])

    def act_batch(self, observation_batch, epsilon: float = 0.0):
        o = torch.as_tensor(
            np.asarray(observation_batch),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            q = self.q(o)
            a = q.argmax(dim=1).long()
            if epsilon > 0.0:
                rand_a = torch.randint(0, self.q.action_size, (o.shape[0],), device=self.device)
                mask = torch.rand((o.shape[0],), device=self.device) < epsilon
                a = torch.where(mask, rand_a, a)
        return a.cpu().numpy()

    def update(self, batch):
        o = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["act"], dtype=torch.long, device=self.device).view(-1)
        r = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).view(-1, 1)
        on = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).view(-1, 1)

        loss = self._loss(o, a, r, on, d)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def _target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return r + self.gamma * self.v(on) * (1.0 - d)

    def _loss_q(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        q_sa = self.q(o).gather(1, a.view(-1, 1))
        target = self._target(on, r, d)
        delta = target - q_sa
        cond = delta.detach().abs() < 1.0
        huber = torch.where(cond, 0.5 * delta**2, delta.abs() - 0.5)
        return huber.mean()

    def _loss_v(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q_sa = self.q(o).gather(1, a.view(-1, 1)).detach()
        v_t = self.v(o)
        diff = q_sa - v_t
        weight = (self.v.tau - (diff < 0.0).float()).abs().detach()
        return (weight * diff.pow(2)).mean()

    def _loss(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self._loss_q(o, a, r, on, d) + self._loss_v(o, a)

    def loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self._loss(o, a, r, on, d)

        return torch.zeros((), dtype=torch.float32, device=self.device)


