"""Implicit Q-Learning discrete agent (Q-learning symmetric version)."""

import numpy as np
import torch
import torch.nn.functional as F
from .._spec import TorchAgent
from ice_offline.dataset._spec import TorchBuffer

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

class IQLDiscreteAgent(TorchAgent):
    # ====================
    # Init
    # ====================
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        learning_rate: float = 6.25e-5,
        gamma: float = 0.99,
        v_tau: float = 0.7,
        device: str = "cpu",
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.v_tau = v_tau
        self.q = None
        self.v = None
        self.optim = None
        self.q = _Q(obs_size, act_size).to(self.device)
        self.v = _V(obs_size, tau=self.v_tau).to(self.device)
        self.optim = _Adam(self.learning_rate)(
            list(self.q.parameters()) + list(self.v.parameters())
        )

    # ====================
    # Public API
    # ====================
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

    def update(self, batch: TorchBuffer):
        o = batch.obs_list
        a = batch.act_list.long().view(-1)
        r = batch.rew_list.view(-1, 1)
        on = batch.next_obs_list
        d = batch.done_list.view(-1, 1)

        loss = self.loss_critic(o, a, r, on, d)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "q": self.q.state_dict(),
            "v": self.v.state_dict(),
            "optimizer": self.optim.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.q.load_state_dict(state["q"])
        self.v.load_state_dict(state["v"])
        optim_key = "optimizer" if "optimizer" in state else "optim"
        self.optim.load_state_dict(state[optim_key])

    # ====================
    # critic mathmatics
    # ====================
    def _target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # target = r + gamma * V(s') * (1-done)
        with torch.no_grad():
            return r + self.gamma * self.v(on) * (1.0 - d)

    def _loss_q(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # loss_q = Huber(Q(s,a) - target)
        q_sa = self.q(o).gather(1, a.view(-1, 1))
        target = self._target(on, r, d)
        delta = target - q_sa
        cond = delta.detach().abs() < 1.0
        huber = torch.where(cond, 0.5 * delta**2, delta.abs() - 0.5)
        return huber.mean()

    def _loss_v(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # loss_v = E[ |tau - 1(Q-V<0)| * (Q-V)^2 ]
        q_sa = self.q(o).gather(1, a.view(-1, 1)).detach()
        v_t = self.v(o)
        diff = q_sa - v_t
        weight = (self.v.tau - (diff < 0.0).float()).abs().detach()
        return (weight * diff.pow(2)).mean()

    def loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self._loss_q(o, a, r, on, d) + self._loss_v(o, a)

