"""Q discrete agent (basic Q-learning in offline batch setting)."""

import numpy as np
import torch
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

class _TQ(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self._q = _Q(obs_size, act_size)
        self._targ_q = _Q(obs_size, act_size)
        self._targ_q.load_state_dict(self._q.state_dict())

    def update_target(self) -> None:
        self._targ_q.load_state_dict(self._q.state_dict())

    def q(self, o: torch.Tensor) -> torch.Tensor:
        return self._q(o)

    def tq(self, o: torch.Tensor) -> torch.Tensor:
        return self._targ_q(o)

class QAgentDiscrete(TorchAgent):
    # ====================
    # Init
    # ====================
    def __init__(
        self,
        obs_size: int = 0,
        act_size: int = 0,
        learning_rate: float = 6.25e-5,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        device: str = "cpu",
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self._grad_step = 0
        self.q = None
        self.optim = None

        if obs_size > 0 and act_size > 0:
            self.set_dim(obs_size, act_size)

    def set_dim(self, obs_size: int, act_size: int) -> None:
        self.q = _TQ(obs_size, act_size).to(self.device)
        self.optim = _Adam(self.learning_rate)(self.q._q.parameters())

    def configure(self, env_spec) -> None:
        assert env_spec.observation_shape is not None
        assert env_spec.action_cardinality is not None
        assert len(env_spec.action_cardinality) == 1
        obs_size = int(np.prod(env_spec.observation_shape))
        action_size = int(env_spec.action_cardinality[0])
        self.set_dim(obs_size=obs_size, act_size=action_size)

    # ====================
    # Public API
    # ====================
    def act(self, observation, epsilon: float = 0.0):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.q.q(o)
            if np.random.random() < epsilon:
                a = torch.randint(0, self.q._q.action_size, (1,), device=self.device)
            else:
                a = q.argmax(dim=1).long()
        return int(a.cpu().numpy()[0])

    def update(self, batch: TorchBuffer):
        self._grad_step += 1
        o = batch.obs_list
        a = batch.act_list.long().view(-1)
        r = batch.rew_list.view(-1, 1)
        on = batch.next_obs_list
        d = batch.done_list.view(-1, 1)

        loss = self.loss_critic(o, a, r, on, d)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        if self._grad_step % self.target_update_interval == 0:
            self.q.update_target()

    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "q": self.q.state_dict(),
            "optimizer": self.optim.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.q.load_state_dict(state["q"])
        self.optim.load_state_dict(state["optimizer"])

    # ====================
    # q-learning mathmatics
    # ====================
    def _target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            qn = self.q.tq(on).max(dim=1, keepdim=True).values
            return r + self.gamma * qn * (1.0 - d)

    def _loss_q(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        q_sa = self.q.q(o).gather(1, a.view(-1, 1))
        target = self._target(on, r, d)
        delta = target - q_sa
        cond = delta.detach().abs() < 1.0
        huber = torch.where(cond, 0.5 * delta**2, delta.abs() - 0.5)
        return huber.mean()

    def loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self._loss_q(o, a, r, on, d)
