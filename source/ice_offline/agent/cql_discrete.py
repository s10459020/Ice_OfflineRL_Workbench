"""Conservative Q-Learning discrete agent (minimal fixed structure)."""

import numpy as np
import torch
import torch.nn.functional as F
from ._spec import EnvSpec
from ._spec import TorchAgent
from ice_offline.runner.offline import TransitionBatch


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


class CQLAgentDiscrete(TorchAgent):
    # ====================
    # Init
    # ====================
    def __init__(self, obs_size: int = 0, act_size: int = 0, learning_rate: float = 6.25e-5, gamma: float = 0.99, alpha: float = 1.0, target_update_interval: int = 8000):
        self.device = "cpu"
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self._grad_step = 0
        self.critic = None
        self.optim = None

        if obs_size > 0 and act_size > 0:
            self.set_dim(obs_size, act_size)

    def set_dim(self, obs_size: int, act_size: int) -> None:
        self.critic = _TQ(obs_size=obs_size, act_size=act_size).to(self.device)
        self.optim = _Adam(self.learning_rate)(self.critic._q.parameters())

    def configure(self, env_spec: EnvSpec) -> None:
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
        o = torch.as_tensor(np.asarray(observation, dtype=np.float32)[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.critic.q(o)
            a = q.argmax(dim=1).long()
            if epsilon > 0.0:
                rand_a = torch.randint(0, self.critic._q.action_size, (1,), device=self.device)
                if torch.rand((1,), device=self.device).item() < epsilon:
                    a = rand_a
        return int(a.item())

    def act_batch(self, observation_batch, epsilon: float = 0.0):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.critic.q(o)
            a = q.argmax(dim=1).long()
            if epsilon > 0.0:
                batch_size = int(a.shape[0])
                rand_a = torch.randint(0, self.critic._q.action_size, (batch_size,), device=self.device)
                replace_mask = torch.rand((batch_size,), device=self.device) < epsilon
                a = torch.where(replace_mask, rand_a, a)
        return a.cpu().numpy()

    def update(self, batch):
        self._grad_step += 1
        observation = batch["obs"]
        action = batch["act"]
        reward = batch["rew"]
        next_observation = batch["next_obs"]
        done = batch["done"]

        o = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.long, device=self.device).view(-1)
        r = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        on = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)

        loss = self._loss(o, a, r, on, d)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self._grad_step % self.target_update_interval == 0:
            self.critic.update_target()

    def _save(self) -> dict[str, torch.Tensor]:
        return {
            "q": self.critic.state_dict(),
            "optimizer": self.optim.state_dict(),
        }

    def _load(self, state: dict[str, torch.Tensor]) -> None:
        q_key = "q" if "q" in state else "critic"
        optim_key = "optimizer" if "optimizer" in state else "optim"
        self.critic.load_state_dict(state[q_key])
        self.optim.load_state_dict(state[optim_key])

    # ====================
    # cql mathmatics
    # ====================
    def _target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # target = r + gamma * max Q_target(s', a'_policy) * (1-done)
        with torch.no_grad():
            an = self.critic.q(on).argmax(dim=1)
            qn = self.critic.tq(on).gather(1, an.view(-1, 1))
            return r + self.gamma * qn * (1.0 - d)

    def _loss_td(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # L_TD = E[ (Q(s,a) - target)^2 ]
        # Huber loss = delta < a ? : L2(Q-target) : L1(loss)
        q = self.critic.q(o).gather(1, a.view(-1, 1)) # Q(si,ai) for batch
        target = self._target(on, r, d)
        delta = target - q

        cond = delta.detach().abs() < 1.0
        huber = torch.where(cond, 0.5 * delta**2, delta.abs() - 0.5)
        return huber.mean()

    def _loss_cql(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # L_CQL(H) = E_s~D[logsumexp( Q(.|a) )] - E_(s,a)~D[ Q(s,a) ]
        # E_D[s]: input o
        # E_D[s,a]: input o,a
        q_all = self.critic.q(o)
        q_a = torch.logsumexp(q_all, dim=1, keepdim=True)

        one_hot = F.one_hot(a.view(-1), num_classes=self.critic._q.action_size).float()
        q_sa = (q_all * one_hot).sum(dim=1, keepdim=True)

        return (q_a - q_sa).mean()

    def _loss(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # L = L_TD + alpha * L_CQL(H)
        loss_td = self._loss_td(o, a, r, on, d)
        loss_cql = self._loss_cql(o, a)
        return loss_td + self.alpha * loss_cql

    def loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self._loss(o, a, r, on, d)


def eval_cql_discrete_loss(agent: "CQLAgentDiscrete", transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss": float(agent.loss_critic(o, a, r, on, d).item())}


def eval_cql_discrete_loss_td(agent: "CQLAgentDiscrete", transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    return {"loss_td": float(agent._loss_td(o, a, r, on, d).item())}


def eval_cql_discrete_loss_cql(agent: "CQLAgentDiscrete", transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss_cql": float(agent._loss_cql(o, a).item())}
