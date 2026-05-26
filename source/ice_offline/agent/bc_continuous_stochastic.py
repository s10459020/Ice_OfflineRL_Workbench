"""Behavior Cloning continuous agent (stochastic)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from ice_offline.agent._spec import EnvSpec
from ice_offline.agent._spec import TorchAgent
from ice_offline.run.evaluator import TransitionBatch


class _Pi(torch.nn.Module):
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

    def dist(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.network(o)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(self.min_logstd, self.max_logstd)
        a_mean = torch.tanh(mean)
        return a_mean, logstd

    def mode(self, o: torch.Tensor) -> torch.Tensor:
        a_mean, _ = self.dist(o)
        return a_mean

    def sample(self, o: torch.Tensor) -> torch.Tensor:
        a_mean, logstd = self.dist(o)
        return Normal(a_mean, logstd.exp()).rsample().clamp(-1.0, 1.0)

    def forward(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_mean, logstd = self.dist(o)
        return a_mean, logstd


class BCAgentContinuousStochastic(TorchAgent):
    def __init__(self, obs_size: int = 0, act_size: int = 0):
        self.device = "cpu"
        self.learning_rate = 1e-3
        self.policy = None
        self.optimizer = None
        if obs_size > 0 and act_size > 0:
            self.set_dim(obs_size, act_size)

    def set_dim(self, obs_size: int, act_size: int) -> None:
        self.policy = _Pi(obs_size, act_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    def configure(self, env_spec: EnvSpec) -> None:
        assert env_spec.observation_shape is not None
        assert env_spec.action_shape is not None
        obs_size = int(np.prod(env_spec.observation_shape))
        act_size = int(np.prod(env_spec.action_shape))
        self.set_dim(obs_size=obs_size, act_size=act_size)

    def act(self, observation, greedy: bool = True):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.policy.mode(o) if greedy else self.policy.sample(o)
        return a.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.policy.mode(o) if greedy else self.policy.sample(o)
        return a.cpu().numpy()

    def update(self, batch):
        observation = batch["obs"]
        action = batch["act"]

        o = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        loss = self._loss(o, a)
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

    def _loss(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_pred = self.policy.sample(o)
        return F.mse_loss(a_pred, a)

    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._loss(o, a)


def eval_bc_stochastic_loss_pi(
    agent: "BCAgentContinuousStochastic",
    transitions: TransitionBatch,
) -> dict[str, float]:
    o, a, _, _, _ = transitions
    return {"loss_pi": float(agent.loss_actor(o, a).item())}

