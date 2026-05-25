"""Implicit Q-Learning continuous agent (minimal fixed structure)."""

import numpy as np
import torch
from torch.distributions import Normal
from ice_offline.agent._spec import EnvSpec
from ice_offline.agent._spec import TorchAgent
from ice_offline.runner.evaluator import TransitionBatch


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


class _Pi(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, beta: float, max_weight: float):
        super().__init__()
        self.beta = beta
        self.max_weight = max_weight
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd = torch.nn.Parameter(torch.zeros(1, act_size, dtype=torch.float32))
        self.min_logstd = -5.0
        self.max_logstd = 2.0

    def forward(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.hidden(o)
        mean = self.mean_head(h)
        squashed_mean = torch.tanh(mean)
        base = self.max_logstd - self.min_logstd
        # use_std_parameter
        clipped_logstd = self.min_logstd + torch.sigmoid(self.logstd) * base
        return mean, squashed_mean, clipped_logstd

    def dist(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, squashed_mean, logstd = self(o)
        return squashed_mean, logstd.exp()

    def mode(self, o: torch.Tensor) -> torch.Tensor:
        squashed_mean, _ = self.dist(o)
        return squashed_mean

    def sample(self, o: torch.Tensor) -> torch.Tensor:
        squashed_mean, std = self.dist(o)
        return Normal(squashed_mean, std).rsample().clamp(-1.0, 1.0)

    def log_prob(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        squashed_mean, std = self.dist(o)
        return Normal(squashed_mean, std).log_prob(a).sum(dim=-1, keepdims=True)


class _Q(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size + act_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([o, a], dim=1))


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


class _QQ(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int, tau: float):
        super().__init__()
        self.tau = tau
        self.q1 = _Q(obs_size, act_size)
        self.q2 = _Q(obs_size, act_size)
        self.targ_q1 = _Q(obs_size, act_size)
        self.targ_q2 = _Q(obs_size, act_size)
        self.targ_q1.load_state_dict(self.q1.state_dict())
        self.targ_q2.load_state_dict(self.q2.state_dict())

    def qq(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.q1(o, a), self.q2(o, a)], dim=0)

    def targ_qq(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.targ_q1(o, a), self.targ_q2(o, a)], dim=0)

    def qq_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.qq(o, a).min(dim=0).values

    def tqq_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.targ_qq(o, a).min(dim=0).values

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p_targ, p in zip(self.targ_q1.parameters(), self.q1.parameters()):
                p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p_targ, p in zip(self.targ_q2.parameters(), self.q2.parameters()):
                p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)


class IQLAgentContinuous(TorchAgent):
    def __init__(
        self,
        obs_size: int = 0,
        act_size: int = 0,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        q_tau: float = 0.005,
        v_tau: float = 0.7,
        beta: float = 3.0,
        max_weight: float = 100.0,
    ):
        self.device = "cpu"
        self.obs_size = obs_size
        self.act_size = act_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.q_tau = q_tau
        self.v_tau = v_tau
        self.beta = beta
        self.max_weight = max_weight
        self.actor = None
        self.critic = None
        self.v = None
        self.actor_optim = None
        self.critic_optim = None

        if obs_size > 0 and act_size > 0:
            self.set_dim(obs_size, act_size)

    def set_dim(self, obs_size: int, act_size: int) -> None:
        self.obs_size = obs_size
        self.act_size = act_size
        self.actor = _Pi(obs_size, act_size, beta=self.beta, max_weight=self.max_weight).to(self.device)
        self.critic = _QQ(obs_size, act_size, tau=self.q_tau).to(self.device)
        self.v = _V(obs_size, tau=self.v_tau).to(self.device)
        self.actor_optim = _Adam(self.actor_learning_rate)(
            self.actor.parameters()
        )
        self.critic_optim = _Adam(self.critic_learning_rate)(
              list(self.critic.q1.parameters())
            + list(self.critic.q2.parameters())
            + list(self.v.parameters())
        )

    def configure(self, env_spec: EnvSpec) -> None:
        assert env_spec.observation_shape is not None
        assert env_spec.action_shape is not None
        obs_size = int(np.prod(env_spec.observation_shape))
        act_size = int(np.prod(env_spec.action_shape))
        self.set_dim(obs_size=obs_size, act_size=act_size)

    # ====================
    # public API
    # ====================
    def act(self, observation, greedy: bool = True):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor.mode(o) if greedy else self.actor.sample(o)
        return action.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor.mode(o) if greedy else self.actor.sample(o)
        return action.cpu().numpy()

    def update(self, batch):
        observation = batch["obs"]
        action = batch["act"]
        reward = batch["rew"]
        next_observation = batch["next_obs"]
        done = batch["done"]

        o = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        r = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1, 1)
        on = torch.as_tensor(next_observation, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)

        critic_loss = self._loss_critic(o, a, r, on, d)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = self._loss_actor(o, a)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic.update_target_soft()

    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "actor": self.actor.state_dict(),
            "q": self.critic.state_dict(),
            "v": self.v.state_dict(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["q"])
        self.v.load_state_dict(state["v"])
        self.actor_optim.load_state_dict(state["actor_optimizer"])
        self.critic_optim.load_state_dict(state["critic_optimizer"])

    # ====================
    # critic
    # ====================
    def _target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return r + self.gamma * self.v(on) * (1.0 - d)

    def _loss_q(
        self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        target = self._target(on, r, d)
        qq = self.critic.qq(o, a)
        # loss_q = E_batch{(target - Q)^2}
        return (qq[0] - target).pow(2).mean() + (qq[1] - target).pow(2).mean()

    def _loss_v(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # L2_tau = |tau - 1(u<0)| * u^2
        # loss_v = E_batch{ L2_tau(Q-V) }
        q_t = self.critic.tqq_min(o, a)
        v_t = self.v(o)
        diff = q_t.detach() - v_t
        weight = (self.v.tau - (diff < 0.0).float()).abs().detach()
        return (weight * diff.pow(2)).mean()

    def _loss_critic(
        self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        return self._loss_q(o, a, r, on, d) + self._loss_v(o, a)

    # ====================
    # actor
    # ====================
    def _weight(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # weight = -exp( beta * (Q - V))
        with torch.no_grad():
            q_t = self.critic.tqq_min(o, a)
            v_t = self.v(o)
            adv = q_t - v_t
            return -(self.actor.beta * adv).exp().clamp(max=self.actor.max_weight)

    def _loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # loss_pi = E_batch{weight * log_pi}
        weight = self._weight(o, a)
        log_pi = self.actor.log_prob(o, a)
        return (weight * log_pi).mean()

    def loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self._loss_critic(o, a, r, on, d)

    def loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._loss_actor(o, a)


def eval_iql_continuous_loss_q(agent: "IQLAgentContinuous", transitions: TransitionBatch) -> dict[str, float]:
    o, a, r, on, d = transitions
    with torch.no_grad():
        return {"loss_q": float(agent._loss_q(o, a, r, on, d).item())}


def eval_iql_continuous_loss_v(agent: "IQLAgentContinuous", transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    with torch.no_grad():
        return {"loss_v": float(agent._loss_v(o, a).item())}


def eval_iql_continuous_loss_pi(agent: "IQLAgentContinuous", transitions: TransitionBatch) -> dict[str, float]:
    o, a, _, _, _ = transitions
    with torch.no_grad():
        return {"loss_pi": float(agent.loss_actor(o, a).item())}


