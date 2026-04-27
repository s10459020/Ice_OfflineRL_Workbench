"""Implicit Q-Learning continuous agent (minimal fixed structure)."""

import numpy as np
import torch
from d3rlpy.models.torch.distributions import GaussianDistribution


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
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
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
        mu = self.mean_head(h)
        squashed_mu = torch.tanh(mu)
        base = self.max_logstd - self.min_logstd
        clipped_logstd = self.min_logstd + torch.sigmoid(self.logstd) * base
        return mu, squashed_mu, clipped_logstd


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
    def __init__(self, obs_size: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.network(o)


class IQLAgentContinuous:
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        rho: float = 0.005,
        tau: float = 0.7,
        weight_temp: float = 3.0,
        max_weight: float = 100.0,
    ):
        self.device = "cpu"
        self.act_size = act_size
        self.gamma = gamma
        self.rho = rho
        self.tau = tau
        self.weight_temp = weight_temp
        self.max_weight = max_weight

        self.policy = _Pi(obs_size, act_size).to(self.device)
        self.q1 = _Q(obs_size, act_size).to(self.device)
        self.q2 = _Q(obs_size, act_size).to(self.device)
        self.targ_q1 = _Q(obs_size, act_size).to(self.device)
        self.targ_q2 = _Q(obs_size, act_size).to(self.device)
        self.v = _V(obs_size).to(self.device)
        self.targ_q1.load_state_dict(self.q1.state_dict())
        self.targ_q2.load_state_dict(self.q2.state_dict())

        self.actor_optim = _Adam(actor_learning_rate)(self.policy.parameters())
        self.critic_optim = _Adam(critic_learning_rate)(
            list(self.q1.parameters()) + list(self.q2.parameters()) + list(self.v.parameters())
        )

    # ====================
    # act
    # ====================
    def action_best_batch(self, obs_batch):
        o = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, squashed_mu, _ = self.policy(o)
        return squashed_mu.cpu().numpy()

    def action_best(self, obs):
        obs_np = np.asarray(obs, dtype=np.float32)[None, :]
        return self.action_best_batch(obs_np)[0]

    def action_sample_batch(self, obs_batch):
        o = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self._sample_a(o)
        return action.cpu().numpy()

    def action_sample(self, obs):
        obs_np = np.asarray(obs, dtype=np.float32)[None, :]
        return self.action_sample_batch(obs_np)[0]

    # ====================
    # helper
    # ====================
    def _build_dist(self, o: torch.Tensor) -> GaussianDistribution:
        mu, squashed_mu, logstd = self.policy(o)
        return GaussianDistribution(loc=squashed_mu, std=logstd.exp(), raw_loc=mu)

    def _sample_a(self, o: torch.Tensor) -> torch.Tensor:
        return self._build_dist(o).sample()

    def _log_pi(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self._build_dist(o).log_prob(a)

    def _QQ(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.q1(o, a), self.q2(o, a)], dim=0)

    def _targ_QQ(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.targ_q1(o, a), self.targ_q2(o, a)], dim=0)

    # ====================
    # update
    # ====================
    def update(self, batch):
        o = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).view(-1, 1)
        on = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).view(-1, 1)

        critic_loss = self._loss_critic(o, a, r, on, d)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = self._loss_actor(o, a)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._update_target_soft()

    def _update_target_soft(self) -> None:
        with torch.no_grad():
            for p_targ, p in zip(self.targ_q1.parameters(), self.q1.parameters()):
                p_targ.data.mul_(1.0 - self.rho).add_(self.rho * p.data)
            for p_targ, p in zip(self.targ_q2.parameters(), self.q2.parameters()):
                p_targ.data.mul_(1.0 - self.rho).add_(self.rho * p.data)

    # ====================
    # critic
    # ====================
    def _target_q(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return r + self.gamma * self.v(on) * (1.0 - d)

    def _l2_tau(self, u: torch.Tensor) -> torch.Tensor:
        # L2_tau = |tau - 1(u<0)| * u^2
        weight = (self.tau - (u < 0.0).float()).abs().detach()
        return weight * u.pow(2)

    def _loss_q(
        self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        # loss_q = E_batch{(target - Q)^2}
        target = self._target_q(on, r, d)
        qq = self._QQ(o, a)
        return (qq[0] - target).pow(2).mean() + (qq[1] - target).pow(2).mean()

    def _loss_v(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # loss_v = E_batch{L2(Q-V)}
        q_t = self._targ_QQ(o, a).min(dim=0).values
        v_t = self.v(o)
        diff = q_t.detach() - v_t
        return self._l2_tau(diff).mean()

    def _loss_critic(
        self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        return self._loss_q(o, a, r, on, d) + self._loss_v(o, a)

    # ====================
    # actor
    # ====================
    def _weight(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_t = self._targ_QQ(o, a).min(dim=0).values
            v_t = self.v(o)
            adv = q_t - v_t
            return (self.weight_temp * adv).exp().clamp(max=self.max_weight)

    def _loss_actor(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # loss_pi = E_batch{-weight * log_pi}
        log_probs = self._log_pi(o, a)
        weight = self._weight(o, a)
        return -(weight * log_probs).mean()
