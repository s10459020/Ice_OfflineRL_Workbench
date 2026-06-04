from dataclasses import dataclass

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from ice_offline.agent._spec import TorchAgent
from ice_offline.dataset._spec import TorchBuffer

class _SquashedGaussianDistribution:
    def __init__(self, loc: torch.Tensor, std: torch.Tensor):
        self._mean = loc
        self._std = std
        self._dist = Normal(self._mean, self._std)

    def sample_with_log_prob(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw_y = self._dist.rsample()
        log_prob = self._log_prob_from_raw_y(raw_y)
        return torch.tanh(raw_y), log_prob

    def sample_n_with_log_prob(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw_y = self._dist.rsample((n,))
        log_prob = self._log_prob_from_raw_y(raw_y)
        return torch.tanh(raw_y).transpose(0, 1), log_prob.transpose(0, 1)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self._mean)

    def _log_prob_from_raw_y(self, raw_y: torch.Tensor) -> torch.Tensor:
        jacob = 2 * (math.log(2) - raw_y - F.softplus(-2 * raw_y))
        return (self._dist.log_prob(raw_y) - jacob).sum(dim=-1, keepdims=True)

class _Pi(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        min_logstd: float = -20.0,
        max_logstd: float = 2.0,
        initial_alpha: float = 1.0,
    ):
        super().__init__()
        self.act_size = act_size
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd_head = torch.nn.Linear(256, act_size)
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float32))
        self.log_alpha.data.fill_(math.log(initial_alpha))

    def temp(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def dist(self, o: torch.Tensor) -> _SquashedGaussianDistribution:
        h = self.hidden(o)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(self.min_logstd, self.max_logstd)
        return _SquashedGaussianDistribution(loc=mean, std=logstd.exp())

class _Actor(torch.nn.Module):
    def __init__(self, obs_size: int, act_size: int):
        super().__init__()
        self.pi = _Pi(obs_size, act_size)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        return self.pi.dist(o).mode()

    def sample(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 對 E_pi[a]{ ... } 的樣本近似，取得a與log pi(a)用於後續計算
        # target_Q: r + gamma* E_pi[a]{ targ_Q(s',a')_min - alpha* log_pi(a'|s') }
        dist = self.pi.dist(o)
        a, log_prob = dist.sample_with_log_prob()
        return a, log_prob

    def sample_n(self, o: torch.Tensor, n_action_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        # E_pi[a]{ ... } 樣本近似的N個抽樣版本，回傳(B*N, A)格式的資料
        dist = self.pi.dist(o)
        a, log_prob = dist.sample_n_with_log_prob(n_action_samples)

        a = a.reshape(-1, a.shape[-1])                  # (B, N, A) => (B*N, A)
        log_prob = log_prob.reshape(-1, 1)              # (B, N, 1) => (B*N, 1)
        return a, log_prob

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
        x = torch.cat([o, a], dim=1)
        return self.network(x)

class _Critic(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        device: str,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_threshold: float = 10.0,
        conservative_weight: float = 5.0,
        n_action_samples: int = 10,
        initial_alpha: float = 1.0,
    ):
        super().__init__()
        self.act_size = act_size
        self.gamma = gamma
        self.tau = tau
        self.alpha_threshold = alpha_threshold
        self.conservative_weight = conservative_weight
        self.n_action_samples = n_action_samples
        self.device = device

        # CQL twin Q networks
        self.q1 = _Q(obs_size, act_size)
        self.q2 = _Q(obs_size, act_size)

        # CQL target Q networks
        self.targ_q1 = _Q(obs_size, act_size)
        self.targ_q2 = _Q(obs_size, act_size)
        self.sync_target_hard()

        # CQL conservative penalty multiplier
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float32))
        self.log_alpha.data.fill_(math.log(initial_alpha))

    def q_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        return torch.cat([q1, q2], dim=1).min(dim=1, keepdim=True).values

    def target_q_min(self, on: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.targ_q1(on, a)
        q2 = self.targ_q2(on, a)
        return torch.cat([q1, q2], dim=1).min(dim=1, keepdim=True).values

    def sample_n(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = o.shape[0]
        zero_tensor = torch.zeros(
            (batch * self.n_action_samples, self.act_size), device=self.device
        ) # (B*N, A)

        a_rand = zero_tensor.uniform_(-1.0, 1.0)
        random_log_prob = torch.full(
            (batch * self.n_action_samples, 1),
            math.log(0.5**self.act_size),
            device=self.device,
        ) # (B*N, A)
        return a_rand, random_log_prob

    def eval_q_n(self, o: torch.Tensor, a_sample: torch.Tensor) -> torch.Tensor:
        o = o.repeat_interleave(self.n_action_samples, dim=0)  # (B, O) => (B*N, O)
        a = a_sample.reshape(-1, a_sample.shape[-1])           # (B*N, A)
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        return torch.stack([q1, q2], dim=0)

    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(0.0, 1e6)

    # ====================
    # CQL target sync
    # ====================
    def sync_target_hard(self) -> None:
        self.targ_q1.load_state_dict(self.q1.state_dict())
        self.targ_q2.load_state_dict(self.q2.state_dict())

    def update_target_soft(self) -> None:
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.targ_q1.parameters()):
                p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)
            for p, p_targ in zip(self.q2.parameters(), self.targ_q2.parameters()):
                p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)

@dataclass
class CQLSoftQAgent(TorchAgent):
    obs_size: int
    act_size: int
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    actor_alpha_learning_rate: float = 1e-4
    critic_alpha_learning_rate: float = 1e-4
    device: str = "cpu"

    def __post_init__(self):
        self.actor = _Actor(obs_size=self.obs_size, act_size=self.act_size).to(self.device)
        self.critic = _Critic(
            obs_size=self.obs_size,
            act_size=self.act_size,
            device=self.device,
        ).to(self.device)
        self.actor_optim = torch.optim.Adam(
            list(self.actor.pi.hidden.parameters())
            + list(self.actor.pi.mean_head.parameters())
            + list(self.actor.pi.logstd_head.parameters()),
            lr=self.actor_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.critic_optim = torch.optim.Adam(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
            lr=self.critic_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.alpha_optim = torch.optim.Adam(
            [self.actor.pi.log_alpha],
            lr=self.actor_alpha_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.alpha_prime_optim = torch.optim.Adam(
            [self.critic.log_alpha],
            lr=self.critic_alpha_learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
        )

    # ====================
    # Act
    # ====================
    def act(self, observation, greedy: bool = True):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if greedy:
                action = self.actor(o)
            else:
                action, _ = self.actor.sample(o)
        return action.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if greedy:
                a = self.actor(o)
            else:
                a, _ = self.actor.sample(o)
        return a.cpu().numpy()

    # ====================
    # Update
    # ====================
    def update(self, batch: TorchBuffer):
        o = batch.obs_list
        a = batch.act_list
        r = batch.rew_list.view(-1, 1)
        on = batch.next_obs_list
        d = batch.done_list.view(-1, 1)

        self.update_critic(o, a, r, on, d)
        self.update_actor(o)
        self.critic.update_target_soft()

    def update_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> None:
        self.critic_optim.zero_grad()
        critic_loss = self.loss_critic(o, a, r, on, d, update_alpha=True)
        critic_loss.backward()
        self.critic_optim.step()

    def update_actor(self, o: torch.Tensor) -> None:
        self.actor_optim.zero_grad()
        actor_loss = self.loss_actor(o, update_alpha=True)
        actor_loss.backward()
        self.actor_optim.step()

    def update_alpha_cql(self, conservative_loss_detached: torch.Tensor) -> None:
        self.alpha_prime_optim.zero_grad()
        alpha_loss = self.loss_alpha_cql(conservative_loss_detached)
        alpha_loss.backward()
        self.alpha_prime_optim.step()

    def update_alpha_sac(self, log_prob_detached: torch.Tensor) -> None:
        self.alpha_optim.zero_grad()
        alpha_loss = self.loss_alpha_sac(log_prob_detached)
        alpha_loss.backward()
        self.alpha_optim.step()

    # ====================
    # Save and load
    # ====================
    def _save_dict(self) -> dict[str, torch.Tensor]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
            "actor_alpha_optimizer": self.alpha_optim.state_dict(),
            "critic_alpha_optimizer": self.alpha_prime_optim.state_dict(),
        }

    def _load_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optim.load_state_dict(state["actor_optimizer"])
        self.critic_optim.load_state_dict(state["critic_optimizer"])
        self.alpha_optim.load_state_dict(state["actor_alpha_optimizer"])
        self.alpha_prime_optim.load_state_dict(state["critic_alpha_optimizer"])

    # ====================
    # Critic loss
    # ====================
    def td_target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # soft_q_backup(SAC form): True 
        with torch.no_grad():
            an, log_prob = self.actor.sample(on)
            qn = self.critic.target_q_min(on, an)
            entropy = self.actor.pi.temp() * log_prob
            return r + self.critic.gamma * (qn - entropy) * (1.0 - d)

    def loss_td(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # TD3 Double Q 
        # loss_i = E_D[ (Qi-y)^2 ]
        # E_D[...]: input ...
        target = self.td_target(on, r, d)
        q1 = self.critic.q1(o, a)
        q2 = self.critic.q2(o, a)
        loss_td_1 = F.mse_loss(q1, target)
        loss_td_2 = F.mse_loss(q2, target)
        return torch.stack([loss_td_1, loss_td_2], dim=0)

    def loss_conservative(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        # L_CQL(H) = E_D[s]{log *              sum_a[exp(Q)] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         sum_a[p* exp(Q)/p] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         E_a[exp(Q-log(p))] } - E_D[s,a]{Q}
        #         ~= E_D[s]{log * 1/N * sum_N[exp(Q-log(p))] } - E_D[s,a]{Q} # sample approximation
        #         => logsumexp(Q-log(p)) - E_(s,a)~D[Q]  # 單步loss     
        #
        # E_D[s]: input o
        # E_D[s,a]: input o,a
        # CQL sample approximation: a ~ p(a) => Uniform/ pi(.|s)/ pi(.|s') 三種N次
        batch = o.shape[0]

        a_s, logp = self.actor.sample_n(o, self.critic.n_action_samples)
        an, logpn = self.actor.sample_n(on, self.critic.n_action_samples)
        ar, logpr = self.critic.sample_n(o)

        q = self.critic.eval_q_n(o, a_s).view(2, batch, self.critic.n_action_samples)
        qn = self.critic.eval_q_n(o, an).view(2, batch, self.critic.n_action_samples)
        qr = self.critic.eval_q_n(o, ar).view(2, batch, self.critic.n_action_samples)
        q_cat = torch.cat([q, qn, qr], dim=2)               # (2,B,3N)

        logp = logp.view(1, batch, self.critic.n_action_samples)   # (1,B,N)
        logpn = logpn.view(1, batch, self.critic.n_action_samples) # (1,B,N)
        logpr = logpr.view(1, batch, self.critic.n_action_samples) # (1,B,N)
        logp_cat = torch.cat([logp, logpn, logpr], dim=2)   # (1,B,3N)

        logsumexp = torch.logsumexp(q_cat - logp_cat, dim=2, keepdim=True)  # (2,B,1)
        data_q = torch.stack([self.critic.q1(o, a), self.critic.q2(o, a)], dim=0)
        return (logsumexp - data_q).mean(dim=[1, 2])        # (2), double Q

    def loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor, update_alpha: bool = True) -> torch.Tensor:
        # CQL loss: loss_td + alpha * loss_cql
        # TD3 double Q: sum_i[ loss_td + alpha * loss_cql ]
        # Lagrange 乘子: alpha
        loss_td = self.loss_td(o, a, r, on, d)             # (2,)
        loss_cql = self.loss_conservative(o, a, on)         # (2,)
        loss_cql = self.critic.conservative_weight * (loss_cql - self.critic.alpha_threshold)

        if update_alpha:
            self.update_alpha_cql(loss_cql.detach())
        loss_critic = loss_td.sum() + (self.critic.alpha() * loss_cql).sum() # d3rl design
        return loss_critic

    def loss_alpha_cql(self, conservative_loss_detached: torch.Tensor) -> torch.Tensor:
        # loss = -E[ alpha * L_cql ]
        # Lagrangian dual，若L_cql項長期偏大，則加強修改力度
        return -(self.critic.alpha() * conservative_loss_detached).mean()

    # ====================
    # Actor loss
    # ====================
    def loss_actor(self, o: torch.Tensor, update_alpha: bool = True) -> torch.Tensor:
        # TD3 Clipped Double Q: Q_min = min(Q1, Q2)
        # SAC loss = E_D[s],pi[a]{ temp * log_pi - Q_min }
        # E_D[s]: input o
        # E_pi[a]: sample a
        a, log_prob = self.actor.sample(o)
        if update_alpha:
            self.update_alpha_sac(log_prob.detach())
        q_t = self.critic.q_min(o, a)
        return (self.actor.pi.temp() * log_prob - q_t).mean()

    def loss_alpha_sac(self, log_prob_detached: torch.Tensor) -> torch.Tensor:
        # loss = -E[ alpha * (log_pi - target_entropy) ]
        # SAC設計，若log_prob長期偏大，則加強修改力度
        with torch.no_grad():
            target_alpha = log_prob_detached - self.act_size
        return -(self.actor.pi.temp() * target_alpha).mean()

