"""Conservative Q-Learning continuous agent (minimal fixed structure)."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from d3rlpy.models.torch.distributions import SquashedGaussianDistribution


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
        self.act_size = act_size
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(obs_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(256, act_size)
        self.logstd_head = torch.nn.Linear(256, act_size)
        self.min_logstd = -20.0
        self.max_logstd = 2.0
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float32))

    def forward(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(o)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(self.min_logstd, self.max_logstd)
        return mean, logstd

    def temp(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def dist(self, o: torch.Tensor) -> SquashedGaussianDistribution:
        mean, logstd = self(o)
        return SquashedGaussianDistribution(loc=mean, std=logstd.exp())

    def mode(self, o: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(o)[0])

    def sample(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 對 E_pi[a]{ ... } 的樣本近似，取得a與log pi(a)用於後續計算
        # target_Q: r + gamma* E_pi[a]{ targ_Q(s',a')_min - alpha* log_pi(a'|s') }
        dist = self.dist(o)
        a, log_prob = dist.sample_with_log_prob()
        return a, log_prob

    def sample_n(self, o: torch.Tensor, n_action_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        # E_pi[a]{ ... } 樣本近似的N個抽樣版本，回傳(B*N, A)格式的資料
        dist = self.dist(o)
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


class _QQ(torch.nn.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        gamma: float,
        tau: float,
        alpha_threshold: float,
        n_action_samples: int,
        device: str,
    ):
        super().__init__()
        self.act_size = act_size
        self.gamma = gamma
        self.tau = tau
        self.alpha_threshold = alpha_threshold
        self.n_action_samples = n_action_samples
        self.device = device

        self.q1 = _Q(obs_size, act_size)
        self.q2 = _Q(obs_size, act_size)
        self.targ_q1 = _Q(obs_size, act_size)
        self.targ_q2 = _Q(obs_size, act_size)
        self.targ_q1.load_state_dict(self.q1.state_dict())
        self.targ_q2.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float32))

    def qq(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        return torch.stack([q1, q2], dim=0)

    def qq_min(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.qq(o, a).min(dim=0).values

    def tqq_min(self, on: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.targ_q1(on, a)
        q2 = self.targ_q2(on, a)
        return torch.stack([q1, q2], dim=0).min(dim=0).values

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
        qq = self.qq(o, a)                                     # (2, B*N, 1)
        return qq

    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(0.0, 1e6)

    def update_target_soft(self) -> None:
        # DDPG soft target: theta_target <= (1 - tau)theta_target + (tau)theta
        # TD3 double Q: theta_target_i, theta_i
        with torch.no_grad():
            for p_targ, p in zip(self.targ_q1.parameters(), self.q1.parameters()):
                p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p_targ, p in zip(self.targ_q2.parameters(), self.q2.parameters()):
                p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)


class CQLAgentContinuous:
    def __init__(self, obs_size: int, act_size: int, actor_learning_rate: float = 1e-4, critic_learning_rate: float = 3e-4, actor_alpha_learning_rate: float = 1e-4, critic_alpha_learning_rate: float = 1e-4, gamma: float = 0.99, tau: float = 0.005, actor_initial_alpha: float = 1.0, critic_initial_alpha: float = 1.0, alpha_threshold: float = 10.0, n_action_samples: int = 10):
        self.device = "cpu"
        self.act_size = act_size

        self.policy = _Pi(obs_size=obs_size, act_size=act_size).to(self.device)
        self.critic = _QQ(
            obs_size=obs_size,
            act_size=act_size,
            gamma=gamma,
            tau=tau,
            alpha_threshold=alpha_threshold,
            n_action_samples=n_action_samples,
            device=self.device,
        ).to(self.device)
        self.policy.log_alpha.data.fill_(math.log(actor_initial_alpha))
        self.critic.log_alpha.data.fill_(math.log(critic_initial_alpha))
        self.actor_optim = _Adam(actor_learning_rate)(
            list(self.policy.hidden.parameters())
            + list(self.policy.mean_head.parameters())
            + list(self.policy.logstd_head.parameters())
        )
        self.critic_optim = _Adam(critic_learning_rate)(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters())
        )
        self.alpha_optim = _Adam(actor_alpha_learning_rate)([self.policy.log_alpha])
        self.alpha_prime_optim = _Adam(critic_alpha_learning_rate)([self.critic.log_alpha])

    # ====================
    # public API
    # ====================
    def act(self, observation, greedy: bool = True):
        observation_np = np.asarray(observation, dtype=np.float32)[None, :]
        o = torch.as_tensor(observation_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if greedy:
                action = self.policy.mode(o)
            else:
                action, _ = self.policy.sample(o)
        return action.cpu().numpy()[0]

    def act_batch(self, observation_batch, greedy: bool = True):
        o = torch.as_tensor(np.asarray(observation_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if greedy:
                a = self.policy.mode(o)
            else:
                a, _ = self.policy.sample(o)
        return a.cpu().numpy()

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

        # critic
        critic_loss = self._loss_critic(o, a, r, on, d)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor
        actor_loss = self._loss_actor(o)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic.update_target_soft()
        
    def update_alpha_cql(self, conservative_loss_detached: torch.Tensor) -> None:
        self.alpha_prime_optim.zero_grad()
        alpha_loss = self._loss_alpha_cql(conservative_loss_detached)
        alpha_loss.backward()
        self.alpha_prime_optim.step()

    def update_alpha_sac(self, log_prob_detached: torch.Tensor) -> None:
        self.alpha_optim.zero_grad()
        alpha_loss = self._loss_alpha_sac(log_prob_detached)
        alpha_loss.backward()
        self.alpha_optim.step()

    # ====================
    # critic mathmatics
    # ====================
    def _target(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        #  AC target: r + gamma*                             Q2  * (1-done)
        # TD3 target: r + gamma*                         Q2_min  * (1-done) # Clipped Double Q
        # DQN target: r + gamma*                    targ_Q2_min  * (1-done) # target Q
        # SAC target: r + gamma* ( targ_Q2_min - alpha* log_pi2 )* (1-done) # maximum entropy
        #
        # DQN form
        # max_q_backup(max Q): False
        # soft_q_backup(SAC form): False 
        with torch.no_grad():
            an = self.policy.mode(on)
            qn = self.critic.tqq_min(on, an)
            return r + self.critic.gamma * qn * (1.0 - d)

    def _loss_td(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # TD3 Double Q 
        # loss_i = E_D[ (Qi-y)^2 ]
        # E_D[...]: input ...
        target = self._target(on, r, d)
        qq = self.critic.qq(o, a)
        loss_td_1 = F.mse_loss(qq[0], target)
        loss_td_2 = F.mse_loss(qq[1], target)
        return torch.stack([loss_td_1, loss_td_2], dim=0)

    def _loss_cql(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
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

        a_s, logp = self.policy.sample_n(o, self.critic.n_action_samples)
        an, logpn = self.policy.sample_n(on, self.critic.n_action_samples)
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
        data_q = self.critic.qq(o, a)                             # (2,B,1)
        return (logsumexp - data_q).mean(dim=[1, 2])        # (2), double Q

    def _loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # CQL loss: loss_td + alpha * loss_cql
        # TD3 double Q: sum_i[ loss_td + alpha * loss_cql ]
        # Lagrange 乘子: alpha
        loss_td = self._loss_td(o, a, r, on, d)             # (2,)
        loss_cql = self._loss_cql(o, a, on)                 # (2,)
        loss_cql = 5.0 * (loss_cql - self.critic.alpha_threshold)  # fix weight

        self.update_alpha_cql(loss_cql.detach())
        loss_critic = loss_td.sum() + (self.critic.alpha() * loss_cql).sum() # d3rl design
        return loss_critic

    def _loss_alpha_cql(self, conservative_loss_detached: torch.Tensor) -> torch.Tensor:
        # loss = -E[ alpha * L_cql ]
        # Lagrangian dual，若L_cql項長期偏大，則加強修改力度
        return -(self.critic.alpha() * conservative_loss_detached).mean()

    # ====================
    # actor mathmatics
    # ====================
    def _loss_actor(self, o: torch.Tensor) -> torch.Tensor:
        # TD3 Clipped Double Q: Q_min = min(Q1, Q2)
        # SAC loss = E_D[s],pi[a]{ temp * log_pi - Q_min }
        # E_D[s]: input o
        # E_pi[a]: sample a
        a, log_prob = self.policy.sample(o)
        self.update_alpha_sac(log_prob.detach())
        q_t = self.critic.qq_min(o, a)
        return (self.policy.temp() * log_prob - q_t).mean()

    def _loss_alpha_sac(self, log_prob_detached: torch.Tensor) -> torch.Tensor:
        # loss = -E[ alpha * (log_pi - target_entropy) ]
        # SAC設計，若log_prob長期偏大，則加強修改力度
        with torch.no_grad():
            target_alpha = log_prob_detached - self.act_size
        return -(self.policy.temp() * target_alpha).mean()

