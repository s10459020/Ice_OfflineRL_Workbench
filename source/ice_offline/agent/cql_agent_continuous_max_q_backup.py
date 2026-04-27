"""Conservative Q-Learning continuous agent (minimal fixed structure)."""

import math

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

    def forward(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(o)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h).clamp(self.min_logstd, self.max_logstd)
        return mean, logstd


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


class CQLAgentContinuousMaxQBackup:
    def __init__(self, obs_size: int, act_size: int, actor_learning_rate: float = 1e-4, critic_learning_rate: float = 3e-4, alpha_sac_learning_rate: float = 1e-4, alpha_cql_learning_rate: float = 1e-4, gamma: float = 0.99, tau: float = 0.005, initial_alpha_sac: float = 1.0, initial_alpha_cql: float = 1.0, alpha_threshold: float = 10.0, n_action_samples: int = 10):
        self.device = "cpu"
        self.act_size = act_size
        self.gamma = gamma
        self.tau = tau
        self.alpha_threshold = alpha_threshold
        self.n_action_samples = n_action_samples

        self.policy = _Pi(obs_size, act_size).to(self.device)
        self.q1 = _Q(obs_size, act_size).to(self.device)
        self.q2 = _Q(obs_size, act_size).to(self.device)
        self.targ_q1 = _Q(obs_size, act_size).to(self.device)
        self.targ_q2 = _Q(obs_size, act_size).to(self.device)
        self.targ_q1.load_state_dict(self.q1.state_dict())
        self.targ_q2.load_state_dict(self.q2.state_dict())

        self.log_alpha_sac = torch.nn.Parameter(
            torch.tensor([[math.log(initial_alpha_sac)]], dtype=torch.float32)
        )
        self.log_alpha_cql = torch.nn.Parameter(
            torch.tensor([[math.log(initial_alpha_cql)]], dtype=torch.float32)
        )

        self.alpha_sac_optim = _Adam(alpha_sac_learning_rate)([self.log_alpha_sac])
        self.alpha_cql_optim = _Adam(alpha_cql_learning_rate)([self.log_alpha_cql])
        self.actor_optim = _Adam(actor_learning_rate)(self.policy.parameters())
        self.critic_optim = _Adam(critic_learning_rate)(
            list(self.q1.parameters()) + list(self.q2.parameters())
        )

    # ====================
    # act
    # ====================
    def action_best_batch(self, obs_batch):
        o = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, _ = self.policy(o)
            action = torch.tanh(mean)
        return action.cpu().numpy()

    def action_best(self, obs):
        return self.action_best_batch([obs])[0]

    def action_sample_batch(self, obs_batch):
        o = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _ = self._sample_a(o)
        return action.cpu().numpy()

    def action_sample(self, obs):
        return self.action_sample_batch([obs])[0]

    # ====================
    # helper function
    # ====================
    def _sample_a(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 對 E_pi[a]{ ... } 的樣本近似，取得a與log pi(a)用於後續計算
        # target_Q: r + gamma* E_pi[a]{ targ_Q(s',a')_min - alpha* log_pi(a'|s') }
        mean, logstd = self.policy(o)
        dist = SquashedGaussianDistribution(loc=mean, std=logstd.exp())
        a, log_prob = dist.sample_with_log_prob()
        return a, log_prob

    def sample_a_n(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # E_pi[a]{ ... } 樣本近似的N個抽樣版本，回傳(B*N, A)格式的資料
        mean, logstd = self.policy(o)
        dist = SquashedGaussianDistribution(loc=mean, std=logstd.exp())
        a, log_prob = dist.sample_n_with_log_prob(self.n_action_samples)

        a = a.reshape(-1, a.shape[-1])                  # (B, N, A) => (B*N, A)
        log_prob = log_prob.reshape(-1, 1)              # (B, N, 1) => (B*N, 1)
        return a, log_prob

    def sample_rand_n(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        qq = self._QQ(o, a)                                    # (2, B*N, 1)
        return qq

    def _QQ(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        return torch.stack([q1, q2], dim=0)

    def _targ_QQ(self, on: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1 = self.targ_q1(on, a)
        q2 = self.targ_q2(on, a)
        return torch.stack([q1, q2], dim=0)

    # ====================
    # update
    # ====================
    def update(self, batch):
        o = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        a = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        r = torch.as_tensor(batch["rew"], dtype=torch.float32, device=self.device).view(-1, 1)
        on = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        d = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).view(-1, 1)

        # critic
        critic_loss = self._loss_critic(o, a, r, on, d)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor
        actor_action, actor_log_prob = self._sample_a(o)
        self._update_alpha_sac(actor_log_prob.detach())
        actor_loss = self._loss_actor(o, actor_action, actor_log_prob)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._update_target_soft()

    def _update_target_soft(self) -> None:
        # DDPG soft target: theta_target <= (1 - tau)theta_target + (tau)theta
        # TD3 double Q: theta_target_i, theta_i
        with torch.no_grad():
            for p_targ, p in zip(self.targ_q1.parameters(), self.q1.parameters()):
                p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p_targ, p in zip(self.targ_q2.parameters(), self.q2.parameters()):
                p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def _update_alpha_cql(self, conservative_loss_detached: torch.Tensor) -> None:
        # L_alpha = -E[ alpha * L_cql ]
        # Lagrangian dual，若L_cql項長期偏大，則加強修改力度
        self.alpha_cql_optim.zero_grad()
        alpha_cql_loss = -(self._alpha_cql() * conservative_loss_detached).mean()
        alpha_cql_loss.backward()
        self.alpha_cql_optim.step()

    def _update_alpha_sac(self, log_prob_detached: torch.Tensor) -> None:
        # L_temp = -E[ temp * (log_pi - target_entropy) ]
        # SAC設計，若log_prob長期偏大，則加強修改力度
        self.alpha_sac_optim.zero_grad()
        with torch.no_grad():
            target_alpha_sac = log_prob_detached - self.act_size
        alpha_sac_loss = -(self._alpha_sac() * target_alpha_sac).mean()
        alpha_sac_loss.backward()
        self.alpha_sac_optim.step()

    # ====================
    # critic mathmatics
    # ====================
    def _alpha_cql(self) -> torch.Tensor:
        return self.log_alpha_cql.exp().clamp(0.0, 1e6)

    def _alpha_sac(self) -> torch.Tensor:
        return self.log_alpha_sac.exp()

    def _target_q(self, on: torch.Tensor, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        #  AC target: r + gamma*            Q2* (1-done)
        # TD3 target: r + gamma*        Q2_min* (1-done)                    # Clipped Double Q
        # DQN target: r + gamma*   targ_Q2_min* (1-done)                    # target Q
        # SAC target: r + gamma* ( targ_Q2_min - alpha* log_pi2 )* (1-done) # maximum entropy
        with torch.no_grad():
            an, _ = self.sample_a_n(on)
            on_flat = on.repeat_interleave(self.n_action_samples, dim=0)
            qq = self._targ_QQ(on_flat, an).view(2, on.shape[0], self.n_action_samples, 1)
            next_q = qq.min(dim=0).values.max(dim=1).values
            target = next_q
            return r + self.gamma * target * (1.0 - d)

    def _loss_td(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # TD3 Double Q 
        # loss_i = E_D[ (Qi-y)^2 ]
        target = self._target_q(on, r, d)
        qq = self._QQ(o, a)
        loss_td_1 = F.mse_loss(qq[0], target)
        loss_td_2 = F.mse_loss(qq[1], target)
        return torch.stack([loss_td_1, loss_td_2], dim=0)

    def _loss_cql(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        # L_CQL(H) = E_D[s]{log *              sum_a[exp(Q)] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         sum_a[p* exp(Q)/p] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         E_a[exp(Q-log(p))] } - E_D[s,a]{Q}
        #         ~= E_D[s]{log * 1/N * sum_N[exp(Q-log(p))] } - E_D[s,a]{Q} # continuous sample
        #         => logsumexp(Q-log(p)) - E_(s,a)~D[Q]  # 單步loss
        #
        # CQL sample approximation: a ~ p(a) => Uniform/ pi(.|s)/ pi(.|s') 三種N次
        batch = o.shape[0]

        a_s, logp = self.sample_a_n(o)
        an, logpn = self.sample_a_n(on)
        ar, logpr = self.sample_rand_n(o)

        q = self.eval_q_n(o, a_s).view(2, batch, self.n_action_samples)
        qn = self.eval_q_n(o, an).view(2, batch, self.n_action_samples)
        qr = self.eval_q_n(o, ar).view(2, batch, self.n_action_samples)
        q_cat = torch.cat([q, qn, qr], dim=2)               # (2,B,3N)

        logp = logp.view(1, batch, self.n_action_samples)   # (1,B,N)
        logpn = logpn.view(1, batch, self.n_action_samples) # (1,B,N)
        logpr = logpr.view(1, batch, self.n_action_samples) # (1,B,N)
        logp_cat = torch.cat([logp, logpn, logpr], dim=2)   # (1,B,3N)

        logsumexp = torch.logsumexp(q_cat - logp_cat, dim=2, keepdim=True)  # (2,B,1)
        data_q = self._QQ(o, a)                             # (2,B,1)
        return (logsumexp - data_q).mean(dim=[1, 2])        # (2), double Q

    def _loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # CQL loss: loss_td + alpha * loss_cql
        # TD3 double Q: sum_i[ loss_td + alpha * loss_cql ]
        # Lagrange 乘子: alpha
        loss_td = self._loss_td(o, a, r, on, d)             # (2,)
        loss_cql = self._loss_cql(o, a, on)                 # (2,)
        loss_cql = 5.0 * (loss_cql - self.alpha_threshold)  # fix weight

        self._update_alpha_cql(loss_cql.detach())
        loss_critic = loss_td.sum() + (self._alpha_cql() * loss_cql).sum() # d3rl design
        return loss_critic

    # ====================
    # actor mathmatics
    # ====================
    def _loss_actor(self, o: torch.Tensor, a: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        # TD3 Clipped Double Q: Q_min = min(Q1, Q2)
        # SAC loss = E_D[s],pi[a]{ temp * log_pi - Q_min }
        q_t = self._QQ(o, a).min(dim=0).values
        loss_actor = (self._alpha_sac() * log_prob - q_t).mean()   # batch mean
        return loss_actor
