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

    def forward(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hidden(obs_t)
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

    def forward(self, obs_t: torch.Tensor, act_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_t, act_t], dim=1)
        return self.network(x)


class CQLAgentContinuous:
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
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, _ = self.policy(obs_t)
            action = torch.tanh(mean)
        return action.cpu().numpy()

    def action_best(self, obs):
        return self.action_best_batch([obs])[0]

    def action_sample_batch(self, obs_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _ = self._sample_a(obs_t)
        return action.cpu().numpy()

    def action_sample(self, obs):
        return self.action_sample_batch([obs])[0]

    def _sample_a(self, o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 對 E_pi[a]{ ... } 的樣本近似，取得a與log pi(a)用於後續計算
        # target_Q: r + gamma* E_pi[a]{ targ_Q(s',a')_min - alpha* log_pi(a'|s') }
        mean, logstd = self.policy(o)
        dist = SquashedGaussianDistribution(loc=mean, std=logstd.exp())
        a, log_prob = dist.sample_with_log_prob()
        return a, log_prob

    def _sample_policy_q_values(self, policy_obs_t: torch.Tensor, value_obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 對 1/N sun_N[ ... ] 的樣本近似
        batch = policy_obs_t.shape[0]
        mean, logstd = self.policy(policy_obs_t)  # (B, A)
        dist = SquashedGaussianDistribution(loc=mean, std=logstd.exp())
        action, log_prob = dist.sample_n_with_log_prob(self.n_action_samples)  # (B,N,A), (B,N,1)

        repeated_value_obs = value_obs_t.unsqueeze(1).repeat(1, self.n_action_samples, 1)
        flat_value_obs = repeated_value_obs.reshape(-1, value_obs_t.shape[1])
        flat_action = action.reshape(-1, action.shape[-1])
        q1 = self.q1(flat_value_obs, flat_action).view(batch, self.n_action_samples, 1)
        q2 = self.q2(flat_value_obs, flat_action).view(batch, self.n_action_samples, 1)
        q = torch.cat([q1, q2], dim=2).permute(2, 0, 1)  # (2, B, N)
        lp = log_prob.view(1, batch, self.n_action_samples)  # (1,B,N)
        return q, lp

    def _sample_random_q_values(self, obs_t: torch.Tensor) -> tuple[torch.Tensor, float]:
        batch = obs_t.shape[0]
        repeated_obs = obs_t.unsqueeze(1).repeat(1, self.n_action_samples, 1)
        flat_obs = repeated_obs.reshape(-1, obs_t.shape[1])
        zero_tensor = torch.zeros(
            (batch * self.n_action_samples, self.act_size), device=self.device
        )
        random_actions = zero_tensor.uniform_(-1.0, 1.0)
        q1 = self.q1(flat_obs, random_actions).view(batch, self.n_action_samples, 1)
        q2 = self.q2(flat_obs, random_actions).view(batch, self.n_action_samples, 1)
        q = torch.cat([q1, q2], dim=2).permute(2, 0, 1)  # (2,B,N)
        random_log_prob = math.log(0.5**self.act_size)
        return q, random_log_prob

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

        self._soft_update_targets()

    def _soft_update_targets(self) -> None:
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
            an, next_log_prob = self._sample_a(on)
            next_q = torch.minimum(
                self.targ_q1(on, an),
                self.targ_q2(on, an),
            )
            target = next_q - self._alpha_sac() * next_log_prob
            return r + self.gamma * target * (1.0 - d)

    def _loss_td(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # TD3 Double Q 
        # loss_i = E_D[ (Qi-y)^2 ]
        # loss   = loss_1 + loss_2
        target = self._target_q(on, r, d)
        q1, q2 = self.q1(o, a), self.q2(o, a)
        return F.mse_loss(q1, target) + F.mse_loss(q2, target)

    def _loss_cql(self, o: torch.Tensor, a: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        # L_CQL(H) = E_D[s]{log *              sum_a[exp(Q)] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         sum_a[p* exp(Q)/p] } - E_D[s,a]{Q}
        #          = E_D[s]{log *         E_a[exp(Q-log(p))] } - E_D[s,a]{Q}
        #         ~= E_D[s]{log * 1/N * sum_N[exp(Q-log(p))] } - E_D[s,a]{Q} # continuous sample
        #         => logsumexp(Q-log(p)) - E_(s,a)~D[Q]  # 單步loss
        #
        # TD3 double Q =>
        policy_q_t, logp_t = self._sample_policy_q_values(o, o)
        policy_q_tp1, logp_tp1 = self._sample_policy_q_values(on, o)
        random_q, random_logp = self._sample_random_q_values(o)
        target_values = torch.cat(
            [
                policy_q_t - logp_t,
                policy_q_tp1 - logp_tp1,
                random_q - random_logp,
            ],
            dim=2,
        )  # (2,B,3N)
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)  # (2,B,1)
        data_q1 = self.q1(o, a)
        data_q2 = self.q2(o, a)
        data_q = torch.cat([data_q1, data_q2], dim=1).T.unsqueeze(-1)  # (2,B,1)
        return (logsumexp - data_q).mean(dim=[1, 2])  # batch mean

    def _loss_critic(self, o: torch.Tensor, a: torch.Tensor, r: torch.Tensor, on: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # CQL loss = loss_td + alpha * loss_cql
        # Lagrange 乘子: alpha
        loss_td = self._loss_td(o, a, r, on, d)
        loss_cql = self._loss_cql(o, a, on)  # (2,)
        loss_cql = 5.0 * (loss_cql - self.alpha_threshold)

        self._update_alpha_cql(loss_cql.detach())
        loss_critic = loss_td + (self._alpha_cql() * loss_cql).sum()
        return loss_critic

    # ====================
    # actor mathmatics
    # ====================
    def _loss_actor(self, o: torch.Tensor, a: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        # TD3 Clipped Double Q: Q_min = min(Q1, Q2)
        # SAC loss = E_D[s],pi[a]{ temp * log_pi - Q_min }
        q_t = torch.minimum(self.q1(o, a), self.q2(o, a))
        loss_actor = (self._alpha_sac() * log_prob - q_t).mean()   # batch mean
        return loss_actor
