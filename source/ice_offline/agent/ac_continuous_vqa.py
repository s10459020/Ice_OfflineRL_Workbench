import numpy as np


class ActorCriticAgent:
    # ====================
    # Init
    # ====================
    def __init__(self, action_dim: int, obs_dim: int, gamma: float = 0.99, actor_alpha: float = 0.01, critic_alpha: float = 0.01, seed: int = 42,) -> None:
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha
        self._rng = np.random.default_rng(seed)

        # Linear Gaussian actor parameters: mean = observation @ pW + pb
        self.pW = np.zeros((self.obs_dim, self.action_dim), dtype=np.float32)
        self.pb = np.zeros(self.action_dim, dtype=np.float32)

        # Keep exploration scale fixed for now.
        self.std = np.ones(self.action_dim, dtype=np.float32)

        # Linear critic parameters: Q(s, a) = [s, a] @ qW + qb
        self.qW = np.zeros(self.obs_dim + self.action_dim, dtype=np.float32)
        self.qb = np.float32(0.0)

        # Linear critic parameters: V(s) = s @ vw + vb
        self.vw = np.zeros(self.obs_dim, dtype=np.float32)
        self.vb = np.float32(0.0)

    # ====================
    # Public API
    # ====================
    def act(self, observation: np.ndarray) -> np.ndarray:
        # a ~ Normal(mean(s), std^2)
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        mean, std = self._pi(obs_vector)
        return self._rng.normal(loc=mean, scale=std).astype(np.float32)

    def update(self, o: np.ndarray, a: np.ndarray, r: float, o_: np.ndarray, done: bool) -> None:
        obs_vector = np.asarray(o, dtype=np.float32).reshape(-1)
        act_vector = np.asarray(a, dtype=np.float32).reshape(-1)
        next_obs_vector = np.asarray(o_, dtype=np.float32).reshape(-1)

        # weight = A(s, a) = Q(s, a) - V(s)
        # pi_theta <= theta + alpha * nabla_log_pi(a|s) * weight
        #           = theta + alpha * nabla[pW, pb]     * weight
        weight = self._A(obs_vector, act_vector)
        grad_pW, grad_pb = self._nabla_log_pi(obs_vector, act_vector)
        self.pW += self.actor_alpha * (weight * grad_pW)
        self.pb += self.actor_alpha * (weight * grad_pb)

        # q_theta <= theta - alpha * nabla_Lq
        #          = theta - alpha * nabla[qW, qb]
        grad_qW, grad_qb = self._nabla_Lq(obs_vector, act_vector, float(r), next_obs_vector, bool(done))
        self.qW -= self.critic_alpha * grad_qW
        self.qb -= self.critic_alpha * grad_qb

        # v_theta <= theta - alpha * nabla_Lv
        #          = theta - alpha * nabla[vw, vb]
        grad_vw, grad_vb = self._nabla_Lv(obs_vector, float(r), next_obs_vector, bool(done))
        self.vw -= self.critic_alpha * grad_vw
        self.vb -= self.critic_alpha * grad_vb

    # ====================
    # critic mathmatics
    # ====================
    def _Q(self, obs_vector: np.ndarray, act_vector: np.ndarray) -> np.float32:
        # Q(s, a) = [s, a] @ qW + qb
        sa = np.concatenate([obs_vector, act_vector]).astype(np.float32)
        return np.float32(sa @ self.qW + self.qb)

    def _V(self, obs_vector: np.ndarray) -> np.float32:
        # V(s) = s @ vw + vb
        return np.float32(obs_vector @ self.vw + self.vb)

    def _A(self, obs_vector: np.ndarray, act_vector: np.ndarray) -> np.float32:
        # A(s, a) = Q(s, a) - V(s)
        q_t = self._Q(obs_vector, act_vector)
        v_t = self._V(obs_vector)
        return np.float32(q_t - v_t)

    def _delta_q(self, obs_vector: np.ndarray, act_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> np.float32:
        # Q(s, a) = E[r + grama * V(s')]
        #
        # delta_q = Q_target - Q
        #         = Q_target(s, a) - Q(s, a)
        #         = r + gamma * V(s') - Q(s, a)
        q_t = self._Q(obs_vector, act_vector)
        next_v = np.float32(0.0) if done else self._V(next_obs_vector)
        return np.float32(reward + self.gamma * next_v - q_t)

    def _delta_v(self, obs_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> np.float32:
        # V(s) = E[Q(s,a)]
        #      = E[r + grama*V(s')]
        #
        # delta_v = Q - V
        #         = r + gamma * V(s') - V(s)
        v_t = self._V(obs_vector)
        next_v = np.float32(0.0) if done else self._V(next_obs_vector)
        return np.float32(reward + self.gamma * next_v - v_t)

    def _nabla_Q(self, obs_vector: np.ndarray, act_vector: np.ndarray) -> tuple[np.ndarray, np.float32]:
        # d/d{theta}{Q(s,a)} = [d/d{qW}{Q(s,a)}, d/d{qb}{Q(s,a)}]
        #                    = [[s, a], 1]
        sa = np.concatenate([obs_vector, act_vector]).astype(np.float32)
        grad_qW = sa
        grad_qb = np.float32(1.0)
        return grad_qW, grad_qb

    def _nabla_V(self, obs_vector: np.ndarray) -> tuple[np.ndarray, np.float32]:
        # d/d{theta} {V} = [d/d{vw}{V}, d/d{vb}{V}]
        #                = [s, 1]
        grad_vw = obs_vector
        grad_vb = np.float32(1.0)
        return grad_vw, grad_vb

    def _nabla_Lq(self, obs_vector: np.ndarray, act_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> tuple[np.ndarray, np.float32]:
        # Loss Q = MSE(delta_q)
        #        = 1/2(r + grama*V(s') - Q(s, a))**2
        #
        # nabla_Lq(s, a) = d/d{theta} {Lq}
        #                = d/d{Q} {Lq}                                * d/d{theta} {Q(s, a)}
        #                = d/d{Q} 1/2{delta_q}**2                     * nabla_Q
        #                = delta_q * d/d{Q} {delta_q}                 * nabla_Q
        #                = delta_q * d/d{Q}{r + grama*V(s') - Q(s,a)} * nabla_Q
        #                = delta_q * -1                               * nabla_Q
        #                = -delta_q * [grad_qW, grad_qb]
        delta_q = self._delta_q(obs_vector, act_vector, reward, next_obs_vector, done)
        grad_qW, grad_qb = self._nabla_Q(obs_vector, act_vector)
        return -delta_q * grad_qW, -delta_q * grad_qb

    def _nabla_Lv(self, obs_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> tuple[np.ndarray, np.float32]:
        # Loss V = MSE(delta_v)
        #        = 1/2(r + grama*V(s') - V(s))**2
        #
        # nabla_Lv(s) = d/d{theta} {Lv}
        #             = d/d{V} {Lv}                              * d/d{theta} {V}
        #             = d/d{V} 1/2{delta_v}**2                   * nabla_V
        #             = delta_v * d/d{V} {delta_v}               * nabla_V
        #             = delta_v * d/d{V}{r + gamma*V(s') - V(s)} * nabla_V
        #             = delta_v * -1                             * nabla_V
        #             = -delta_v * [grad_vw, grad_vb]
        delta_v = self._delta_v(obs_vector, reward, next_obs_vector, done)
        grad_vw, grad_vb = self._nabla_V(obs_vector)
        return -delta_v * grad_vw, -delta_v * grad_vb

    # ====================
    # actor mathmatics
    # ====================
    def _mean(self, obs_vector: np.ndarray) -> np.ndarray:
        # mean(s) = s @ pW + pb
        return obs_vector @ self.pW + self.pb

    def _pi(self, obs_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # mean(s) = s @ pW + pb
        # var = 1 # fixed
        #
        # pi(.|s) = Normal(x; mean, var)
        mean = self._mean(obs_vector)
        std = self.std
        return mean, std

    def _nabla_log_pi(self, obs_vector: np.ndarray, act_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # nabla_log_pi(a|s) = d/d{theta} {log pi(a|s)}
        #                   = d/d{mean} { log pi(a|s) }                                                      * d/d{theta} {mean}
        #                   = d/d{mean} { log Normal(a; mean, var) }                                         * [d/d{pW} {mean}, d/d{pb} {mean}]
        #                   = d/d{mean} { log(1) - log_sqrt(2*pi*var)  +   log_exp(-(a-mean)**2 / (2*var)) } * [s, 1]
        #                   = d/d{mean} {      0 -  0.5*log(2*pi*var)  -           ((a-mean)**2 / (2*var)) } * [s, 1]
        #                   = d/d{mean} {        -  0.5*log(2*pi*var)} - d/d{mean} {(a-mean)**2 / (2*var)  } * [s, 1]
        #                   =                                          - d/d{mean} {(a-mean)**2}/ (2*var)    * [s, 1]
        #                   = -[-2(a - mean)] / (2*var) * [s, 1]
        #                   =    ((a - mean)  /    var) * [s, 1]
        #                   = [grad_pW, grad_pb]
        mean = self._mean(obs_vector)
        var = self.std ** 2

        nabla_mean_log_pi = (act_vector - mean) / var
        grad_pW = np.outer(obs_vector, nabla_mean_log_pi)
        grad_pb = nabla_mean_log_pi
        return grad_pW, grad_pb
