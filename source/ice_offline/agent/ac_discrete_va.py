import numpy as np


class ActorCriticAgent:
    # ====================
    # Init
    # ====================
    def __init__(self, n_actions: int, obs_dim: int, gamma: float = 0.99, actor_alpha: float = 0.01, critic_alpha: float = 0.01, seed: int = 42,) -> None:
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha
        self._rng = np.random.default_rng(seed)

        # Linear categorical actor parameters: logits = observation @ pW + pb
        self.pW = np.zeros((self.obs_dim, self.n_actions), dtype=np.float32)
        self.pb = np.zeros(self.n_actions, dtype=np.float32)

        # Linear critic parameters: V(s) = s @ vw + vb
        self.vw = np.zeros(self.obs_dim, dtype=np.float32)
        self.vb = np.float32(0.0)

    # ====================
    # Public API
    # ====================
    def act(self, observation: np.ndarray) -> int:
        # a ~ Categorical(pi(a|s))
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        categorical = self._pi(obs_vector)
        return int(self._rng.choice(self.n_actions, p=categorical))

    def update(self, o: np.ndarray, a: int, r: float, o_: np.ndarray, done: bool) -> None:
        obs_vector = np.asarray(o, dtype=np.float32).reshape(-1)
        next_obs_vector = np.asarray(o_, dtype=np.float32).reshape(-1)

        # nabla_J = E[sum_t[nabla_log_pi(a_t|s_t) * A_t]]
        #         = E[sum_t[nabla_J_step]]
        # p_theta <= theta + rate * nabla_J
        #          = theta + (rate * T) * (1/T * nabla_J)
        #         ~= theta + alpha * nabla_J_step
        nabla_pW, nabla_pb = self._nabla_J_step(obs_vector, int(a), float(r), next_obs_vector, bool(done))
        self.pW += self.actor_alpha * nabla_pW
        self.pb += self.actor_alpha * nabla_pb

        # v_theta <= theta - alpha * nabla_L
        #          = theta - alpha * nabla[vw, vb]
        grad_vw, grad_vb = self._nabla_L(obs_vector, float(r), next_obs_vector, bool(done))
        self.vw -= self.critic_alpha * grad_vw
        self.vb -= self.critic_alpha * grad_vb

    # ====================
    # critic mathmatics
    # ====================
    def _Q(self, obs_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> np.float32:
        # Q(s, a) = r + gamma * V(s')
        next_v = np.float32(0.0) if done else self._V(next_obs_vector)
        return np.float32(reward + self.gamma * next_v)

    def _V(self, obs_vector: np.ndarray) -> np.float32:
        # V(s) = s @ vw + vb
        return np.float32(obs_vector @ self.vw + self.vb)

    def _A(self, obs_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> np.float32:
        # A(s, a) = Q(s, a) - V(s)
        q_t = self._Q(obs_vector, reward, next_obs_vector, done)
        v_t = self._V(obs_vector)
        return np.float32(q_t - v_t)

    def _delta_v(self, obs_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> np.float32:
        # V(s) = E[Q(s,a)]
        #      = E[r + grama*V(s')]
        #
        # delta_v = Q - V
        #         = r + gamma * V(s') - V(s)
        q_t = self._Q(obs_vector, reward, next_obs_vector, done)
        v_t = self._V(obs_vector)
        return np.float32(q_t - v_t)

    def _nabla_V(self, obs_vector: np.ndarray) -> tuple[np.ndarray, np.float32]:
        # d/d{theta} {V} = [d/d{vw}{V}, d/d{vb}{V}]
        #                = [s, 1]
        grad_vw = obs_vector
        grad_vb = np.float32(1.0)
        return grad_vw, grad_vb

    def _nabla_L(self, obs_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> tuple[np.ndarray, np.float32]:
        # Loss V = MSE(delta_v)
        #        = 1/2(r + grama*V(s') - V(s))**2
        #
        # nabla_L(s) = d/d{theta} {Lv}
        #            = d/d{V} {Lv}                              * d/d{theta} {V}
        #            = d/d{V} 1/2{delta_v}**2                   * nabla_V
        #            = delta_v * d/d{V} {delta_v}               * nabla_V
        #            = delta_v * d/d{V}{r + gamma*V(s') - V(s)} * nabla_V
        #            = delta_v * -1                             * nabla_V
        #            = -delta_v * [grad_vw, grad_vb]
        delta_v = self._delta_v(obs_vector, reward, next_obs_vector, done)
        grad_vw, grad_vb = self._nabla_V(obs_vector)
        return -delta_v * grad_vw, -delta_v * grad_vb

    # ====================
    # actor mathmatics
    # ====================
    def _pi(self, obs_vector: np.ndarray) -> np.ndarray:
        # z = logits(s_t) = s_t @ pW + pb
        #
        # pi(.|s_t) = Categorical(a; z)
        #           = softmax(z)
        #           = exp(z_a) / sum[exp(z_j)]
        logits = obs_vector @ self.pW + self.pb
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits)

    def _nabla_log_pi(self, obs_vector: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        # nabla_log_pi(a_t|s_t) = d/d{theta} {log pi(a_t|s_t)}
        #                       = d/d{z} {log pi(a_t|s_t)}                * d/d{theta} {z}
        #                       = d/d{z} {log(exp(z_i) / sum[exp(z_j)]) } * [d/d{pW} {z}, d/d{pb} {z}]
        #                       = d/d{z} {z_i - log(sum[exp(z_j)])}       * [s_t, 1]
        #                       = (one_hot(a_t) - softmax(z_i))           * [s_t, 1]
        #                       = (one_hot(a_t) - pi(.|s_t))              * [s_t, 1]
        #                       = [grad_pW, grad_pb]
        policy = self._pi(obs_vector)
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[action] = 1.0
        nabla_log_z = one_hot - policy

        grad_pW = np.outer(obs_vector, nabla_log_z)
        grad_pb = nabla_log_z
        return grad_pW, grad_pb

    def _nabla_J_step(self, obs_vector: np.ndarray, action: int, reward: float, next_obs_vector: np.ndarray, done: bool) -> tuple[np.ndarray, np.ndarray]:
        # J(theta) =   E_tau[R(tau)]
        #          = sum_tau[p(tau) * R(tau)]
        #
        # nabla_J(theta) = d/d{theta}{sum_tau[p * R]}
        #                = sum_tau[d/d{theta}{p * R}]
        #                = sum_tau[p * R * d/d{theta}{log p}]
        #                = E[ R * d/d{theta}{sum_t[log pi]}]
        #                = E[ R * sum_t[nabla_log_pi]]
        #                = E[ sum_t[G * nabla_log_pi]]
        #                = E[ sum_t[Q * nabla_log_pi]]
        #                = E[ sum_t[A * nabla_log_pi]]
        #
        # nabla_J_step = A * nabla_log_pi
        advantage = self._A(obs_vector, reward, next_obs_vector, done)
        grad_pW, grad_pb = self._nabla_log_pi(obs_vector, action)
        return advantage * grad_pW, advantage * grad_pb

