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

        # Linear critic parameters: Q(s, .) = s @ qW + qb
        self.qW = np.zeros((self.obs_dim, self.n_actions), dtype=np.float32)
        self.qb = np.zeros(self.n_actions, dtype=np.float32)

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

        # weight = A(s, a) = Q(s, a) - V(s)
        # pi_theta <= theta + alpha * nabla_log_pi(a|s) * weight
        #           = theta + alpha * nabla[pW, pb]     * weight
        weight = self._A(obs_vector, int(a))
        grad_pW, grad_pb = self._nabla_log_pi(obs_vector, int(a))
        self.pW += self.actor_alpha * (weight * grad_pW)
        self.pb += self.actor_alpha * (weight * grad_pb)

        # q_theta <= theta - alpha * nabla_L
        #          = theta - alpha * nabla[qW, qb]
        grad_qW, grad_qb = self._nabla_L(obs_vector, int(a), float(r), next_obs_vector, bool(done))
        self.qW -= self.critic_alpha * grad_qW
        self.qb -= self.critic_alpha * grad_qb

    # ====================
    # critic mathmatics
    # ====================
    def _Q(self, obs_vector: np.ndarray, action: int) -> np.float32:
        # Q(s, a) = [s @ qW + qb][a]
        return np.float32((obs_vector @ self.qW + self.qb)[action])

    def _Q_all(self, obs_vector: np.ndarray) -> np.ndarray:
        # Q(s, .) = s @ qW + qb
        return obs_vector @ self.qW + self.qb

    def _V(self, obs_vector: np.ndarray) -> np.float32:
        # V(s) = E_pi[Q(s, a)]
        policy = self._pi(obs_vector)
        q_values = self._Q_all(obs_vector)
        return np.float32(np.sum(policy * q_values))

    def _A(self, obs_vector: np.ndarray, action: int) -> np.float32:
        # A(s, a) = Q(s, a) - V(s)
        q_t = self._Q(obs_vector, action)
        v_t = self._V(obs_vector)
        return np.float32(q_t - v_t)

    def _delta_q(self, obs_vector: np.ndarray, action: int, reward: float, next_obs_vector: np.ndarray, done: bool) -> np.float32:
        # Q(s, a) = E[r + grama * V(s')]
        #
        # delta_q = Q_target - Q
        #         = Q_target(s, a) - Q(s, a)
        #         = r + gamma * V(s') - Q(s, a)
        q_t = self._Q(obs_vector, action)
        next_v = np.float32(0.0) if done else self._V(next_obs_vector)
        return np.float32(reward + self.gamma * next_v - q_t)

    def _nabla_Q(self, obs_vector: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        # d/d{theta} {Q(s,a)} = d/d{Q(s)}  {Q(s,a)} * [d/d{qW}{Q(s)}, d/d{qb}{Q(s)}]
        #                     = one_hot(a) * [s, 1]
        #                     = [grad_qW, grad_qb]
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[action] = 1.0

        grad_qW = np.outer(obs_vector, one_hot)
        grad_qb = one_hot
        return grad_qW, grad_qb

    def _nabla_L(self, obs_vector: np.ndarray, action: int, reward: float, next_obs_vector: np.ndarray, done: bool) -> tuple[np.ndarray, np.ndarray]:
        # Loss Q = MSE(delta_q)
        #        = 1/2(r + grama*V(s') - Q(s, a))**2
        #
        # nabla_L(s, a) = d/d{theta} {L}
        #               = d/d{Q} {L}                                * d/d{theta} {Q(s, a)}
        #               = d/d{Q} 1/2{delta_q}**2                    * nabla_Q
        #               = delta_q * d/d{Q} {delta_q}                * nabla_Q
        #               = delta_q * d/d{Q}{r + grama*V(s') - Q(s,a)} * nabla_Q
        #               = delta_q * -1                              * nabla_Q
        #               = -delta_q * [grad_qW, grad_qb]
        delta_q = self._delta_q(obs_vector, action, reward, next_obs_vector, done)
        grad_qW, grad_qb = self._nabla_Q(obs_vector, action)
        return -delta_q * grad_qW, -delta_q * grad_qb

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
