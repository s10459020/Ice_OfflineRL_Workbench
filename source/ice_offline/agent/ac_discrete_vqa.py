import numpy as np


class ActorCriticAgent:
    # ====================
    # Init
    # ====================
    def __init__(self, n_actions: int, obs_dim: int, gamma: float = 0.99, actor_alpha: float = 0.01, critic_alpha: float = 0.01, seed: int = 42) -> None:
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha
        self._rng = np.random.default_rng(seed)

        # Linear categorical actor parameters: logits = o @ pW + pb
        self.pW = np.zeros((self.obs_dim, self.n_actions), dtype=np.float32)
        self.pb = np.zeros(self.n_actions, dtype=np.float32)

        # Linear critic parameters: Q(o, .) = o @ qW + qb
        self.qW = np.zeros((self.obs_dim, self.n_actions), dtype=np.float32)
        self.qb = np.zeros(self.n_actions, dtype=np.float32)

        # Linear critic parameters: V(o) = o @ vw + vb
        self.vw = np.zeros(self.obs_dim, dtype=np.float32)
        self.vb = np.float32(0.0)

    # ====================
    # Public API
    # ====================
    def act(self, observation: np.ndarray) -> int:
        # a ~ Categorical(pi(a|s))
        o = np.asarray(observation, dtype=np.float32).reshape(-1)
        categorical = self._pi(o)
        return int(self._rng.choice(self.n_actions, p=categorical))

    def update(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool) -> None:
        o = np.asarray(observation, dtype=np.float32).reshape(-1)
        on = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        a = int(action)
        r = float(reward)
        d = bool(done)

        # nabla_J = E[sum_t[nabla_log_pi(a_t|s_t) * A_t]]
        #         = E[sum_t[nabla_J_step]]
        # p_theta <= theta + rate * nabla_J
        #         ~= theta + alpha * nabla_J_step
        nabla_pW, nabla_pb = self._nabla_j(o, a)
        self.pW += self.actor_alpha * nabla_pW
        self.pb += self.actor_alpha * nabla_pb

        # q_theta <= theta - alpha * nabla_loss_q
        nabla_qW, nabla_qb = self.nabla_loss_q(o, a, r, on, d)
        self.qW -= self.critic_alpha * nabla_qW
        self.qb -= self.critic_alpha * nabla_qb

        # v_theta <= theta - alpha * nabla_loss_v
        nabla_vw, nabla_vb = self.nabla_loss_v(o)
        self.vw -= self.critic_alpha * nabla_vw
        self.vb -= self.critic_alpha * nabla_vb

    # ====================
    # critic mathmatics
    # ====================
    def _Q(self, o: np.ndarray) -> np.ndarray:
        # Q(s, .) = s @ qW + qb
        return (o @ self.qW + self.qb).astype(np.float32)

    def _V(self, o: np.ndarray) -> np.float32:
        # V(s) = s @ vw + vb
        return np.float32(o @ self.vw + self.vb)

    def _A(self, o: np.ndarray, a: int) -> np.float32:
        # A(s, a) = Q(s, a) - V(s)
        q = np.float32(self._Q(o)[a])
        v = self._V(o)
        return np.float32(q - v)

    def _target_q(self, r: float, on: np.ndarray, d: bool) -> np.float32:
        # Q(s, a) = E[r + gamma * V(sn)]  # bellman
        vn = np.float32(0.0) if d else self._V(on)
        target_q = np.float32(r + self.gamma * vn)
        return target_q

    def _target_v(self, o: np.ndarray) -> np.float32:
        # V(s) = sum_a[ pi(a|s)*  Q_theta(s,a) ]  # discrete
        pi = self._pi(o)  # (A,)
        q_all = self._Q(o)  # (A,)
        target_v = np.float32(np.sum(pi * q_all))
        return target_v

    def _nabla_Q(self, o: np.ndarray, a: int) -> tuple[np.ndarray, np.ndarray]:
        # d/d{theta}{Q(s,a)} = [d/d{qW}{Q(s)} , d/d{qb}{Q(s)}]_a
        #                    = [s, 1] * one_hot(a) 
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[a] = 1.0

        grad_qW = np.outer(o, one_hot)
        grad_qb = one_hot
        return grad_qW, grad_qb

    def _nabla_V(self, o: np.ndarray) -> tuple[np.ndarray, np.float32]:
        # d/d{theta}{V} = [d/d{qW}{v} , d/d{qb}{v}]
        #               = [s, 1]
        grad_vw = o
        grad_vb = np.float32(1.0)
        return grad_vw, grad_vb

    def nabla_loss_q(self, o: np.ndarray, a: int, r: float, on: np.ndarray, d: bool) -> tuple[np.ndarray, np.ndarray]:
        # Loss_Q = 1/2( Q - target_Q )^2
        # nabla_loss_Q = (Q - target_Q) * nabla_Q
        q = np.float32(self._Q(o)[a])
        target_q = self._target_q(r, on, d)
        grad_qW, grad_qb = self._nabla_Q(o, a)
        delta = q - target_q
        return delta * grad_qW, delta * grad_qb

    def nabla_loss_v(self, o: np.ndarray) -> tuple[np.ndarray, np.float32]:
        # Loss_V = 1/2( V - E_pi[Q] )^2
        # nabla_loss_V = (V - target_V) * nabla_V
        v = self._V(o)
        target_v = self._target_v(o)
        grad_vw, grad_vb = self._nabla_V(o)
        delta = v - target_v
        return delta * grad_vw, delta * grad_vb

    # ====================
    # actor mathmatics
    # ====================
    def _pi(self, o: np.ndarray) -> np.ndarray:
        # z = logits(s_t) = s_t @ pW + pb
        # pi(.|s_t) = Categorical(a; z)
        #           = softmax(z)
        #           = exp(z_a) / sum[exp(z_j)]
        logits = o @ self.pW + self.pb
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits)

    def _nabla_log_pi(self, o: np.ndarray, a: int) -> tuple[np.ndarray, np.ndarray]:
        # nabla_log_pi(a_t|s_t) = d/d{theta} {log pi(a_t|s_t)}
        #                       = d/d{z} {log pi(a_t|s_t)}                * d/d{theta} {z}
        #                       = d/d{z} {log(exp(z_i) / sum[exp(z_j)]) } * [d/d{pW} {z}, d/d{pb} {z}]
        #                       = d/d{z} {z_i - log(sum[exp(z_j)])}       * [s_t, 1]
        #                       = (one_hot(a_t) - softmax(z_i))           * [s_t, 1]
        #                       = (one_hot(a_t) - pi(.|s_t))              * [s_t, 1]
        #                       = [grad_pW, grad_pb]
        policy = self._pi(o)
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[a] = 1.0
        nabla_log_z = one_hot - policy

        grad_pW = np.outer(o, nabla_log_z)
        grad_pb = nabla_log_z
        return grad_pW, grad_pb

    def _nabla_j(self, o: np.ndarray, a: int) -> tuple[np.ndarray, np.ndarray]:
        # nabla_J = E{sum_t[nabla_log_pi * A(s,a) ]} 
        #    step = nabla_log_pi * A
        advantage = self._A(o, a)
        grad_pW, grad_pb = self._nabla_log_pi(o, a)
        return advantage * grad_pW, advantage * grad_pb

