import numpy as np


class PolicyGradientAgent:
    # ====================
    # Init
    # ====================
    def __init__(self, n_actions: int, obs_dim: int, gamma: float = 0.99, alpha: float = 0.01, seed: int = 42,) -> None:
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.alpha = alpha
        self._rng = np.random.default_rng(seed)

        # Linear categorical policy parameters: logits = observation @ W + b
        self.W = np.zeros((self.obs_dim, self.n_actions), dtype=np.float32)
        self.b = np.zeros(self.n_actions, dtype=np.float32)

        self.episode_observations: list[np.ndarray] = []
        self.episode_actions: list[int] = []
        self.episode_rewards: list[float] = []

    # ====================
    # Public API
    # ====================
    def act(self, observation: np.ndarray) -> int:
        # a ~ Categorical(pi(a|s))
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        categorical = self._pi(obs_vector)
        return int(self._rng.choice(self.n_actions, p=categorical))

    def clear_episode(self) -> None:
        self.episode_observations.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def update(self, o: np.ndarray, a: int, r: float, o_: np.ndarray, done: bool) -> None:
        self._record_step(o, a, r)

        if not done:
            return

        # p_theta <= theta + alpha * nabla_J
        nabla_W, nabla_b = self._nabla_J()
        self.W += self.alpha * nabla_W
        self.b += self.alpha * nabla_b
        self.clear_episode()

    # ====================
    # return mathmatics
    # ====================
    def _record_step(self, observation: np.ndarray, action: int, reward: float) -> None:
        self.episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.episode_actions.append(int(action))
        self.episode_rewards.append(float(reward))

    def _G(self) -> np.ndarray:
        # G_t = sum[(gamma ** k) * r_{t+k}]
        rewards = self.episode_rewards
        returns = np.zeros(len(rewards), dtype=np.float32)
        running_return = 0.0

        for idx in range(len(rewards) - 1, -1, -1):
            running_return = rewards[idx] + self.gamma * running_return
            returns[idx] = running_return

        return returns

    # ====================
    # actor mathmatics
    # ====================
    def _pi(self, obs_vector: np.ndarray) -> np.ndarray:
        # z = logits(s_t) = s_t @ W + b
        #
        # pi(.|s_t) = Categorical(a; z)
        #           = softmax(z)
        #           = exp(z_a) / sum[exp(z_j)]
        logits = obs_vector @ self.W + self.b
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits)

    def _nabla_log_pi(self, obs_vector: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        # nabla_log_pi(a_t|s_t) = d/d{theta} {log pi(a_t|s_t)}
        #                       = d/d{z} {log pi(a_t|s_t)}                * d/d{theta} {z}
        #                       = d/d{z} {log(exp(z_i) / sum[exp(z_j)]) } * [d/d{W} {z}, d/d{b} {z}]
        #                       = d/d{z} {z_i - log(sum[exp(z_j)])}       * [s_t, 1]
        #                       = (one_hot(a_t) - softmax(z_i))           * [s_t, 1]
        #                       = (one_hot(a_t) - pi(.|s_t))              * [s_t, 1]
        #                       = [nabla_W_log_pi, nabla_b_log_pi]
        policy = self._pi(obs_vector)
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[action] = 1.0
        nabla_log = one_hot - policy

        nabla_W_log_pi = np.outer(obs_vector, nabla_log)
        nabla_b_log_pi = nabla_log
        return nabla_W_log_pi, nabla_b_log_pi
    
    def _nabla_J(self) -> tuple[np.ndarray, np.ndarray]:
        # J(theta) =   E_tau[R(tau)]
        #          = sum_tau[p(tau) * R(tau)]
        #
        # nabla_J(theta) = d/d{theta}{sum_tau[p * R]}
        #                = sum_tau[d/d{theta}{p * R}]
        #                = sum_tau[p * R * d/d{theta}{log p}]
        #                = E[ R * d/d{theta}{sum_t[log pi]}]
        #                = E[ R * sum_t[nabla_log_pi]]
        #                = E[ sum_t[G * nabla_log_pi]]
        nabla_W = np.zeros_like(self.W)
        nabla_b = np.zeros_like(self.b)

        returns = self._G()
        for observation, action, return_t in zip(
            self.episode_observations,
            self.episode_actions,
            returns,
        ):
            obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            grad_W, grad_b = self._nabla_log_pi(obs_vector, action)
            nabla_W += return_t * grad_W
            nabla_b += return_t * grad_b

        return nabla_W, nabla_b

