import numpy as np


class PolicyGradientAgent:
    # ====================
    # Init
    # ====================
    def __init__(self, action_dim: int, obs_dim: int, gamma: float = 0.99, alpha: float = 0.01, seed: int = 42,) -> None:
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.alpha = alpha
        self._rng = np.random.default_rng(seed)

        # Linear Gaussian policy parameters: mean = observation @ W + b
        self.W = np.zeros((self.obs_dim, self.action_dim), dtype=np.float32)
        self.b = np.zeros(self.action_dim, dtype=np.float32)

        # Keep exploration scale fixed for now.
        self.std = np.ones(self.action_dim, dtype=np.float32)

        self.episode_observations: list[np.ndarray] = []
        self.episode_actions: list[np.ndarray] = []
        self.episode_rewards: list[float] = []

    # ====================
    # Public API
    # ====================
    def act(self, observation: np.ndarray) -> np.ndarray:
        # a ~ pi(.|s) = Normal(mu(s), sigma^2)
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        mean, std = self._pi(obs_vector)
        return self._rng.normal(loc=mean, scale=std).astype(np.float32)

    def clear_episode(self) -> None:
        self.episode_observations.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def update(self, o: np.ndarray, a: np.ndarray, r: float, o_: np.ndarray, done: bool) -> None:
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
    def _record_step(self, observation: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.episode_actions.append(np.asarray(action, dtype=np.float32).copy())
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
    def _pi(self, obs_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # mean(s) = s @ W + b
        # var = 1 # fixed
        #
        # pi(.|s) = Normal(x; mean, var)
        mean = obs_vector @ self.W + self.b
        std = self.std
        return mean, std

    def _nabla_log_pi(self, obs_vector: np.ndarray, act_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # nabla_log_pi(a|s) = d/d{theta} {log pi(a|s)}
        #                   = d/d{mean} { log pi(a|s) }                                                      * d/d{theta} {mean}
        #                   = d/d{mean} { log Normal(a; mean, var) }                                         * [d/d{W} {mean}, d/d{b} {mean}]
        #                   = d/d{mean} { log(1) - log_sqrt(2*pi*var)  +   log_exp(-(a-mean)**2 / (2*var)) } * [s, 1]
        #                   = d/d{mean} {      0 -  0.5*log(2*pi*var)  -           ((a-mean)**2 / (2*var)) } * [s, 1]
        #                   = d/d{mean} {        -  0.5*log(2*pi*var)} - d/d{mean} {(a-mean)**2 / (2*var)  } * [s, 1]
        #                   =                                          - d/d{mean} {(a-mean)**2}/ (2*var)    * [s, 1]
        #                   = -[-2(a - mean)] / (2*var) * [s, 1]
        #                   =    ((a - mean)  /    var) * [s, 1]
        #                   = [nabla_W_log_pi, nabla_b_log_pi]
        mean, std = self._pi(obs_vector)
        var = std ** 2

        nabla_mean_log_pi = (act_vector - mean) / var
        nabla_W_log_pi = np.outer(obs_vector, nabla_mean_log_pi)
        nabla_b_log_pi = nabla_mean_log_pi
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
            act_vector = np.asarray(action, dtype=np.float32).reshape(-1)
            grad_W, grad_b = self._nabla_log_pi(obs_vector, act_vector)
            nabla_W += return_t * grad_W
            nabla_b += return_t * grad_b

        return nabla_W, nabla_b

