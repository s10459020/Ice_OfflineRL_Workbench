import numpy as np


class PolicyGradientAgent:
    # ====================
    # Init
    # ====================
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        gamma: float = 0.99,
        alpha: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.alpha = alpha
        self._rng = np.random.default_rng(seed)

        # Linear Gaussian policy parameters: mean = observation @ W + b
        self.W = np.zeros((self.obs_dim, self.action_dim), dtype=np.float32)
        self.b = np.zeros(self.action_dim, dtype=np.float32)

        # Keep exploration scale fixed for now; later we can learn it.
        self.log_std = np.zeros(self.action_dim, dtype=np.float32)

        self.episode_observations: list[np.ndarray] = []
        self.episode_actions: list[np.ndarray] = []
        self.episode_rewards: list[float] = []
    # ====================
    # Public API
    # ====================
    def act(self, observation: np.ndarray) -> np.ndarray:
        # a_t ~ pi(.|s_t) = Normal(mu(s_t), sigma^2)
        policy_mean, policy_std = self._pi(observation)
        return self._rng.normal(loc=policy_mean, scale=policy_std).astype(np.float32)

    def record_step(self, observation: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.episode_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.episode_rewards.append(float(reward))

    def clear_episode(self) -> None:
        self.episode_observations.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def update(self) -> None:
        # theta <- theta + alpha * nabla_theta J(theta)
        grad_W, grad_b = self._estimate_nabla_J()
        self._gradient_ascent(grad_W, grad_b)
        self.clear_episode()

    # ====================
    # mathmatics
    # ====================
    def _G(self, rewards: list[float]) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running_return = 0.0

        for idx in range(len(rewards) - 1, -1, -1):
            running_return = rewards[idx] + self.gamma * running_return
            returns[idx] = running_return

        return returns
    
    def _pi(self, observation: np.ndarray, action: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | np.float32:
        # pi(.|s_t) returns distribution parameters (mu, sigma).
        # pi(a_t|s_t) returns Gaussian density under diagonal covariance.
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        policy_mean = obs_vector @ self.W + self.b
        policy_std = np.exp(self.log_std)
        if action is None:
            return policy_mean, policy_std

        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        variance = policy_std ** 2
        norm = np.sqrt(2.0 * np.pi * variance)
        exp_term = np.exp(-0.5 * ((action_vector - policy_mean) ** 2) / variance)
        density = np.prod(exp_term / norm)
        return np.float32(density)

    def _nabla_log_pi(
        self,
        action: np.ndarray,
        policy_mean: np.ndarray,
        policy_std: np.ndarray,
    ) -> np.ndarray:
        # nabla_log_pi wrt mu:
        # log pi(a_t|s_t)
        #   = -0.5 * ( (a_t - mu)^2 / sigma^2 + 2 log(sigma) + log(2*pi) )
        # d/dmu log pi(a_t|s_t)
        #   = (a_t - mu) / sigma^2
        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        variance = policy_std ** 2
        return (action_vector - policy_mean) / variance

    def _estimate_nabla_J(self) -> tuple[np.ndarray, np.ndarray]:
        # nabla_J = E { sum[ nabla_log_pi(a_t|s_t) * G_t ] }
        # grad_J  =     sum[ nabla_log_pi(a_t|s_t) * G_t ]
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        returns = self._G(self.episode_rewards)

        for observation, action, return_t in zip(
            self.episode_observations,
            self.episode_actions,
            returns,
        ):
            obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            policy_mean, policy_std = self._pi(obs_vector)
            nabla_log_pi = self._nabla_log_pi(action, policy_mean, policy_std)
            grad_W += np.outer(obs_vector, nabla_log_pi) * return_t
            grad_b += nabla_log_pi * return_t

        return grad_W, grad_b

    def _gradient_ascent(self, grad_W: np.ndarray, grad_b: np.ndarray) -> None:
        self.W += self.alpha * grad_W
        self.b += self.alpha * grad_b
