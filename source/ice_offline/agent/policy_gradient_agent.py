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
        return self._sample_action(observation)

    def record_step(self, observation: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.episode_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.episode_rewards.append(float(reward))

    def clear_episode(self) -> None:
        self.episode_observations.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    # def compute_episode_objective(self) -> np.float32:
    #     returns = self._compute_returns(self.episode_rewards)
    #     objective = 0.0
    #
    #     for observation, action, return_t in zip(
    #         self.episode_observations,
    #         self.episode_actions,
    #         returns,
    #     ):
    #         log_prob = self._compute_log_prob(observation, action)
    #         objective += log_prob * return_t
    #
    #     return np.float32(objective)
    #
    # def compute_episode_loss(self) -> np.float32:
    #     return np.float32(-self.compute_episode_objective())

    def _compute_episode_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        returns = self._compute_returns(self.episode_rewards)
        std = self._compute_std()
        variance = std ** 2

        for observation, action, return_t in zip(
            self.episode_observations,
            self.episode_actions,
            returns,
        ):
            obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
            mean = self._compute_mean(obs_vector)

            grad_log_prob_mean = (action_vector - mean) / variance
            grad_W += np.outer(obs_vector, grad_log_prob_mean) * return_t
            grad_b += grad_log_prob_mean * return_t

        return grad_W, grad_b

    def _apply_gradients(self, grad_W: np.ndarray, grad_b: np.ndarray) -> None:
        self.W += self.alpha * grad_W
        self.b += self.alpha * grad_b

    def update_episode(self) -> None:
        grad_W, grad_b = self._compute_episode_gradients()
        self._apply_gradients(grad_W, grad_b)
        self.clear_episode()

    # ====================
    # Policy Math
    # ====================
    def _compute_mean(self, observation: np.ndarray) -> np.ndarray:
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        return obs_vector @ self.W + self.b

    def _compute_std(self) -> np.ndarray:
        return np.exp(self.log_std)

    def _compute_log_prob(self, observation: np.ndarray, action: np.ndarray) -> np.float32:
        mean = self._compute_mean(observation)
        std = self._compute_std()
        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)

        # 1D Gaussian density:
        # p(a) = 1 / (sqrt(2 * pi) * std) * exp(-(a - mean)^2 / (2 * std^2))
        #
        # Taking log gives:
        # log p(a) = -0.5 * ( ((a - mean)^2 / std^2) + 2 * log(std) + log(2 * pi) ).
        
        log_prob = -0.5 * (
            ((action_vector - mean) ** 2) / std ** 2
            + 2.0 * self.log_std
            + np.log(2.0 * np.pi)
        )
        return np.float32(np.sum(log_prob))

    def _sample_action(self, observation: np.ndarray) -> np.ndarray:
        mean = self._compute_mean(observation)
        std = self._compute_std()
        return self._rng.normal(loc=mean, scale=std).astype(np.float32)

    # ====================
    # Episode Utilities
    # ====================
    def _compute_returns(self, rewards: list[float]) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running_return = 0.0

        for idx in range(len(rewards) - 1, -1, -1):
            running_return = rewards[idx] + self.gamma * running_return
            returns[idx] = running_return

        return returns
