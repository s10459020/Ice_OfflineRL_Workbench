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
    # Common
    # ====================
    def _compute_mean(self, observation: np.ndarray) -> np.ndarray:
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        return obs_vector @ self.W + self.b

    def _compute_std(self) -> np.ndarray:
        return np.exp(self.log_std)

    def _compute_policy(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # For the continuous action case, policy is a Normal distribution:
        #   policy(a|s) = Normal(policy_mean, policy_std)
        #
        # The parameters are produced as:
        #   policy_mean = observation @ W + b
        #   policy_std  = exp(log_std)
        #
        # So _compute_policy returns the parameters that define this Normal policy.
        policy_mean = self._compute_mean(observation)
        policy_std = self._compute_std()
        return policy_mean, policy_std

    def _compute_nabla_policy(
        self,
        action: np.ndarray,
        policy_mean: np.ndarray,
        policy_std: np.ndarray,
    ) -> np.ndarray:
        # Let policy be Normal(policy_mean, policy_std), with variance = policy_std^2.
        # For one action dimension:
        #   log policy(a|s)
        #   = -0.5 * ( (a - policy_mean)^2 / variance
        #              + 2 log(policy_std)
        #              + log(2 pi) )
        #
        # Differentiate with respect to policy_mean:
        #   d/d(policy_mean) log policy(a|s)
        #   = (a - policy_mean) / variance
        #
        # For all action dimensions together:
        #   nabla_policy = (action - policy_mean) / variance
        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        variance = policy_std ** 2
        return (action_vector - policy_mean) / variance

    # ====================
    # Act
    # ====================
    def act(self, observation: np.ndarray) -> np.ndarray:
        policy_mean, policy_std = self._compute_policy(observation)
        return self._sample_action(policy_mean, policy_std)
    
    def _sample_action(self, policy_mean: np.ndarray, policy_std: np.ndarray) -> np.ndarray:
        return self._rng.normal(loc=policy_mean, scale=policy_std).astype(np.float32)

    # ====================
    # Record
    # ====================
    def record_step(self, observation: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.episode_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.episode_rewards.append(float(reward))

    def clear_episode(self) -> None:
        self.episode_observations.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    # ====================
    # Update
    # ====================
    def update_episode(self) -> None:
        grad_W, grad_b = self._compute_episode_gradients()
        self._apply_gradients(grad_W, grad_b)
        self.clear_episode()

    def _compute_episode_gradients(self) -> tuple[np.ndarray, np.ndarray]:
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        returns = self._compute_returns(self.episode_rewards)

        for observation, action, return_t in zip(
            self.episode_observations,
            self.episode_actions,
            returns,
        ):
            obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            policy_mean, policy_std = self._compute_policy(obs_vector)
            nabla_policy = self._compute_nabla_policy(action, policy_mean, policy_std)
            grad_W += np.outer(obs_vector, nabla_policy) * return_t
            grad_b += nabla_policy * return_t

        return grad_W, grad_b
        
    def _compute_returns(self, rewards: list[float]) -> np.ndarray:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running_return = 0.0

        for idx in range(len(rewards) - 1, -1, -1):
            running_return = rewards[idx] + self.gamma * running_return
            returns[idx] = running_return

        return returns

    def _apply_gradients(self, grad_W: np.ndarray, grad_b: np.ndarray) -> None:
        self.W += self.alpha * grad_W
        self.b += self.alpha * grad_b
