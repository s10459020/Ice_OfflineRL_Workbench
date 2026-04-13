import numpy as np


class PolicyGradientAgent:
    # ====================
    # Init
    # ====================
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        gamma: float = 0.99,
        alpha: float = 0.01,
        seed: int = 42,
    ) -> None:
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
    # Common
    # ====================
    def _compute_logits(self, observation: np.ndarray) -> np.ndarray:
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        return obs_vector @ self.W + self.b

    def pi(
        self,
        observation: np.ndarray,
        action: int | None = None,
    ) -> np.ndarray | np.float32:
        # Let z be the action scores produced by _compute_logits(observation).
        # These scores are not probabilities yet.
        #
        # To turn scores into a policy, apply softmax:
        #   softmax(z)_k = exp(z_k) / sum_j exp(z_j)
        #
        # We define the discrete policy by this softmax output:
        #   pi(s, a_k) = softmax(z)_k
        #
        # So this function first converts logits z into the policy vector pi(s),
        # and if action is given, it returns the selected scalar pi(s, action).
        logits = self._compute_logits(observation)
        shifted_logits = logits - np.max(logits)
        exp_logits = np.exp(shifted_logits)
        policy = exp_logits / np.sum(exp_logits)
        if action is None:
            return policy
        return np.float32(policy[action])

    def nabla_pi(self, observation: np.ndarray, action: int) -> np.ndarray:
        # Let z be logits and policy_k = softmax(z)_k = exp(z_k) / sum_j exp(z_j).
        # For the sampled action a:
        #   log policy(a|s) = log softmax(z)_a
        #                   = z_a - log(sum_j exp(z_j))
        #
        # Differentiate with respect to z_k:
        #   d/dz_k log policy(a|s) = 1[a = k] - exp(z_k) / sum_j exp(z_j)
        #                          = 1[a = k] - policy_k
        #
        # Writing all k together as a vector gives:
        #   nabla_policy = one_hot(action) - policy
        policy = self.pi(observation)
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[action] = 1.0
        return one_hot - policy

    # ====================
    # Act
    # ====================
    def act(self, observation: np.ndarray) -> int:
        policy = self.pi(observation)
        return int(self._rng.choice(self.n_actions, p=policy))

    # ====================
    # Record
    # ====================
    def record_step(self, observation: np.ndarray, action: int, reward: float) -> None:
        self.episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.episode_actions.append(int(action))
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
            nabla_policy = self.nabla_pi(obs_vector, action)
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
