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
    # Public API
    # ====================
    
    def act(self, observation: np.ndarray) -> int:
        # a ~ Categorical(pi(a|s))
        policy = self._pi(observation)
        return int(self._rng.choice(self.n_actions, p=policy))
    
    def record_step(self, observation: np.ndarray, action: int, reward: float) -> None:
        self.episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.episode_actions.append(int(action))
        self.episode_rewards.append(float(reward))

    def clear_episode(self) -> None:
        self.episode_observations.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        
    def update(self) -> None:
        grad_W, grad_b = self._estimate_grad_estimate()
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
    
    def _pi(self, observation: np.ndarray, action: int | None = None) -> np.ndarray | np.float32:
        # logits = z(s_t) = s_t @ W + b
        #
        # pi( . |s_t) = softmax(z(s_t))
        # pi(a_i|s_t) = softmax_i(z(s_t))
        #             = exp(z(s_t)[a_i]) / sum[exp(z(s_t)[a_j])]
        #             = exp(z_i) / sum[exp(z_j)]
        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        logits = obs_vector @ self.W + self.b
        shifted_logits = logits - np.max(logits)

        # softmax_i = exp(z_i) / sum[exp(z_j)]
        exp_logits = np.exp(shifted_logits)
        policy = exp_logits / np.sum(exp_logits)

        if action is None:
            return policy
        return np.float32(policy[action])

    def _nabla_log_pi(self, observation: np.ndarray, action: int) -> tuple[np.ndarray, np.ndarray]:
        # nabla_log_pi(.|s_t) = d/d{theta} {log pi(a_t|s_t)} 
        #                     = d/d{z} {log pi(a_t|s_t)}                * d/d{theta} {z} 
        #                     = d/d{z} {log(exp(z_i) / sum[exp(z_j)]) } * [d/d{W} {z}, d/d{b} {z}]
        #                     = d/d{z} {z_i - log(sum[exp(z_j)])}       * [s_t, 1]
        #                     = (one_hot(a_t) - softmax(z_i))           * [s_t, 1]
        #                     = (one_hot(a_t) - pi(.|s_t))              * [s_t, 1]  
        #                     = [nabla_W_log_pi, nabla_b_log_pi]
        policy = self._pi(observation)
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[action] = 1.0
        nabla_log = one_hot - policy

        obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        nabla_W_log_pi = np.outer(obs_vector, nabla_log)
        nabla_b_log_pi = nabla_log 
        return (nabla_W_log_pi, nabla_b_log_pi)

    def _estimate_grad_estimate(self) -> tuple[np.ndarray, np.ndarray]:
        #    nabla_J = E { sum[ nabla_log_pi(a_t|s_t) * G_t ] }
        # estimate_J =     sum[ nabla_log_pi(a_t|s_t) * G_t ] 
        #            =     sum[                [W, b] * G_t ]

        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        returns = self._G(self.episode_rewards)

        for observation, action, return_t in zip(
            self.episode_observations,
            self.episode_actions,
            returns,
        ):
            obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            W, b = self._nabla_log_pi(obs_vector, action)
            grad_W += W * return_t
            grad_b += b * return_t

        return (grad_W, grad_b)

    def _gradient_ascent(self, grad_W: np.ndarray, grad_b: np.ndarray) -> None:
        self.W += self.alpha * grad_W
        self.b += self.alpha * grad_b
