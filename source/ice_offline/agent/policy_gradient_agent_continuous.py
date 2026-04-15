import numpy as np
from pathlib import Path

from ._agent_interface import model_path


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

    def record_step(self, observation: np.ndarray, action: np.ndarray, reward: float) -> None:
        self.episode_observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.episode_actions.append(np.asarray(action, dtype=np.float32).copy())
        self.episode_rewards.append(float(reward))

    def clear_episode(self) -> None:
        self.episode_observations.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def update(self) -> None:
        grad_W, grad_b = self._estimate_nabla_J()
        self._gradient_ascent(grad_W, grad_b)
        self.clear_episode()

    # ====================
    # Persistence
    # ====================
    def save(self, model_id: str | Path, step: int) -> Path:
        path = model_path(model_id, step, ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            action_dim=np.asarray(self.action_dim, dtype=np.int32),
            obs_dim=np.asarray(self.obs_dim, dtype=np.int32),
            gamma=np.asarray(self.gamma, dtype=np.float32),
            alpha=np.asarray(self.alpha, dtype=np.float32),
            W=self.W,
            b=self.b,
            std=self.std,
        )
        return path

    @classmethod
    def load(cls, model_id: str | Path, step: int) -> "PolicyGradientAgent":
        payload = np.load(model_path(model_id, step, ".npz"))

        agent = cls(
            action_dim=int(payload["action_dim"]),
            obs_dim=int(payload["obs_dim"]),
            gamma=float(payload["gamma"]),
            alpha=float(payload["alpha"]),
        )
        agent.W = np.asarray(payload["W"], dtype=np.float32)
        agent.b = np.asarray(payload["b"], dtype=np.float32)
        agent.std = np.asarray(payload["std"], dtype=np.float32)
        return agent

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
    
    def _mean(self, obs_vector: np.ndarray) -> np.ndarray:
        # mu(s) = s @ W + b
        return obs_vector @ self.W + self.b

    def _pi(self, obs_vector: np.ndarray):
        # mean(s) = s @ W + b
        # var = 1 # fixed
        #
        # pi(.|s) = Normal(x; mean, var)
        #         = 1 / sqrt(2*pi*var) * exp( -(x-mean)**2 / 2*var )
        #         
        # return parameters for stochastic sample (Normal) and greedy action (mean)
        mean = self._mean(obs_vector)
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

        mean = self._mean(obs_vector)
        std = self.std
        var = std ** 2

        nabla_mean_log_pi = (act_vector - mean) / var
        nabla_W_log_pi = np.outer(obs_vector, nabla_mean_log_pi)
        nabla_b_log_pi = nabla_mean_log_pi
        return (nabla_W_log_pi, nabla_b_log_pi)

    def _estimate_nabla_J(self) -> tuple[np.ndarray, np.ndarray]:
        #  nabla_J = E { sum[ nabla_log_pi(a|s) * G_t ] }
        # estimate =     sum[ nabla_log_pi(a|s) * G_t ] 
        #          =     sum[            [W, b] * G_t ]
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        returns = self._G(self.episode_rewards)

        for observation, action, return_t in zip(
            self.episode_observations,
            self.episode_actions,
            returns,
        ):
            obs_vector = np.asarray(observation, dtype=np.float32).reshape(-1)
            act_vector = np.asarray(action, dtype=np.float32).reshape(-1)
            W, b = self._nabla_log_pi(obs_vector, act_vector)
            grad_W += W * return_t
            grad_b += b * return_t

        return grad_W, grad_b

    def _gradient_ascent(self, grad_W: np.ndarray, grad_b: np.ndarray) -> None:
        self.W += self.alpha * grad_W
        self.b += self.alpha * grad_b
