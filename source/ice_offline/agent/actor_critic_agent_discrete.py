import numpy as np
from pathlib import Path

from ._agent_interface import model_path


class ActorCriticAgent:
    # ====================
    # Init
    # ====================
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        gamma: float = 0.99,
        actor_alpha: float = 0.01,
        critic_alpha: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha
        self._rng = np.random.default_rng(seed)

        # Linear categorical actor parameters: logits = observation @ pW + pb
        self.pW = np.zeros((self.obs_dim, self.n_actions), dtype=np.float32)
        self.pb = np.zeros(self.n_actions, dtype=np.float32)

        # Linear critic parameters: V(s) = s @ vw + vb
        self.vw = np.zeros(self.obs_dim, dtype=np.float32)
        self.vb = np.float32(0.0)

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

        # pi <=
        delta = self._delta(obs_vector, float(r), next_obs_vector, bool(done))
        pW, pb = self._nabla_log_pi(obs_vector, int(a))
        self.pW += self.actor_alpha * (delta * pW)
        self.pb += self.actor_alpha * (delta * pb)

        # V(s) <= V(s) - alpha * nabla_L
        #       = V(s) - alpha * nabla[vw, vb]
        nabla_vw, nabla_vb = self._nabla_L(obs_vector, r, next_obs_vector, done)
        self.vw -= self.critic_alpha * nabla_vw
        self.vb -= self.critic_alpha * nabla_vb

    # ====================
    # Persistence
    # ====================
    def save(self, model_id: str | Path, step: int) -> Path:
        path = model_path(model_id, step, ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            n_actions=np.asarray(self.n_actions, dtype=np.int32),
            obs_dim=np.asarray(self.obs_dim, dtype=np.int32),
            gamma=np.asarray(self.gamma, dtype=np.float32),
            actor_alpha=np.asarray(self.actor_alpha, dtype=np.float32),
            critic_alpha=np.asarray(self.critic_alpha, dtype=np.float32),
            pW=self.pW,
            pb=self.pb,
            vw=self.vw,
            vb=np.asarray(self.vb, dtype=np.float32),
        )
        return path

    @classmethod
    def load(cls, model_id: str | Path, step: int) -> "ActorCriticAgent":
        payload = np.load(model_path(model_id, step, ".npz"))

        agent = cls(
            n_actions=int(payload["n_actions"]),
            obs_dim=int(payload["obs_dim"]),
            gamma=float(payload["gamma"]),
            actor_alpha=float(payload["actor_alpha"]),
            critic_alpha=float(payload["critic_alpha"]),
        )
        agent.pW = np.asarray(payload["pW"], dtype=np.float32)
        agent.pb = np.asarray(payload["pb"], dtype=np.float32)
        agent.vw = np.asarray(payload["vw"], dtype=np.float32)
        agent.vb = np.float32(payload["vb"])
        return agent

    # ====================
    # policy mathmatics
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
        #                       = [nabla_pW_log_pi, nabla_pb_log_pi]
        policy = self._pi(obs_vector)
        one_hot = np.zeros(self.n_actions, dtype=np.float32)
        one_hot[action] = 1.0
        nabla_log = one_hot - policy

        nabla_pW_log_pi = np.outer(obs_vector, nabla_log)
        nabla_pb_log_pi = nabla_log
        return nabla_pW_log_pi, nabla_pb_log_pi

    # ====================
    # critic mathmatics
    # ====================
    def _delta(self, obs_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> np.float32:
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        value = self._V(obs_vector)
        next_value = np.float32(0.0) if done else self._V(next_obs_vector)
        return np.float32(reward + self.gamma * next_value - value)

    def _V(self, obs_vector: np.ndarray) -> np.float32:
        # V(s_t) = s_t @ vw + vb
        return np.float32(obs_vector @ self.vw + self.vb)

    def _nabla_V(self, obs_vector: np.ndarray) -> tuple[np.ndarray, np.float32]:
        # d/d{theta} {V} = [d/d{vw}{V}, d/d{vb}{V}]
        #                = [s, 1]
        grad_vw = obs_vector
        grad_vb = np.float32(1.0)
        return grad_vw, grad_vb
    
    def _nabla_L(self, obs_vector: np.ndarray, reward: float, next_obs_vector: np.ndarray, done: bool) -> tuple[np.ndarray, np.float32]:
        # L = 1/2(r + grama*V(s') - V(s))**2
        #
        # nabla_L(s) = d/d{theta} {L}       
        # nabla_L(s) = d/d{V} {L}                      * d/d{theta} {V}
        #            = d/d{V} {r + grama*V(s') - V(s)} * [s, 1]
        #            =       -(r + grama*V(s') - V(s)) * [s, 1]
        #            = -delta * [s, 1]
        delta = self._delta(obs_vector, reward, next_obs_vector, done)
        grad_vw, grad_vb = self._nabla_V(obs_vector)
        return -delta * grad_vw, -delta * grad_vb
