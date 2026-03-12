from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np


QTableState = Any


class QTableAgent:
    """Tabular Q-learning agent (action selection + TD update)."""

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.q_table: dict[QTableState, np.ndarray] = {}
        self._rng = random.Random(seed)

    def _ensure_state(self, state: QTableState) -> np.ndarray:
        action_values = self.q_table.get(state)
        if action_values is None:
            action_values = np.zeros(self.n_actions, dtype=np.float32)
            self.q_table[state] = action_values
        return action_values

    def act(self, state: QTableState, *, greedy: bool = False) -> int:
        action_values = self._ensure_state(state)
        if not greedy and self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_actions)
        best_value = float(np.max(action_values))
        best_actions = np.flatnonzero(action_values == best_value)
        if len(best_actions) == 1:
            return int(best_actions[0])
        return int(self._rng.choice(best_actions.tolist()))

    def update(
        self,
        state: QTableState,
        action: int,
        reward: float,
        next_state: QTableState,
        done: bool,
    ) -> None:
        q_s = self._ensure_state(state)
        q_next = self._ensure_state(next_state)
        best_next = float(np.max(q_next))
        td_target = float(reward) + (0.0 if done else self.gamma * best_next)
        td_error = td_target - float(q_s[action])
        q_s[action] += self.alpha * td_error

    def save(self, path: str | Path) -> Path:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q_table": self.q_table,
        }
        with model_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> "QTableAgent":
        model_path = Path(path)
        with model_path.open("rb") as f:
            payload = pickle.load(f)

        agent = cls(
            n_actions=int(payload["n_actions"]),
            alpha=float(payload["alpha"]),
            gamma=float(payload["gamma"]),
            epsilon=float(payload["epsilon"]),
        )
        q_table = payload["q_table"]
        agent.q_table = {
            state: np.asarray(values, dtype=np.float32)
            for state, values in q_table.items()
        }
        return agent
